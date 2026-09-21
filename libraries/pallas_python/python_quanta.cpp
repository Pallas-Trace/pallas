#include "python_quanta.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>

#include "pallas/pallas.h"
#include "pallas/pallas_read.h"

// shared file-local helpers
namespace {

// data transformers
template <typename T>
py::array_t<T> vector_view(std::vector<T>& v, py::handle owner) {
    return py::array_t<T>({static_cast<py::ssize_t>(v.size())}, {static_cast<py::ssize_t>(sizeof(T))}, v.data(), owner);
}

std::vector<uint64_t> parse_edges(const quanta::BinEdgesArg& bin_edges_ns) {
    auto bins = bin_edges_ns.unchecked<1>();
    if (bins.shape(0) < 2)
        throw py::value_error("bin_edges_ns must contain at least 2 values (N+1 edges for N bins)");
    std::vector<uint64_t> edges;
    edges.reserve(static_cast<size_t>(bins.shape(0)));
    for (py::ssize_t i = 0; i < bins.shape(0); ++i)
        edges.push_back(bins(i));
    if (!std::is_sorted(edges.begin(), edges.end()))
        throw py::value_error("bin_edges_ns must be non-decreasing");
    return edges;
}

std::vector<const pallas::Thread*> select_threads(pallas::GlobalArchive& trace, const quanta::ThreadIdsArg& thread_ids) {
    auto tids = thread_ids.unchecked<1>();
    std::unordered_set<uint32_t> requested;
    requested.reserve(static_cast<size_t>(tids.shape(0)));
    for (py::ssize_t i = 0; i < tids.shape(0); ++i)
        requested.insert(tids(i));

    std::vector<const pallas::Thread*> out;
    for (auto& thread : trace.getThreadList()) {
        if (requested.find(thread->id) != requested.end())
            out.push_back(thread);
    }
    return out;
}

}  // namespace

// quanta query api
namespace quanta {

// file-local internals
namespace {

Mode parse_mode(const std::string& mode) {
    if (mode == "fast")
        return Mode::Fast;
    if (mode == "balanced")
        return Mode::Balanced;
    if (mode == "exact")
        return Mode::Exact;
    throw py::value_error("calc_quanta_base: invalid mode '" + mode + "'. Valid options: " + std::string(MODE_LIST));
}

// defines row ordering
bool row_cmpr(const ResultRow& a, const ResultRow& b) {
    if (a.excl_ns != b.excl_ns) {
        return a.excl_ns > b.excl_ns;
    }
    if (a.token_type != b.token_type) {
        return a.token_type < b.token_type;
    }
    return a.token_id < b.token_id;
}

void append_result_rows(uint32_t tid, uint64_t b0, uint64_t b1, const std::map<pallas::Token, uint64_t>& totals, int top_k, Result& res) {
    uint64_t total = 0;
    std::vector<ResultRow> rows;
    rows.reserve(totals.size());

    for (const auto& [token, excl] : totals) {
        if (excl == 0) {
            continue;
        }

        total += excl;
        rows.push_back(ResultRow{
          .start_ns = b0,
          .finish_ns = b1,
          .thread_id = tid,
          .token_type = static_cast<uint8_t>(token.type),
          .token_id = token.id,
          .excl_ns = excl,
          .proportion = 0.0f,
        });
    }

    if (total == 0) {
        return;
    }

    if (top_k > 0 && rows.size() > static_cast<size_t>(top_k)) {
        auto middle = rows.begin() + static_cast<std::ptrdiff_t>(top_k);
        std::partial_sort(rows.begin(), middle, rows.end(), row_cmpr);

        uint64_t other_excl = 0;
        for (auto it = middle; it != rows.end(); ++it) {
            other_excl += it->excl_ns;
        }
        rows.resize(static_cast<size_t>(top_k));

        if (other_excl > 0) {
            rows.push_back(ResultRow{
              .start_ns = b0,
              .finish_ns = b1,
              .thread_id = tid,
              .token_type = OTHER_TOKEN_TYPE,
              .token_id = OTHER_TOKEN_ID,
              .excl_ns = other_excl,
              .proportion = 0.0f,
            });
        }
    }

    for (auto& row : rows) {
        row.proportion = static_cast<float>(static_cast<double>(row.excl_ns) / static_cast<double>(total));
    }

    for (const auto& row : rows) {
        res.start_ns.push_back(row.start_ns);
        res.finish_ns.push_back(row.finish_ns);
        res.thread_id.push_back(row.thread_id);
        res.token_type.push_back(row.token_type);
        res.token_id.push_back(row.token_id);
        res.excl_ns.push_back(row.excl_ns);
        res.proportion.push_back(row.proportion);
    }
}

void accumulate_bins(const std::vector<uint64_t>& edges, const pallas::Token& token, uint64_t start_ns, uint64_t end_ns, std::vector<TokenTotals>& per_bin_totals) {
    if (end_ns <= start_ns) {
        return;
    }

    const uint64_t global_start = edges.front();
    const uint64_t global_end = edges.back();

    if (end_ns <= global_start || start_ns >= global_end) {
        return;
    }

    uint64_t start = std::max(start_ns, global_start);
    const uint64_t end = std::min(end_ns, global_end);
    if (end <= start) {
        return;
    }

    auto it = std::upper_bound(edges.begin(), edges.end(), start);
    size_t bin_idx = (it == edges.begin()) ? 0 : static_cast<size_t>((it - edges.begin()) - 1);

    while (bin_idx + 1 < edges.size() && start < end) {
        const uint64_t seg_end = std::min(end, edges[bin_idx + 1]);
        if (seg_end > start) {
            per_bin_totals[bin_idx][token] += (seg_end - start);
        }
        start = seg_end;
        ++bin_idx;
    }
}

TokenTotals collect_token_totals(const pallas::Thread& thread, uint64_t start_ns, uint64_t end_ns, Mode mode) {
    TokenTotals out;

    auto accumulate = [&out](const auto& snapshot) {
        for (const auto& kv : snapshot) {
            const pallas::Token& token = std::get<0>(kv.first);
            const uint64_t excl = static_cast<uint64_t>(kv.second);
            if (excl != 0)
                out[token] += excl;
        }
    };

    if (mode == Mode::Fast)
        accumulate(thread.getSnapshotViewFast(start_ns, end_ns));
    else
        accumulate(thread.getSnapshotView(start_ns, end_ns));

    return out;
}

// NOTE: this can be replaced by a fixed getSnapshotViewExact in core lib
void collect_exact_totals(const pallas::Thread& thread, const std::vector<uint64_t>& edges, int top_k, Result& res) {
    const size_t n_bins = edges.size() - 1;

    std::vector<TokenTotals> per_bin(n_bins);

    pallas::ThreadReader reader(thread.archive, thread.id, PALLAS_READ_FLAG_UNROLL_ALL);
    auto current_token = reader.pollCurToken();
    if (!current_token.isValid()) {
        reader.archive = nullptr;
        return;
    }

    const uint64_t query_start = edges.front();
    const uint64_t query_end = edges.back();

    std::vector<pallas::Token> active_blocks;
    struct FrameBlock {
        pallas::Token token;
        const pallas::Sequence* seq;
    };
    std::vector<FrameBlock> frame_blocks;

    auto add_slice = [&](uint64_t slice_start, uint64_t slice_end) {
        if (active_blocks.empty()) {
            return;
        }

        const uint64_t clipped_start = std::max(slice_start, query_start);
        const uint64_t clipped_end = std::min(slice_end, query_end);
        if (clipped_start < clipped_end) {
            accumulate_bins(edges, active_blocks.back(), clipped_start, clipped_end, per_bin);
        }
    };

    auto pop_block = [&](const pallas::Token& tok) {
        if (!active_blocks.empty() && active_blocks.back() == tok) {
            active_blocks.pop_back();
            return;
        }
        auto it = std::find(active_blocks.rbegin(), active_blocks.rend(), tok);
        if (it != active_blocks.rend()) {
            active_blocks.erase(std::next(it).base());
        }
    };

    uint64_t prev_timestamp = static_cast<uint64_t>(reader.currentState.currentFrame->current_timestamp);

    while (current_token.isValid()) {
        const uint64_t current_timestamp = static_cast<uint64_t>(reader.currentState.currentFrame->current_timestamp);

        add_slice(prev_timestamp, current_timestamp);

        if (current_token.type == pallas::TypeEvent && reader.currentState.current_frame_index > 0) {
            frame_blocks.clear();

            for (int i = 1; i <= reader.currentState.current_frame_index; ++i) {
                const pallas::Token& seq_token = reader.getFrameInCallstack(i);
                if (seq_token.type == pallas::TypeLoop) {
                    continue;
                }

                auto* seq = thread.getSequence(seq_token);
                if (seq == nullptr || seq->type != pallas::SEQUENCE_BLOCK) {
                    continue;
                }

                frame_blocks.push_back({seq_token, seq});
            }

            for (auto it = frame_blocks.rbegin(); it != frame_blocks.rend(); ++it) {
                const bool begins_here = (current_token == it->seq->tokens.front());
                const bool ends_here = (current_token == it->seq->tokens.back());

                if (ends_here && !begins_here) {
                    pop_block(it->token);
                }
            }

            for (const auto& fb : frame_blocks) {
                const bool begins_here = (current_token == fb.seq->tokens.front());
                const bool ends_here = (current_token == fb.seq->tokens.back());

                if (begins_here && !ends_here)
                    active_blocks.push_back(fb.token);
            }
        }

        prev_timestamp = current_timestamp;
        if (current_timestamp >= query_end) {
            break;
        }
        current_token = reader.getNextToken();
    }

    add_slice(prev_timestamp, query_end);
    reader.archive = nullptr;

    for (size_t i = 0; i < n_bins; ++i) {
        append_result_rows(thread.id, edges[i], edges[i + 1], per_bin[i], top_k, res);
    }
}

}  // namespace

Result calc(pallas::GlobalArchive& trace, ThreadIdsArg thread_ids, BinEdgesArg bin_edges_ns, const std::string& mode, int top_k) {
    const Mode parsed_mode = parse_mode(mode);
    const std::vector<uint64_t> edges = parse_edges(bin_edges_ns);
    const auto threads = select_threads(trace, thread_ids);
    const size_t n_bins = edges.size() - 1;

    Result res;
    const size_t reserve_per_bin = (top_k > 0) ? static_cast<size_t>(top_k) : 4;
    res.reserve(std::max<size_t>(1, threads.size() * n_bins * reserve_per_bin));

    for (const pallas::Thread* thread : threads) {
        if (parsed_mode == Mode::Exact) {
            collect_exact_totals(*thread, edges, top_k, res);
            continue;
        }

        for (size_t i = 0; i < n_bins; ++i) {
            const uint64_t b0 = edges[i];
            const uint64_t b1 = edges[i + 1];
            if (b1 <= b0) {
                continue;
            }
            auto totals = collect_token_totals(*thread, b0, b1, parsed_mode);
            append_result_rows(thread->id, b0, b1, totals, top_k, res);
        }
    }

    return res;
}

}  // namespace quanta

// python bindings
void setup_quanta(py::module_& m, py::class_<pallas::GlobalArchive>& trace_cls) {
    auto res_cls = py::class_<quanta::Result>(m, "QuantaRes");

    res_cls.def(py::init<>())
      .def("__len__", [](const quanta::Result& self) { return self.size(); })
      .def_property_readonly("start_ns", [](quanta::Result& self) { return vector_view(self.start_ns, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("finish_ns", [](quanta::Result& self) { return vector_view(self.finish_ns, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("thread_id", [](quanta::Result& self) { return vector_view(self.thread_id, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("token_type", [](quanta::Result& self) { return vector_view(self.token_type, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("token_id", [](quanta::Result& self) { return vector_view(self.token_id, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("excl_ns", [](quanta::Result& self) { return vector_view(self.excl_ns, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("proportion", [](quanta::Result& self) { return vector_view(self.proportion, py::cast(&self, py::return_value_policy::reference)); });

    m.attr("QUANTA_MODE_FAST") = py::str("fast");
    m.attr("QUANTA_MODE_BALANCED") = py::str("balanced");
    m.attr("QUANTA_MODE_EXACT") = py::str("exact");
    m.attr("QUANTA_MODE_DEFAULT") = py::str("fast");
    m.attr("QUANTA_MODES") = py::make_tuple("fast", "balanced", "exact");

    trace_cls.def(
      "calc_binned_proportions",
      [](pallas::GlobalArchive& trace, quanta::ThreadIdsArg thread_ids, quanta::BinEdgesArg bin_edges_ns, const std::string& mode, int top_k) {
          return quanta::calc(trace, std::move(thread_ids), std::move(bin_edges_ns), mode, top_k);
      },
      py::arg("thread_ids"), py::arg("bin_edges_ns"), py::arg("mode") = "fast", py::arg("top_k") = -1,
      R"pbdoc(
Compute per-bin proportion of exclusive-time per token (function) for selected threads.

Parameters
----------
thread_ids : numpy.ndarray[uint32]
    Thread ids to include.
bin_edges_ns : numpy.ndarray[uint64]
    Bin boundaries in nanoseconds, length N+1 for N bins.
mode : str, default "fast"
    Snapshot mode. One of:
      - "fast": very fast approximation; can be quite error prone around boundaries
      - "balanced": boundary-aware occurrence-based computation; good balance of speed and accuracy
      - "exact": reader-walk exact computation; potentially very compute-heavy
top_k : int, default -1
    If > 0, keep only the top-k entries per (thread, bin) and merge the rest
    into an "other" bucket.

Returns
-------
QuantaRes
    Struct-of-arrays result with fields:
    start_ns, finish_ns, thread_id, token_type, token_id, excl_ns, proportion
)pbdoc");
}
