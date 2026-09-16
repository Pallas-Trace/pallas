#include "python_quanta.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

#include "pallas/pallas_read.h"

namespace {

template <typename T>
py::array_t<T> vector_view(std::vector<T>& v, py::handle owner) {
    return py::array_t<T>({static_cast<py::ssize_t>(v.size())}, {static_cast<py::ssize_t>(sizeof(T))}, v.data(), owner);
}

enum class QuantaMode {
    Fast,
    Balanced,
    ExactOld,
    ExactNew,
};

inline const char* quanta_mode_list() {
    return "fast, balanced, exact_old, exact_new, exact";
}

inline QuantaMode parse_quanta_mode(const std::string& mode) {
    if (mode == "fast")
        return QuantaMode::Fast;
    if (mode == "balanced")
        return QuantaMode::Balanced;
    if (mode == "exact_old")
        return QuantaMode::ExactOld;
    if (mode == "exact_new")
        return QuantaMode::ExactNew;
    if (mode == "exact")
        return QuantaMode::ExactNew;

    throw py::value_error("calc_quanta_base: invalid mode '" + mode + "'. Expected one of: " + std::string(quanta_mode_list()));
}

struct QuantaRow {
    uint64_t start_ns;
    uint64_t finish_ns;
    uint32_t thread_id;
    uint8_t token_type;
    uint64_t token_id;
    uint64_t excl_ns;
    float proportion;
};

constexpr uint8_t OTHER_TOKEN_TYPE = 255;
constexpr uint64_t OTHER_TOKEN_ID = 0;

bool row_less(const QuantaRow& a, const QuantaRow& b) {
    if (a.excl_ns != b.excl_ns) {
        return a.excl_ns > b.excl_ns;
    }
    if (a.token_type != b.token_type) {
        return a.token_type < b.token_type;
    }
    return a.token_id < b.token_id;
};

std::map<pallas::Token, uint64_t> snapshot_by_token(const pallas::Thread& thread, uint64_t start_ns, uint64_t end_ns, QuantaMode qmode) {
    std::map<pallas::Token, uint64_t> out;

    switch (qmode) {
    case QuantaMode::Fast: {
        auto snapshot = thread.getSnapshotViewFast(start_ns, end_ns);
        for (const auto& kv : snapshot) {
            const auto& key = kv.first;
            const pallas::Token& token = std::get<0>(key);
            const uint64_t excl = static_cast<uint64_t>(kv.second);
            if (excl != 0) {
                out[token] += excl;
            }
        }
        break;
    }

    case QuantaMode::Balanced: {
        auto snapshot = thread.getSnapshotView(start_ns, end_ns);
        for (const auto& kv : snapshot) {
            const auto& key = kv.first;
            const pallas::Token& token = std::get<0>(key);
            const uint64_t excl = static_cast<uint64_t>(kv.second);
            if (excl != 0) {
                out[token] += excl;
            }
        }
        break;
    }

    case QuantaMode::ExactOld: {
        break;
    }
    case QuantaMode::ExactNew: {
        break;
    }
    }

    return out;
};

void append_snapshot_rows(QuantaRes& res, uint32_t tid, uint64_t b0, uint64_t b1, const std::map<pallas::Token, uint64_t>& snapshot, int top_k) {
    uint64_t total = 0;
    std::vector<QuantaRow> rows;
    rows.reserve(snapshot.size());

    for (const auto& [token, excl] : snapshot) {
        if (excl == 0) {
            continue;
        }

        total += excl;
        rows.push_back(QuantaRow{
          .start_ns = b0,
          .finish_ns = b1,
          .thread_id = tid,
          .token_type = static_cast<uint8_t>(token.type),
          .token_id = token.id,
          .excl_ns = excl,
          .proportion = 0.0f,
        });
    }

    if (total == 0 || rows.empty()) {
        return;
    }

    if (top_k > 0 && rows.size() > static_cast<size_t>(top_k)) {
        auto middle = rows.begin() + static_cast<size_t>(top_k);
        std::partial_sort(rows.begin(), middle, rows.end(), row_less);

        uint64_t other_excl = 0;
        for (auto it = middle; it != rows.end(); ++it) {
            other_excl += it->excl_ns;
        }

        rows.resize(static_cast<size_t>(top_k));

        if (other_excl > 0) {
            rows.push_back(QuantaRow{
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

std::optional<pallas::Token> find_innermost_block(const pallas::Thread& thread, const pallas::ThreadReader& reader) {
    for (int i = reader.currentState.current_frame_index; i > 0; --i) {
        const auto& sequence_token = reader.getFrameInCallstack(i);
        if (sequence_token.type == pallas::TypeLoop) {
            continue;
        }

        auto* sequence = thread.getSequence(sequence_token);
        if (sequence == nullptr) {
            continue;
        }
        if (sequence->type != pallas::SEQUENCE_BLOCK) {
            continue;
        }

        return sequence_token;
    }

    return std::nullopt;
}

void accumulate_interval_into_bins(std::vector<std::map<pallas::Token, uint64_t>>& per_bin,
                                   const std::vector<uint64_t>& edges,
                                   const pallas::Token& token,
                                   uint64_t start_ns,
                                   uint64_t end_ns) {
    if (end_ns <= start_ns || edges.size() < 2) {
        return;
    }

    const uint64_t global_start = edges.front();
    const uint64_t global_end = edges.back();

    if (end_ns <= global_start || start_ns >= global_end) {
        return;
    }

    uint64_t s = std::max(start_ns, global_start);
    const uint64_t e = std::min(end_ns, global_end);
    if (e <= s) {
        return;
    }

    auto it = std::upper_bound(edges.begin(), edges.end(), s);
    size_t bin_idx = (it == edges.begin()) ? 0 : static_cast<size_t>((it - edges.begin()) - 1);

    while (bin_idx + 1 < edges.size() && s < e) {
        const uint64_t seg_end = std::min(e, edges[bin_idx + 1]);
        if (seg_end > s) {
            per_bin[bin_idx][token] += (seg_end - s);
        }
        s = seg_end;
        ++bin_idx;
    }
}

void calc_quanta_exact_new(const pallas::Thread& thread, const std::vector<uint64_t>& edges, int top_k, QuantaRes& res) {
    const size_t n_bins = edges.size() - 1;
    if (n_bins == 0) {
        return;
    }

    std::vector<std::map<pallas::Token, uint64_t>> per_bin(n_bins);

    pallas::ThreadReader reader(thread.archive, thread.id, PALLAS_READ_FLAG_UNROLL_ALL);
    auto current_token = reader.pollCurToken();
    if (!current_token.isValid()) {
        reader.archive = nullptr;
        return;
    }

    const uint64_t query_start = edges.front();
    const uint64_t query_end = edges.back();

    std::vector<pallas::Token> active_blocks;

    auto add_slice = [&](uint64_t slice_start, uint64_t slice_end) {
        if (active_blocks.empty()) {
            return;
        }

        const uint64_t clipped_start = std::max<uint64_t>(slice_start, query_start);
        const uint64_t clipped_end = std::min<uint64_t>(slice_end, query_end);

        if (clipped_start < clipped_end) {
            accumulate_interval_into_bins(per_bin, edges, active_blocks.back(), clipped_start, clipped_end);
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
            for (int i = reader.currentState.current_frame_index; i > 0; --i) {
                const pallas::Token& seq_token = reader.getFrameInCallstack(i);
                if (seq_token.type == pallas::TypeLoop) {
                    continue;
                }

                auto* seq = thread.getSequence(seq_token);
                if (seq == nullptr || seq->type != pallas::SEQUENCE_BLOCK) {
                    continue;
                }

                const bool begins_here = (current_token == seq->tokens.front());
                const bool ends_here = (current_token == seq->tokens.back());

                if (ends_here && !begins_here) {
                    pop_block(seq_token);
                }
            }

            for (int i = 1; i <= reader.currentState.current_frame_index; ++i) {
                const pallas::Token& seq_token = reader.getFrameInCallstack(i);
                if (seq_token.type == pallas::TypeLoop) {
                    continue;
                }

                auto* seq = thread.getSequence(seq_token);
                if (seq == nullptr || seq->type != pallas::SEQUENCE_BLOCK) {
                    continue;
                }

                const bool begins_here = (current_token == seq->tokens.front());
                const bool ends_here = (current_token == seq->tokens.back());

                if (begins_here && !ends_here) {
                    active_blocks.push_back(seq_token);
                }
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
        append_snapshot_rows(res, thread.id, edges[i], edges[i + 1], per_bin[i], top_k);
    }
}

void calc_quanta_exact_old(const pallas::Thread& thread, const std::vector<uint64_t>& edges, int top_k, QuantaRes& res) {
    const size_t n_bins = edges.size() - 1;
    for (size_t i = 0; i < n_bins; ++i) {
        const uint64_t b0 = edges[i];
        const uint64_t b1 = edges[i + 1];
        if (b1 <= b0) {
            continue;
        }

        std::map<pallas::Token, uint64_t> snapshot_out;
        auto snapshot = thread.getSnapshotViewExact(b0, b1);
        for (const auto& kv : snapshot) {
            const pallas::Token& token = kv.first;
            const uint64_t excl = static_cast<uint64_t>(kv.second);
            if (excl != 0) {
                snapshot_out[token] += excl;
            }
        }

        append_snapshot_rows(res, thread.id, b0, b1, snapshot_out, top_k);
    }
}

const pallas::Thread& find_thread_or_throw(pallas::GlobalArchive& trace, uint32_t thread_id) {
    for (auto& thread : trace.getThreadList()) {
        if (thread->id == thread_id) {
            return *thread;
        }
    }

    throw py::value_error("compare_exact_impls: thread id not found: " + std::to_string(thread_id));
}

struct TokenKey {
    uint8_t token_type;
    uint64_t token_id;

    bool operator<(const TokenKey& other) const {
        if (token_type != other.token_type)
            return token_type < other.token_type;
        return token_id < other.token_id;
    }

    bool operator==(const TokenKey& other) const { return token_type == other.token_type && token_id == other.token_id; }
};

struct BinKey {
    uint64_t start_ns;
    uint64_t finish_ns;

    bool operator<(const BinKey& other) const {
        if (start_ns != other.start_ns)
            return start_ns < other.start_ns;
        return finish_ns < other.finish_ns;
    }

    bool operator==(const BinKey& other) const { return start_ns == other.start_ns && finish_ns == other.finish_ns; }
};

struct DiffRow {
    uint64_t start_ns;
    uint64_t finish_ns;
    uint8_t token_type;
    uint64_t token_id;
    uint64_t old_excl_ns;
    uint64_t new_excl_ns;
    int64_t delta_ns;
    double old_prop;
    double new_prop;
    double abs_delta_prop;
};

std::string token_key_string(uint8_t token_type, uint64_t token_id) {
    return std::to_string(static_cast<unsigned>(token_type)) + ":" + std::to_string(token_id);
}

using TokenMap = std::map<TokenKey, uint64_t>;
using BinTokenMap = std::map<BinKey, TokenMap>;

BinTokenMap quanta_res_to_maps(const QuantaRes& res) {
    BinTokenMap out;
    const size_t n = res.size();
    for (size_t i = 0; i < n; ++i) {
        BinKey b{
          res.start_ns[i],
          res.finish_ns[i],
        };
        TokenKey t{
          res.token_type[i],
          res.token_id[i],
        };
        out[b][t] += res.excl_ns[i];
    }
    return out;
}

TokenMap aggregate_all_bins(const BinTokenMap& m) {
    TokenMap out;
    for (const auto& [bin, token_map] : m) {
        (void)bin;
        for (const auto& [tok, excl] : token_map) {
            out[tok] += excl;
        }
    }
    return out;
}

uint64_t token_map_total(const TokenMap& m) {
    uint64_t total = 0;
    for (const auto& [tok, excl] : m) {
        (void)tok;
        total += excl;
    }
    return total;
}

std::vector<DiffRow> diff_token_maps(const TokenMap& old_map, const TokenMap& new_map, uint64_t start_ns, uint64_t finish_ns) {
    std::map<TokenKey, std::pair<uint64_t, uint64_t>> merged;

    for (const auto& [tok, excl] : old_map) {
        merged[tok].first += excl;
    }
    for (const auto& [tok, excl] : new_map) {
        merged[tok].second += excl;
    }

    const uint64_t old_total = token_map_total(old_map);
    const uint64_t new_total = token_map_total(new_map);

    std::vector<DiffRow> rows;
    rows.reserve(merged.size());

    for (const auto& [tok, pair] : merged) {
        const uint64_t old_excl = pair.first;
        const uint64_t new_excl = pair.second;
        const double old_prop = old_total ? static_cast<double>(old_excl) / static_cast<double>(old_total) : 0.0;
        const double new_prop = new_total ? static_cast<double>(new_excl) / static_cast<double>(new_total) : 0.0;
        rows.push_back(DiffRow{
          start_ns,
          finish_ns,
          tok.token_type,
          tok.token_id,
          old_excl,
          new_excl,
          static_cast<int64_t>(new_excl) - static_cast<int64_t>(old_excl),
          old_prop,
          new_prop,
          std::abs(new_prop - old_prop),
        });
    }

    std::sort(rows.begin(), rows.end(), [](const DiffRow& a, const DiffRow& b) {
        if (a.abs_delta_prop != b.abs_delta_prop)
            return a.abs_delta_prop > b.abs_delta_prop;
        return std::llabs(a.delta_ns) > std::llabs(b.delta_ns);
    });

    return rows;
}

py::dict diff_row_to_dict(const DiffRow& r) {
    py::dict d;
    d["start_ns"] = py::int_(r.start_ns);
    d["finish_ns"] = py::int_(r.finish_ns);
    d["token_type"] = py::int_(r.token_type);
    d["token_id"] = py::int_(r.token_id);
    d["token_key"] = py::str(token_key_string(r.token_type, r.token_id));
    d["old_excl_ns"] = py::int_(r.old_excl_ns);
    d["new_excl_ns"] = py::int_(r.new_excl_ns);
    d["delta_ns"] = py::int_(r.delta_ns);
    d["old_prop"] = py::float_(r.old_prop);
    d["new_prop"] = py::float_(r.new_prop);
    d["abs_delta_prop"] = py::float_(r.abs_delta_prop);
    return d;
}

py::list top_rows_to_pylist(const std::vector<DiffRow>& rows, size_t limit) {
    py::list out;
    const size_t n = std::min(limit, rows.size());
    for (size_t i = 0; i < n; ++i) {
        out.append(diff_row_to_dict(rows[i]));
    }
    return out;
}

}  // namespace

py::dict compare_exact_impls(pallas::GlobalArchive& trace, uint32_t thread_id, py::array_t<uint64_t> bin_edges_ns, int top_k = -1, int top_n = 12) {
    auto bins = bin_edges_ns.unchecked<1>();
    if (bins.shape(0) < 2) {
        throw py::value_error("compare_exact_impls: bin_edges_ns must contain at least 2 values");
    }

    std::vector<uint64_t> edges;
    edges.reserve(static_cast<size_t>(bins.shape(0)));
    for (py::ssize_t i = 0; i < bins.shape(0); ++i) {
        edges.push_back(bins(i));
    }

    const pallas::Thread& thread = find_thread_or_throw(trace, thread_id);

    const size_t n_bins = edges.size() - 1;
    const size_t reserve_per_bin = (top_k > 0) ? static_cast<size_t>(top_k + 1) : static_cast<size_t>(8);

    QuantaRes old_res;
    QuantaRes new_res;
    old_res.reserve(std::max<size_t>(1, n_bins * reserve_per_bin));
    new_res.reserve(std::max<size_t>(1, n_bins * reserve_per_bin));

    calc_quanta_exact_old(thread, edges, top_k, old_res);
    calc_quanta_exact_new(thread, edges, top_k, new_res);

    const auto old_bins = quanta_res_to_maps(old_res);
    const auto new_bins = quanta_res_to_maps(new_res);

    const auto old_total_map = aggregate_all_bins(old_bins);
    const auto new_total_map = aggregate_all_bins(new_bins);

    const uint64_t old_total_ns = token_map_total(old_total_map);
    const uint64_t new_total_ns = token_map_total(new_total_map);

    auto overall_rows = diff_token_maps(old_total_map, new_total_map, edges.front(), edges.back());

    py::list per_bin;
    uint64_t matched_total_ns = 0;
    uint64_t union_total_ns = 0;

    for (size_t i = 0; i < n_bins; ++i) {
        BinKey bk{edges[i], edges[i + 1]};
        const auto old_it = old_bins.find(bk);
        const auto new_it = new_bins.find(bk);

        static const TokenMap empty_map{};
        const TokenMap& old_map = (old_it == old_bins.end()) ? empty_map : old_it->second;
        const TokenMap& new_map = (new_it == new_bins.end()) ? empty_map : new_it->second;

        const uint64_t old_bin_total = token_map_total(old_map);
        const uint64_t new_bin_total = token_map_total(new_map);

        uint64_t intersection = 0;
        uint64_t uni = 0;
        std::map<TokenKey, std::pair<uint64_t, uint64_t>> merged;
        for (const auto& [tok, excl] : old_map)
            merged[tok].first += excl;
        for (const auto& [tok, excl] : new_map)
            merged[tok].second += excl;

        for (const auto& [tok, pair] : merged) {
            (void)tok;
            intersection += std::min(pair.first, pair.second);
            uni += std::max(pair.first, pair.second);
        }

        matched_total_ns += intersection;
        union_total_ns += uni;

        auto rows = diff_token_maps(old_map, new_map, bk.start_ns, bk.finish_ns);

        py::dict item;
        item["bin_index"] = py::int_(static_cast<py::ssize_t>(i));
        item["start_ns"] = py::int_(bk.start_ns);
        item["finish_ns"] = py::int_(bk.finish_ns);
        item["old_total_ns"] = py::int_(old_bin_total);
        item["new_total_ns"] = py::int_(new_bin_total);
        item["matched_ns"] = py::int_(intersection);
        item["union_ns"] = py::int_(uni);
        item["overlap_ratio"] = py::float_(uni ? static_cast<double>(intersection) / static_cast<double>(uni) : 1.0);
        item["largest_diffs"] = top_rows_to_pylist(rows, static_cast<size_t>(top_n));
        per_bin.append(item);
    }

    py::dict summary;
    summary["thread_id"] = py::int_(thread_id);
    summary["n_bins"] = py::int_(static_cast<py::ssize_t>(n_bins));
    summary["top_k"] = py::int_(top_k);
    summary["top_n"] = py::int_(top_n);
    summary["old_total_ns"] = py::int_(old_total_ns);
    summary["new_total_ns"] = py::int_(new_total_ns);
    summary["matched_total_ns"] = py::int_(matched_total_ns);
    summary["union_total_ns"] = py::int_(union_total_ns);
    summary["global_overlap_ratio"] = py::float_(union_total_ns ? static_cast<double>(matched_total_ns) / static_cast<double>(union_total_ns) : 1.0);
    summary["old_row_count"] = py::int_(static_cast<py::ssize_t>(old_res.size()));
    summary["new_row_count"] = py::int_(static_cast<py::ssize_t>(new_res.size()));

    py::dict out;
    out["summary"] = summary;
    out["whole_window"] = top_rows_to_pylist(overall_rows, static_cast<size_t>(top_n));
    out["largest_diffs"] = top_rows_to_pylist(overall_rows, static_cast<size_t>(top_n));
    out["per_bin"] = per_bin;
    out["old"] = py::cast(std::move(old_res));
    out["new"] = py::cast(std::move(new_res));
    return out;
}

QuantaRes calc_quanta_base(pallas::GlobalArchive& trace, py::array_t<uint32_t> thread_ids, py::array_t<uint64_t> bin_edges_ns, const std::string& mode, int top_k = -1) {
    const QuantaMode qmode = parse_quanta_mode(mode);
    auto tids = thread_ids.unchecked<1>();
    auto bins = bin_edges_ns.unchecked<1>();

    if (bins.shape(0) < 2) {
        return {};
    }

    std::unordered_set<uint32_t> req_tids;
    req_tids.reserve(static_cast<size_t>(tids.shape(0)));
    for (py::ssize_t i = 0; i < tids.shape(0); ++i) {
        req_tids.insert(tids(i));
    }

    QuantaRes res;
    const size_t n_bins = static_cast<size_t>(bins.shape(0) - 1);
    const size_t reserve_per_bin = (top_k > 0) ? static_cast<size_t>(top_k) : static_cast<size_t>(4);
    res.reserve(std::max<size_t>(1, req_tids.size() * n_bins * reserve_per_bin));

    std::vector<uint64_t> edges;
    edges.reserve(static_cast<size_t>(bins.shape(0)));
    for (py::ssize_t i = 0; i < bins.shape(0); ++i) {
        edges.push_back(bins(i));
    }

    for (auto& thread : trace.getThreadList()) {
        const uint32_t tid = thread->id;
        if (req_tids.find(tid) == req_tids.end()) {
            continue;
        }

        switch (qmode) {
        case QuantaMode::ExactOld:
            calc_quanta_exact_old(*thread, edges, top_k, res);
            continue;

        case QuantaMode::ExactNew:
            calc_quanta_exact_new(*thread, edges, top_k, res);
            continue;

        case QuantaMode::Fast:
        case QuantaMode::Balanced:
            break;
        }

        for (size_t i = 0; i < n_bins; ++i) {
            const uint64_t b0 = edges[i];
            const uint64_t b1 = edges[i + 1];
            if (b1 <= b0) {
                continue;
            }

            auto snapshot = snapshot_by_token(*thread, b0, b1, qmode);
            append_snapshot_rows(res, tid, b0, b1, snapshot, top_k);
        }
    }

    return res;
}

void setup_quanta(py::module_& m, py::class_<pallas::GlobalArchive>& trace_cls) {
    auto quanta_res_cls = py::class_<QuantaRes>(m, "QuantaRes");

    quanta_res_cls.def(py::init<>())
      .def("__len__", [](const QuantaRes& self) { return self.size(); })
      .def_property_readonly("start_ns", [](QuantaRes& self) { return vector_view(self.start_ns, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("finish_ns", [](QuantaRes& self) { return vector_view(self.finish_ns, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("thread_id", [](QuantaRes& self) { return vector_view(self.thread_id, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("token_type", [](QuantaRes& self) { return vector_view(self.token_type, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("token_id", [](QuantaRes& self) { return vector_view(self.token_id, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("excl_ns", [](QuantaRes& self) { return vector_view(self.excl_ns, py::cast(&self, py::return_value_policy::reference)); })
      .def_property_readonly("proportion", [](QuantaRes& self) { return vector_view(self.proportion, py::cast(&self, py::return_value_policy::reference)); });

    m.attr("QUANTA_MODE_FAST") = py::str("fast");
    m.attr("QUANTA_MODE_BALANCED") = py::str("balanced");
    m.attr("QUANTA_MODE_EXACT") = py::str("exact");
    m.attr("QUANTA_MODE_EXACT_OLD") = py::str("exact_old");
    m.attr("QUANTA_MODE_EXACT_NEW") = py::str("exact_new");
    m.attr("QUANTA_MODE_DEFAULT") = py::str("fast");
    m.attr("QUANTA_MODES") = py::make_tuple("fast", "balanced", "exact", "exact_old", "exact_new");

    trace_cls.def(
      "calc_quanta_base",
      [](pallas::GlobalArchive& trace, py::array_t<uint32_t> thread_ids, py::array_t<uint64_t> bin_edges_ns, const std::string& mode, int top_k) {
          return calc_quanta_base(trace, std::move(thread_ids), std::move(bin_edges_ns), mode, top_k);
      },
      py::arg("thread_ids"), py::arg("bin_edges_ns"), py::arg("mode") = "fast", py::arg("top_k") = -1,
      R"pbdoc(
Compute per-bin exclusive-time quanta for selected threads.

Parameters
----------
thread_ids : numpy.ndarray[uint32]
    Thread ids to include.
bin_edges_ns : numpy.ndarray[uint64]
    Bin boundaries in nanoseconds, length N+1 for N bins.
mode : str, default "fast"
    Snapshot mode. One of:
      - "fast": approximate, optimized for interaction
      - "balanced": boundary-aware occurrence-based computation
      - "exact": reader-walk exact computation
top_k : int, default -1
    If > 0, keep only the top-k entries per (thread, bin) and merge the rest
    into an "other" bucket.

Returns
-------
QuantaRes
    Struct-of-arrays result with fields:
    start_ns, finish_ns, thread_id, token_type, token_id, excl_ns, proportion
)pbdoc");

    trace_cls.def(
      "compare_exact_impls",
      [](pallas::GlobalArchive& trace, uint32_t thread_id, py::array_t<uint64_t> bin_edges_ns, int top_k, int top_n) {
          return compare_exact_impls(trace, thread_id, std::move(bin_edges_ns), top_k, top_n);
      },
      py::arg("thread_id"), py::arg("bin_edges_ns"), py::arg("top_k") = -1, py::arg("top_n") = 12,
      R"pbdoc(
Compare old and new exact quanta implementations for one thread and one bin set.

Returns a dict with:
- summary: totals and overlap statistics
- largest_diffs: top token-level whole-window differences
- per_bin: top token-level differences for each bin
- old: raw QuantaRes from exact_old
- new: raw QuantaRes from exact_new
)pbdoc");
}
