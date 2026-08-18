/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "pallas/pallas.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"

#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_storage.h"

namespace {

using Clock = std::chrono::steady_clock;

enum class Family : size_t {
    EventTimestamps = 0,
    SequenceTimestamps = 1,
    SequenceDurations = 2,
    SequenceExclusiveDurations = 3,
    Count = 4,
};

struct FamilyStats {
    uint64_t vector_count = 0;
    uint64_t value_count = 0;
    uint64_t load_ns = 0;
    uint64_t at_ns = 0;
    uint64_t at_calls = 0;
    uint64_t operator_ns = 0;
    uint64_t operator_calls = 0;
};

struct BulkBenchmarkStats {
    uint64_t open_ns = 0;
    uint64_t checksum = 0;
    size_t thread_count = 0;
    std::array<FamilyStats, static_cast<size_t>(Family::Count)> families{};
};

struct ReplayBenchmarkStats {
    uint64_t open_ns = 0;
    uint64_t walk_ns = 0;
    uint64_t checksum = 0;
    uint64_t token_count = 0;
    uint64_t event_count = 0;
    uint64_t sequence_count = 0;
    uint64_t loop_count = 0;
    size_t thread_count = 0;
};

volatile uint64_t g_benchmark_sink = 0;

const char* family_name(Family family) {
    switch (family) {
        case Family::EventTimestamps:
            return "event_timestamps";
        case Family::SequenceTimestamps:
            return "sequence_timestamps";
        case Family::SequenceDurations:
            return "sequence_durations";
        case Family::SequenceExclusiveDurations:
            return "sequence_exclusive_durations";
        case Family::Count:
            return "unknown";
    }
    return "unknown";
}

uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

double divide_or_zero(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) {
        return 0.0;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

template <typename Fn>
void for_each_vector(pallas::GlobalArchive& trace, Fn&& fn) {
    for (auto* thread : trace.getThreadList()) {
        if (thread == nullptr) {
            continue;
        }

        for (size_t event_idx = 0; event_idx < thread->nb_events; ++event_idx) {
            auto& event = thread->events[event_idx];
            if (event.timestamps != nullptr) {
                fn(Family::EventTimestamps, event.timestamps);
            }
        }

        for (size_t sequence_idx = 0; sequence_idx < thread->nb_sequences; ++sequence_idx) {
            auto& sequence = thread->sequences[sequence_idx];
            if (sequence.timestamps != nullptr) {
                fn(Family::SequenceTimestamps, sequence.timestamps);
            }
            if (sequence.durations != nullptr) {
                fn(Family::SequenceDurations, sequence.durations);
            }
            if (sequence.exclusive_durations != nullptr) {
                fn(Family::SequenceExclusiveDurations, sequence.exclusive_durations);
            }
        }
    }
}

BulkBenchmarkStats run_bulk_benchmark(const std::filesystem::path& trace_path) {
    BulkBenchmarkStats stats{};

    const auto open_start = Clock::now();
    auto* trace = pallas_open_trace(trace_path.c_str());
    stats.open_ns = elapsed_ns(open_start);
    if (trace == nullptr) {
        throw std::runtime_error("Failed to open trace for bulk benchmark");
    }

    stats.thread_count = trace->getThreadList().size();

    for_each_vector(*trace, [&](Family family, pallas::LinkedVectorBase* lv) {
        auto& family_stats = stats.families[static_cast<size_t>(family)];
        family_stats.vector_count++;
        family_stats.value_count += static_cast<uint64_t>(lv->size());

        const auto load_start = Clock::now();
        lv->load_all();
        family_stats.load_ns += elapsed_ns(load_start);
    });

    for_each_vector(*trace, [&](Family family, pallas::LinkedVectorBase* lv) {
        auto& family_stats = stats.families[static_cast<size_t>(family)];
        uint64_t checksum = 0;
        const auto at_start = Clock::now();
        for (size_t idx = 0; idx < lv->size(); ++idx) {
            checksum += lv->at(idx);
        }
        family_stats.at_ns += elapsed_ns(at_start);
        family_stats.at_calls += static_cast<uint64_t>(lv->size());
        stats.checksum += checksum;
        g_benchmark_sink ^= checksum;
    });

    for_each_vector(*trace, [&](Family family, pallas::LinkedVectorBase* lv) {
        auto& family_stats = stats.families[static_cast<size_t>(family)];
        uint64_t checksum = 0;
        const auto operator_start = Clock::now();
        for (size_t idx = 0; idx < lv->size(); ++idx) {
            checksum += (*lv)[idx];
        }
        family_stats.operator_ns += elapsed_ns(operator_start);
        family_stats.operator_calls += static_cast<uint64_t>(lv->size());
        stats.checksum += checksum;
        g_benchmark_sink ^= checksum;
    });

    delete trace;
    return stats;
}

ReplayBenchmarkStats run_replay_benchmark(const std::filesystem::path& trace_path) {
    ReplayBenchmarkStats stats{};

    const auto open_start = Clock::now();
    auto* trace = pallas_open_trace(trace_path.c_str());
    stats.open_ns = elapsed_ns(open_start);
    if (trace == nullptr) {
        throw std::runtime_error("Failed to open trace for replay benchmark");
    }

    stats.thread_count = trace->getThreadList().size();

    uint64_t checksum = 0;
    {
        pallas::MultiThreadReader reader(*trace);
        const auto walk_start = Clock::now();
        for (auto token = reader.pollCurToken(); token != pallas::INVALID_TOKEN;
             token = reader.getNextToken()) {
            auto* current_reader = reader.current_reader;
            stats.token_count++;

            switch (token.type) {
                case pallas::TypeEvent: {
                    const auto occurrence = current_reader->getEventOccurrence(
                            token, current_reader->getCurrentTokenCount(token));
                    checksum += occurrence.timestamp;
                    checksum += static_cast<uint64_t>(occurrence.event->record);
                    stats.event_count++;
                    break;
                }
                case pallas::TypeSequence: {
                    checksum += static_cast<uint64_t>(token.id);
                    stats.sequence_count++;
                    break;
                }
                case pallas::TypeLoop: {
                    checksum += static_cast<uint64_t>(token.id);
                    stats.loop_count++;
                    break;
                }
                case pallas::TypeInvalid:
                    break;
            }
        }
        stats.walk_ns = elapsed_ns(walk_start);
    }
    stats.checksum = checksum;
    g_benchmark_sink ^= checksum;

    delete trace;
    return stats;
}

void write_scalar(std::ofstream& out, const std::string& key, uint64_t value) {
    out << key << '=' << value << '\n';
}

void write_scalar(std::ofstream& out, const std::string& key, double value) {
    out << key << '=' << std::fixed << std::setprecision(6) << value << '\n';
}

std::string csv_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

void write_bulk_csv(const std::filesystem::path& output_path,
                    const std::filesystem::path& trace_path,
                    const BulkBenchmarkStats& stats) {
    std::ofstream out(output_path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open bulk benchmark CSV: " + output_path.string());
    }

    out << "trace_file,thread_count,family,vector_count,value_count,open_trace_ns,materialize_ns,"
           "materialize_ns_per_value,at_call_count,at_total_ns,at_ns_per_call,index_call_count,"
           "index_total_ns,index_ns_per_call,checksum\n";

    uint64_t total_vectors = 0;
    uint64_t total_values = 0;
    uint64_t total_load_ns = 0;
    uint64_t total_at_ns = 0;
    uint64_t total_at_calls = 0;
    uint64_t total_index_ns = 0;
    uint64_t total_index_calls = 0;

    for (size_t idx = 0; idx < static_cast<size_t>(Family::Count); ++idx) {
        const auto family = static_cast<Family>(idx);
        const auto& family_stats = stats.families[idx];

        out << csv_escape(trace_path.string()) << ','
            << static_cast<uint64_t>(stats.thread_count) << ','
            << csv_escape(family_name(family)) << ','
            << family_stats.vector_count << ','
            << family_stats.value_count << ','
            << stats.open_ns << ','
            << family_stats.load_ns << ','
            << std::fixed << std::setprecision(6)
            << divide_or_zero(family_stats.load_ns, family_stats.value_count) << ','
            << family_stats.at_calls << ','
            << family_stats.at_ns << ','
            << divide_or_zero(family_stats.at_ns, family_stats.at_calls) << ','
            << family_stats.operator_calls << ','
            << family_stats.operator_ns << ','
            << divide_or_zero(family_stats.operator_ns, family_stats.operator_calls) << ','
            << stats.checksum << '\n';

        total_vectors += family_stats.vector_count;
        total_values += family_stats.value_count;
        total_load_ns += family_stats.load_ns;
        total_at_ns += family_stats.at_ns;
        total_at_calls += family_stats.at_calls;
        total_index_ns += family_stats.operator_ns;
        total_index_calls += family_stats.operator_calls;
    }

    out << csv_escape(trace_path.string()) << ','
        << static_cast<uint64_t>(stats.thread_count) << ','
        << csv_escape("all_families") << ','
        << total_vectors << ','
        << total_values << ','
        << stats.open_ns << ','
        << total_load_ns << ','
        << std::fixed << std::setprecision(6)
        << divide_or_zero(total_load_ns, total_values) << ','
        << total_at_calls << ','
        << total_at_ns << ','
        << divide_or_zero(total_at_ns, total_at_calls) << ','
        << total_index_calls << ','
        << total_index_ns << ','
        << divide_or_zero(total_index_ns, total_index_calls) << ','
        << stats.checksum << '\n';
}

void write_replay_csv(const std::filesystem::path& output_path,
                      const std::filesystem::path& trace_path,
                      const ReplayBenchmarkStats& stats) {
    std::ofstream out(output_path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open replay benchmark CSV: " + output_path.string());
    }

    out << "trace_file,thread_count,open_trace_ns,replay_traversal_ns,replay_total_ns,token_count,"
           "event_count,sequence_count,loop_count,replay_ns_per_token,checksum\n";
    out << csv_escape(trace_path.string()) << ','
        << static_cast<uint64_t>(stats.thread_count) << ','
        << stats.open_ns << ','
        << stats.walk_ns << ','
        << (stats.open_ns + stats.walk_ns) << ','
        << stats.token_count << ','
        << stats.event_count << ','
        << stats.sequence_count << ','
        << stats.loop_count << ','
        << std::fixed << std::setprecision(6)
        << divide_or_zero(stats.walk_ns, stats.token_count) << ','
        << stats.checksum << '\n';
}

void usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [--bulk|--replay|--both] <trace file>\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    bool run_bulk = false;
    bool run_replay = false;
    const char* trace_arg = nullptr;

    for (int idx = 1; idx < argc; ++idx) {
        if (std::strcmp(argv[idx], "--bulk") == 0) {
            run_bulk = true;
            run_replay = false;
        } else if (std::strcmp(argv[idx], "--replay") == 0) {
            run_bulk = false;
            run_replay = true;
        } else if (std::strcmp(argv[idx], "--both") == 0) {
            run_bulk = true;
            run_replay = true;
        } else if (std::strcmp(argv[idx], "-h") == 0 || std::strcmp(argv[idx], "--help") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            trace_arg = argv[idx];
        }
    }

    if (!run_bulk && !run_replay) {
        run_bulk = true;
        run_replay = true;
    }

    if (trace_arg == nullptr) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::filesystem::path trace_path = std::filesystem::absolute(trace_arg);
    const auto output_dir = trace_path.parent_path();
    const auto bulk_output_path = output_dir / "read_bulk_benchmark.csv";
    const auto replay_output_path = output_dir / "read_replay_benchmark.csv";

    try {
        BulkBenchmarkStats bulk_stats{};
        ReplayBenchmarkStats replay_stats{};

        if (run_bulk) {
            bulk_stats = run_bulk_benchmark(trace_path);
        }
        if (run_replay) {
            replay_stats = run_replay_benchmark(trace_path);
        }

        if (run_bulk) {
            write_bulk_csv(bulk_output_path, trace_path, bulk_stats);
        }
        if (run_replay) {
            write_replay_csv(replay_output_path, trace_path, replay_stats);
        }

        if (run_bulk) {
            std::cout << "Wrote bulk read benchmark to " << bulk_output_path << '\n';
        }
        if (run_replay) {
            std::cout << "Wrote replay read benchmark to " << replay_output_path << '\n';
        }
        std::cout << "Benchmark sink: " << g_benchmark_sink << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "pallas_read_benchmark failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
