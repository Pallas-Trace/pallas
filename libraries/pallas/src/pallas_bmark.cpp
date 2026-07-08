/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#ifdef BMARK

#include "pallas/utils/pallas_bmark.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <unordered_map>

#include "pallas/pallas_archive.h"

namespace pallas {

namespace {

using BenchmarkClock = std::chrono::steady_clock;

/** Thread-local benchmark counters updated directly by the current writer or reader thread. */
thread_local BmarkThreadStats g_bmark_thread_stats{};
/** Global mutex protecting archive-level benchmark aggregation shared across threads. */
std::mutex g_bmark_mutex;
/** Archive-keyed aggregate used to merge thread-local benchmark counters at archive scope. */
std::unordered_map<const Archive*, BmarkThreadStats> g_bmark_archive_stats;

uint64_t now_ns() {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    BenchmarkClock::now().time_since_epoch())
                    .count());
}

uint64_t abs_error(uint64_t exact_value, uint64_t observed_value) {
    return (exact_value >= observed_value) ? (exact_value - observed_value)
                                           : (observed_value - exact_value);
}

BmarkFamilyStats* family_stats(BmarkThreadStats& stats, BmarkFamily family) {
    switch (family) {
        case BmarkFamily::EventTimestamps:
            return &stats.event_timestamps;
        case BmarkFamily::SequenceTimestamps:
            return &stats.sequence_timestamps;
        case BmarkFamily::SequenceDurations:
            return &stats.sequence_durations;
        case BmarkFamily::SequenceExclusiveDurations:
            return &stats.sequence_exclusive_durations;
        case BmarkFamily::Unknown:
            return nullptr;
    }
    return nullptr;
}

const BmarkFamilyStats& family_stats(const BmarkThreadStats& stats, BmarkFamily family) {
    switch (family) {
        case BmarkFamily::EventTimestamps:
            return stats.event_timestamps;
        case BmarkFamily::SequenceTimestamps:
            return stats.sequence_timestamps;
        case BmarkFamily::SequenceDurations:
            return stats.sequence_durations;
        case BmarkFamily::SequenceExclusiveDurations:
            return stats.sequence_exclusive_durations;
        case BmarkFamily::Unknown:
            return stats.event_timestamps;
    }
    return stats.event_timestamps;
}

const char* family_name(BmarkFamily family) {
    switch (family) {
        case BmarkFamily::EventTimestamps:
            return "event_timestamps";
        case BmarkFamily::SequenceTimestamps:
            return "sequence_timestamps";
        case BmarkFamily::SequenceDurations:
            return "sequence_durations";
        case BmarkFamily::SequenceExclusiveDurations:
            return "sequence_exclusive_durations";
        case BmarkFamily::Unknown:
            return "unknown";
    }
    return "unknown";
}

void record_metric(BmarkFamily family, BmarkMetric metric, uint64_t elapsed_ns) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }

    switch (metric) {
        case BmarkMetric::Add:
            stats->add_ns += elapsed_ns;
            stats->add_calls++;
            return;
        case BmarkMetric::At:
            stats->at_ns += elapsed_ns;
            stats->at_calls++;
            return;
        case BmarkMetric::Operator:
            stats->operator_ns += elapsed_ns;
            stats->operator_calls++;
            return;
        case BmarkMetric::Write:
            stats->write_ns += elapsed_ns;
            return;
    }
}

}  // namespace

void BmarkFamilyStats::accumulate(const BmarkFamilyStats& other) {
    pre_raw_bytes += other.pre_raw_bytes;
    raw_bytes += other.raw_bytes;
    compressed_bytes += other.compressed_bytes;
    write_ns += other.write_ns;
    write_calls += other.write_calls;
    subarray_writes += other.subarray_writes;
    value_count += other.value_count;
    add_ns += other.add_ns;
    add_calls += other.add_calls;
    at_ns += other.at_ns;
    at_calls += other.at_calls;
    operator_ns += other.operator_ns;
    operator_calls += other.operator_calls;
    recent_value_hits += other.recent_value_hits;
    recent_value_misses += other.recent_value_misses;
    find_subarray_calls += other.find_subarray_calls;
    find_subarray_steps += other.find_subarray_steps;
    subarray_loads += other.subarray_loads;
    subarray_load_bytes += other.subarray_load_bytes;
    subarray_decompressed_bytes += other.subarray_decompressed_bytes;
    subarray_evictions += other.subarray_evictions;
    subarray_evicted_bytes += other.subarray_evicted_bytes;
    max_abs_error = std::max(max_abs_error, other.max_abs_error);
    sum_abs_error += other.sum_abs_error;
    sum_squared_abs_error += other.sum_squared_abs_error;
    nonzero_error_count += other.nonzero_error_count;
}

void BmarkThreadStats::clear() {
    event_timestamps = {};
    sequence_timestamps = {};
    sequence_durations = {};
    sequence_exclusive_durations = {};
}

void BmarkThreadStats::accumulate(const BmarkThreadStats& other) {
    event_timestamps.accumulate(other.event_timestamps);
    sequence_timestamps.accumulate(other.sequence_timestamps);
    sequence_durations.accumulate(other.sequence_durations);
    sequence_exclusive_durations.accumulate(other.sequence_exclusive_durations);
}

BmarkScopedTimer::BmarkScopedTimer(BmarkFamily family, BmarkMetric metric)
    : family(family), metric(metric), start_ns(now_ns()) {}

BmarkScopedTimer::~BmarkScopedTimer() {
    if (family == BmarkFamily::Unknown) {
        return;
    }
    record_metric(family, metric, now_ns() - start_ns);
}

void bmark_note_write_call(BmarkFamily family, uint64_t logical_value_count) {
    static_cast<void>(logical_value_count);
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }
    stats->write_calls++;
}

void bmark_note_subarray_write(BmarkFamily family,
                               uint64_t pre_raw_bytes,
                               uint64_t raw_bytes,
                               uint64_t compressed_bytes) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }
    stats->pre_raw_bytes += pre_raw_bytes;
    stats->raw_bytes += raw_bytes;
    stats->compressed_bytes += compressed_bytes;
    stats->subarray_writes++;
}

void bmark_note_recent_value_lookup(BmarkFamily family, bool hit) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }
    if (hit) {
        stats->recent_value_hits++;
        return;
    }
    stats->recent_value_misses++;
}

void bmark_note_find_subarray(BmarkFamily family, uint64_t steps) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }
    stats->find_subarray_calls++;
    stats->find_subarray_steps += steps;
}

void bmark_note_subarray_load(BmarkFamily family, uint64_t load_bytes, uint64_t decompressed_bytes) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }
    stats->subarray_loads++;
    stats->subarray_load_bytes += load_bytes;
    stats->subarray_decompressed_bytes += decompressed_bytes;
}

void bmark_note_subarray_evict(BmarkFamily family, uint64_t evicted_bytes) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr) {
        return;
    }
    stats->subarray_evictions++;
    stats->subarray_evicted_bytes += evicted_bytes;
}

void bmark_note_error_values(BmarkFamily family,
                             const uint64_t* exact_values,
                             const uint64_t* observed_values,
                             size_t value_count) {
    auto* stats = family_stats(g_bmark_thread_stats, family);
    if (stats == nullptr || exact_values == nullptr || observed_values == nullptr) {
        return;
    }

    stats->value_count += value_count;
    for (size_t idx = 0; idx < value_count; ++idx) {
        const uint64_t current_abs_error = abs_error(exact_values[idx], observed_values[idx]);
        stats->max_abs_error = std::max(stats->max_abs_error, current_abs_error);
        stats->sum_abs_error += current_abs_error;
        const double abs_error_value = static_cast<double>(current_abs_error);
        stats->sum_squared_abs_error += abs_error_value * abs_error_value;
        if (current_abs_error != 0) {
            stats->nonzero_error_count++;
        }
    }
}

void bmark_flush_thread_stats(Archive* archive) {
    if (archive == nullptr) {
        g_bmark_thread_stats.clear();
        return;
    }

    std::lock_guard<std::mutex> guard(g_bmark_mutex);
    g_bmark_archive_stats[archive].accumulate(g_bmark_thread_stats);
    g_bmark_thread_stats.clear();
}

void bmark_reset_thread_stats() {
    g_bmark_thread_stats.clear();
}

void bmark_write_archive_csv(const Archive* archive, const char* root_path) {
    if (archive == nullptr || root_path == nullptr) {
        return;
    }

    BmarkThreadStats aggregate{};
    {
        std::lock_guard<std::mutex> guard(g_bmark_mutex);
        const auto found = g_bmark_archive_stats.find(archive);
        if (found != g_bmark_archive_stats.end()) {
            aggregate = found->second;
        }
    }

    const auto archive_dir =
            std::filesystem::path(root_path) / ("archive_" + std::to_string(archive->id));
    std::filesystem::create_directories(archive_dir);
    std::ofstream archive_out(archive_dir / "archive_benchmark.csv", std::ios::trunc);
    std::ofstream perf_out(archive_dir / "perf_benchmark.csv", std::ios::trunc);
    archive_out << std::fixed << std::setprecision(6);
    perf_out << std::fixed << std::setprecision(6);
    archive_out << "archive_id,family,pre_raw_bytes,raw_bytes,compressed_bytes,write_ns,write_calls,"
                   "subarray_writes,value_count,add_ns,add_calls,at_ns,at_calls,operator_ns,operator_calls,"
                   "max_abs_error,sum_abs_error,sum_squared_abs_error,nonzero_error_count\n";
    perf_out << "archive_id,family,value_count,recent_value_hits,recent_value_misses,"
                "find_subarray_calls,find_subarray_steps,subarray_loads,subarray_load_bytes,"
                "subarray_decompressed_bytes,subarray_evictions,subarray_evicted_bytes\n";

    const BmarkFamily families[] = {
            BmarkFamily::EventTimestamps,
            BmarkFamily::SequenceTimestamps,
            BmarkFamily::SequenceDurations,
            BmarkFamily::SequenceExclusiveDurations,
    };
    for (const auto family : families) {
        const auto& stats = *family_stats(aggregate, family);
        archive_out << archive->id << ',' << family_name(family) << ',' << stats.pre_raw_bytes << ','
                    << stats.raw_bytes << ',' << stats.compressed_bytes << ',' << stats.write_ns << ','
                    << stats.write_calls << ',' << stats.subarray_writes << ',' << stats.value_count
                    << ',' << stats.add_ns << ',' << stats.add_calls << ',' << stats.at_ns << ','
                    << stats.at_calls << ',' << stats.operator_ns << ',' << stats.operator_calls << ','
                    << stats.max_abs_error << ',' << stats.sum_abs_error << ','
                    << stats.sum_squared_abs_error << ',' << stats.nonzero_error_count << '\n';
        perf_out << archive->id << ',' << family_name(family) << ',' << stats.value_count << ','
                 << stats.recent_value_hits << ',' << stats.recent_value_misses << ','
                 << stats.find_subarray_calls << ',' << stats.find_subarray_steps << ','
                 << stats.subarray_loads << ',' << stats.subarray_load_bytes << ','
                 << stats.subarray_decompressed_bytes << ',' << stats.subarray_evictions << ','
                 << stats.subarray_evicted_bytes << '\n';
    }
}

}  // namespace pallas

#endif
