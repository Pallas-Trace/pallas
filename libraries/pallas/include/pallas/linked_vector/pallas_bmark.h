/** @file
 * Benchmark instrumentation helpers enabled when `BMARK` is defined.
 */
#pragma once

#ifdef BMARK

#include <cstddef>
#include <cstdint>

namespace pallas {

struct Archive;

/**
 * @brief Identifies which linked-vector family a benchmark sample belongs to.
 */
enum class BmarkFamily : uint8_t {
    Unknown = 0,
    EventTimestamps = 1,
    SequenceTimestamps = 2,
    SequenceDurations = 3,
    SequenceExclusiveDurations = 4,
};

/**
 * @brief Identifies which operation is being timed by the benchmark layer.
 */
enum class BmarkMetric : uint8_t {
    Add = 0,
    At = 1,
    Operator = 2,
    Write = 3,
};

/**
 * @brief Archive- or thread-aggregated benchmark counters for one value family.
 *
 * These counters track compression sizes, write overhead, indexed-read costs,
 * cache effectiveness, subarray load or eviction activity, and value-error
 * measurements for one `BmarkFamily`.
 */
struct BmarkFamilyStats {
    /** Logical uncompressed bytes represented by the observed values. */
    uint64_t pre_raw_bytes = 0;
    /** Bytes written by the family-specific encoding layer before backend compression. */
    uint64_t raw_bytes = 0;
    /** Bytes written after backend compression. */
    uint64_t compressed_bytes = 0;
    /** Total time spent in vector or subarray write paths. */
    uint64_t write_ns = 0;
    /** Number of logical write calls recorded for this family. */
    uint64_t write_calls = 0;
    /** Number of physical subarray payload writes emitted for this family. */
    uint64_t subarray_writes = 0;
    /** Number of logical values observed for this family. */
    uint64_t value_count = 0;
    /** Total time spent in `add()` calls. */
    uint64_t add_ns = 0;
    /** Number of `add()` calls observed. */
    uint64_t add_calls = 0;
    /** Total time spent in `at()` calls. */
    uint64_t at_ns = 0;
    /** Number of `at()` calls observed. */
    uint64_t at_calls = 0;
    /** Total time spent in `operator[]` calls. */
    uint64_t operator_ns = 0;
    /** Number of `operator[]` calls observed. */
    uint64_t operator_calls = 0;
    /** Count of recent-value cache hits. */
    uint64_t recent_value_hits = 0;
    /** Count of recent-value cache misses. */
    uint64_t recent_value_misses = 0;
    /** Number of `find_subarray()` invocations. */
    uint64_t find_subarray_calls = 0;
    /** Accumulated lookup steps spent inside `find_subarray()`. */
    uint64_t find_subarray_steps = 0;
    /** Number of subarray payload loads performed. */
    uint64_t subarray_loads = 0;
    /** Number of bytes loaded from persisted subarray payloads. */
    uint64_t subarray_load_bytes = 0;
    /** Number of bytes materialized in memory after subarray reload. */
    uint64_t subarray_decompressed_bytes = 0;
    /** Number of subarray evictions performed. */
    uint64_t subarray_evictions = 0;
    /** Number of bytes evicted from memory. */
    uint64_t subarray_evicted_bytes = 0;
    /** Maximum absolute reconstruction error observed. */
    uint64_t max_abs_error = 0;
    /** Sum of absolute reconstruction errors. */
    uint64_t sum_abs_error = 0;
    /** Sum of squared absolute errors, used for variance or standard-deviation style analysis. */
    double sum_squared_abs_error = 0.0;
    /** Number of values whose absolute error was non-zero. */
    uint64_t nonzero_error_count = 0;

    /**
     * @brief Adds another family-stat sample into this aggregate.
     * @param other Source counters to accumulate.
     */
    void accumulate(const BmarkFamilyStats& other);
};

/**
 * @brief Thread-local benchmark aggregates for all linked-vector families.
 */
struct BmarkThreadStats {
    /** Benchmark counters for event timestamp vectors. */
    BmarkFamilyStats event_timestamps;
    /** Benchmark counters for sequence timestamp vectors. */
    BmarkFamilyStats sequence_timestamps;
    /** Benchmark counters for sequence duration vectors. */
    BmarkFamilyStats sequence_durations;
    /** Benchmark counters for exclusive-duration vectors. */
    BmarkFamilyStats sequence_exclusive_durations;

    /** Resets every family aggregate to zero. */
    void clear();
    /**
     * @brief Adds another thread-stat sample into this aggregate.
     * @param other Source counters to accumulate.
     */
    void accumulate(const BmarkThreadStats& other);
};

/**
 * @brief RAII timer used to accumulate one benchmark metric automatically.
 *
 * Construction records a start timestamp and destruction reports the elapsed
 * duration to the benchmark subsystem for the given `BmarkFamily` and
 * `BmarkMetric`.
 */
class BmarkScopedTimer {
   public:
    /**
     * @brief Starts timing one benchmarked operation.
     * @param family Linked-vector family being timed.
     * @param metric Operation category being timed.
     */
    BmarkScopedTimer(BmarkFamily family, BmarkMetric metric);
    /** Stops timing and reports the elapsed duration. */
    ~BmarkScopedTimer();

   private:
    BmarkFamily family;
    BmarkMetric metric;
    uint64_t start_ns = 0;
};

/** Records one logical write call and the number of values involved. */
void bmark_note_write_call(BmarkFamily family, uint64_t logical_value_count);
/**
 * @brief Records the size contribution of one physical subarray write.
 * @param family Linked-vector family being written.
 * @param pre_raw_bytes Logical uncompressed size represented by the values.
 * @param raw_bytes Encoded size before backend compression.
 * @param compressed_bytes Final persisted size after backend compression.
 */
void bmark_note_subarray_write(BmarkFamily family,
                               uint64_t pre_raw_bytes,
                               uint64_t raw_bytes,
                               uint64_t compressed_bytes);
/** Records a recent-value cache lookup result. */
void bmark_note_recent_value_lookup(BmarkFamily family, bool hit);
/** Records one `find_subarray()` lookup and the number of steps it required. */
void bmark_note_find_subarray(BmarkFamily family, uint64_t steps);
/** Records one subarray payload load and its byte counts. */
void bmark_note_subarray_load(BmarkFamily family, uint64_t load_bytes, uint64_t decompressed_bytes);
/** Records one subarray eviction and the number of bytes released. */
void bmark_note_subarray_evict(BmarkFamily family, uint64_t evicted_bytes);
/**
 * @brief Records reconstruction error statistics by comparing exact and observed values.
 * @param family Linked-vector family being evaluated.
 * @param exact_values Reference values treated as exact.
 * @param observed_values Values reconstructed or observed through the active storage path.
 * @param value_count Number of values to compare.
 */
void bmark_note_error_values(BmarkFamily family,
                             const uint64_t* exact_values,
                             const uint64_t* observed_values,
                             size_t value_count);
/** Flushes the current thread-local benchmark counters into the owning archive aggregate. */
void bmark_flush_thread_stats(Archive* archive);
/** Resets the current thread-local benchmark counters. */
void bmark_reset_thread_stats();
/**
 * @brief Writes the final archive-level benchmark summary as a CSV file.
 * @param archive Archive whose aggregated counters should be persisted.
 * @param root_path Output directory used for the CSV file.
 */
void bmark_write_archive_csv(const Archive* archive, const char* root_path);

}  // namespace pallas

#endif
