#pragma once

#ifdef BMARK

#include <cstddef>
#include <cstdint>

namespace pallas {

struct Archive;

enum class BmarkFamily : uint8_t {
    Unknown = 0,
    EventTimestamps = 1,
    SequenceTimestamps = 2,
    SequenceDurations = 3,
    SequenceExclusiveDurations = 4,
};

enum class BmarkMetric : uint8_t {
    Add = 0,
    At = 1,
    Operator = 2,
    Write = 3,
};

struct BmarkFamilyStats {
    uint64_t pre_raw_bytes = 0;
    uint64_t raw_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint64_t write_ns = 0;
    uint64_t write_calls = 0;
    uint64_t subarray_writes = 0;
    uint64_t value_count = 0;
    uint64_t add_ns = 0;
    uint64_t add_calls = 0;
    uint64_t at_ns = 0;
    uint64_t at_calls = 0;
    uint64_t operator_ns = 0;
    uint64_t operator_calls = 0;
    uint64_t max_abs_error = 0;
    uint64_t sum_abs_error = 0;
    double sum_squared_abs_error = 0.0;
    uint64_t nonzero_error_count = 0;

    void accumulate(const BmarkFamilyStats& other);
};

struct BmarkThreadStats {
    BmarkFamilyStats event_timestamps;
    BmarkFamilyStats sequence_timestamps;
    BmarkFamilyStats sequence_durations;
    BmarkFamilyStats sequence_exclusive_durations;

    void clear();
    void accumulate(const BmarkThreadStats& other);
};

class BmarkScopedTimer {
   public:
    BmarkScopedTimer(BmarkFamily family, BmarkMetric metric);
    ~BmarkScopedTimer();

   private:
    BmarkFamily family;
    BmarkMetric metric;
    uint64_t start_ns = 0;
};

void bmark_note_write_call(BmarkFamily family, uint64_t logical_value_count);
void bmark_note_subarray_write(BmarkFamily family,
                               uint64_t pre_raw_bytes,
                               uint64_t raw_bytes,
                               uint64_t compressed_bytes);
void bmark_note_error_values(BmarkFamily family,
                             const uint64_t* exact_values,
                             const uint64_t* observed_values,
                             size_t value_count);
void bmark_flush_thread_stats(Archive* archive);
void bmark_reset_thread_stats();
void bmark_write_archive_csv(const Archive* archive, const char* root_path);

}  // namespace pallas

#endif
