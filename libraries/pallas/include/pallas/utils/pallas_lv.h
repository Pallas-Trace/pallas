/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Clean linked-vector scaffold built on top of the standalone subarray layer.
 */
#pragma once

#ifndef __cplusplus

typedef struct TimeLV {
} TimeLV;

typedef struct DurationLV {
} DurationLV;

#else

#include <cstddef>
#include <cstdint>
#include <array>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#ifdef BMARK
#include "pallas/utils/pallas_bmark.h"
#endif

#include "pallas_parameter_handler.h"
#include "pallas_subarray.h"

namespace pallas {

class RecentValueRingBuffer {
   public:
    static constexpr size_t kCapacity = 64;

    void clear() {
        next_slot = 0;
        entry_count = 0;
    }

    void push(size_t index, uint64_t value) {
        entries[next_slot] = Entry{index, value};
        next_slot = (next_slot + 1) % kCapacity;
        if (entry_count < kCapacity) {
            entry_count++;
        }
    }

    [[nodiscard]] bool lookup(size_t index, uint64_t& value) const {
        for (size_t i = 0; i < entry_count; ++i) {
            const size_t offset = (next_slot + kCapacity - 1 - i) % kCapacity;
            const auto& entry = entries[offset];
            if (entry.index == index) {
                value = entry.value;
                return true;
            }
        }
        return false;
    }

   private:
    struct Entry {
        size_t index = 0;
        uint64_t value = 0;
    };

    std::array<Entry, kCapacity> entries{};
    size_t next_slot = 0;
    size_t entry_count = 0;
};

/**
 * Common base for the new linked-vector layer.
 *
 */
class LVBase {
   protected:
    // Core linked-vector metadata.
    size_t value_count = 0;     // Number of values stored in the linked vector.
    size_t reference_count = 0; // Number of objects that refer to this linked vector.
    size_t subarray_total = 0;  // Total number of subarrays in this linked vector.
    bool is_contiguous = false; // Whether the current representation is contiguous.
    ParameterHandler& parameter_handler; // Reference to the parameter handler
    const char* file_path = nullptr;     // File Path for ???

    // Loaded subarray tracking and recent-value cache.
    std::set<SubArrayBase*> loaded_subarrays;
    mutable RecentValueRingBuffer recent_values;

    // Metadata for SubArray
    ValueDomain value_domain;
    StoragePolicy storage_policy = StoragePolicy::None;
    void* hbuffer = nullptr;
    size_t hbuffer_bytes = 0;
#ifdef BMARK
    BmarkFamily benchmark_family = BmarkFamily::Unknown;
#endif

    SubArrayBase* first = nullptr;
    SubArrayBase* last = nullptr;

   public:
    virtual ~LVBase();

    /** Core methods supported by the old LinkedVector implementation. */
    [[nodiscard]] size_t size() const {
        return value_count;
    }
    [[nodiscard]] size_t ref_count() const {
        return reference_count;
    }
    [[nodiscard]] size_t subarray_count() const {
        return subarray_total;
    }
    [[nodiscard]] size_t loaded_subarray_count() const {
        return loaded_subarrays.size();
    }
    [[nodiscard]] bool empty() const {
        return value_count == 0;
    }
    [[nodiscard]] bool contiguous() const {
        return is_contiguous;
    }
    void reset_offsets();

    /** Value access helpers. */
    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    [[nodiscard]] uint64_t front() const;
    [[nodiscard]] uint64_t back() const;
    [[nodiscard]] uint64_t* as_flat_array() const;
    [[nodiscard]] std::string values_to_string() const;

    /** TimeLV = Timestamp and DurationLV = Duration. */
    [[nodiscard]] ValueDomain domain() const {
        return value_domain;
    }

    /** Storage and Loading Helpers */
    void set_storage_policy(StoragePolicy policy) {
        storage_policy = policy;
    }
    void setPreferredStoragePolicy(StoragePolicy policy) {
        set_storage_policy(policy);
    }

    void load_all();
    void free_data();
    bool apply_storage_policy();

#ifdef BMARK
    void set_bmark_family(BmarkFamily family) {
        benchmark_family = family;
    }
    [[nodiscard]] BmarkFamily get_bmark_family() const {
        return benchmark_family;
    }
#endif

   protected:
    friend class PLAManager;

    explicit LVBase(ParameterHandler& p, ValueDomain domain, StoragePolicy _policy);
    explicit LVBase(FILE* vector_file,
                    const char* value_file_path,
                    ParameterHandler& p,
                    ValueDomain domain,
                    StoragePolicy _policy,
                    uint8_t abi_version);

    void ensure_hbuffer(size_t bytes);
    void write_common_header(FILE* vector_file) const;
    void evict_loaded_subarrays();
    void load_data(SubArrayBase* subarray);
    [[nodiscard]] void* helper_buffer() const {
        return hbuffer;
    }
    [[nodiscard]] size_t helper_buffer_size() const {
        return hbuffer_bytes;
    }

    [[nodiscard]] SubArrayBase* find_subarray(size_t pos);
    [[nodiscard]] const SubArrayBase* find_subarray(size_t pos) const;

    virtual SubArrayBase* create_subarray(SubArrayBase* previous) const = 0;
    
   #if 1
    public:
    /* Temporary Helpers to help post-mortem analysis*/
    [[nodiscard]] StoragePolicy get_storage_policy() const {
        return storage_policy;
    }
    [[nodiscard]] std::vector<StoragePolicy> get_sub_array_policies() const;
    [[nodiscard]] std::vector<StoragePolicy> get_loaded_sub_array_policies() const;
   #endif
};

class TimeLV : public LVBase {
   public:
    // Runtime-write constructors and file-backed reconstruction constructor.
    explicit TimeLV(ParameterHandler& p);
    explicit TimeLV(ParameterHandler& p, StoragePolicy _policy);
    TimeLV(FILE* vector_file, const char* value_file_path, ParameterHandler& p, uint8_t abi_version);

    // Append one timestamp value to the linked vector.
    AddStatus add(uint64_t val);

    // Timestamp-specific inspection and query helpers.
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::vector<double> getWeights(pallas_timestamp_t start, pallas_timestamp_t end) const;
    [[nodiscard]] size_t getFirstOccurrenceBefore(pallas_timestamp_t ts) const;

    // Header and payload serialization helpers.
    void write_header(FILE* info_file);
    void write_to_file(FILE* info_file, FILE* data_file, const ParameterHandler* parameter_handler);

   protected:
    // Create the next timestamp subarray in the chain.
    SubArrayBase* create_subarray(SubArrayBase* previous) const override;
};

class DurationLV : public LVBase {
   public:
    // Runtime-write constructors and file-backed reconstruction constructor.
    explicit DurationLV(ParameterHandler& p);
    explicit DurationLV(ParameterHandler& p, StoragePolicy _policy);
    DurationLV(FILE* vector_file, const char* value_file_path, ParameterHandler& p, uint8_t abi_version);

    // Append one duration value to the linked vector.
    AddStatus add(uint64_t val);

    // Finalize duration statistics once the current subarray stops accepting values.
    void final_update_mean();

    // Duration-specific inspection and aggregate helpers.
    [[nodiscard]] pallas_duration_t weightedSum(std::vector<double>& weights) const;
    [[nodiscard]] pallas_duration_t computeDurationBetween(size_t start_index, size_t end_index) const;
    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] uint64_t min_value() const;
    [[nodiscard]] uint64_t max_value() const;
    [[nodiscard]] uint64_t mean_value() const;

    // Header and payload serialization helpers.
    void write_header(FILE* info_file);
    void write_to_file(FILE* info_file, FILE* data_file, const ParameterHandler* parameter_handler);

   protected:
    // Create the next duration subarray in the chain.
    SubArrayBase* create_subarray(SubArrayBase* previous) const override;

    // Running duration statistics tracked at the linked-vector level.
    uint64_t min_duration = UINT64_MAX;
    uint64_t max_duration = 0;
    uint64_t mean_duration = 0;
};

}  // namespace pallas

#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
