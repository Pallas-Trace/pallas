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
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "pallas_parameter_handler.h"
#include "pallas_subarray.h"

namespace pallas {


/**
 * Common base for the new linked-vector layer.
 *
 * Unlike the legacy linked-vector implementation, this base owns standalone
 * subarrays directly instead of defining a nested subarray hierarchy.
 */
class LVBase {
   public:
    virtual ~LVBase();

    [[nodiscard]] size_t size() const {
        return value_count;
    }
    [[nodiscard]] size_t ref_count() const {
        return reference_count;
    }
    [[nodiscard]] size_t subarray_count() const {
        return subarray_total;
    }
    [[nodiscard]] bool empty() const {
        return value_count == 0;
    }
    [[nodiscard]] bool contiguous() const {
        return is_contiguous;
    }
    [[nodiscard]] ValueDomain domain() const {
        return value_domain;
    }

    void setPreferredStoragePolicy(StoragePolicy policy) {
        preferred_storage_policy = policy;
    }
    [[nodiscard]] StoragePolicy getPreferredStoragePolicy() const {
        return preferred_storage_policy;
    }

    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    [[nodiscard]] uint64_t front() const;
    [[nodiscard]] uint64_t back() const;

    [[nodiscard]] uint64_t* as_flat_array() const;
    [[nodiscard]] std::string values_to_string() const;

    void load_all_data() {
    }
    void reset_offsets();

   protected:
    explicit LVBase(ParameterHandler& p, ValueDomain domain, StoragePolicy preferred_policy);

    [[nodiscard]] SubArrayBase* find_subarray(size_t pos);
    [[nodiscard]] const SubArrayBase* find_subarray(size_t pos) const;
    virtual SubArrayBase* create_subarray(SubArrayBase* previous) const = 0;

    ParameterHandler& parameter_handler;
    ValueDomain value_domain;
    StoragePolicy preferred_storage_policy = StoragePolicy::None;
    const char* file_path = nullptr;
    size_t value_count = 0;
    size_t reference_count = 0;
    size_t subarray_total = 0;
    bool is_contiguous = false;
    std::set<SubArrayBase*> loaded_subarrays;
    SubArrayBase* first = nullptr;
    SubArrayBase* last = nullptr;
};

class TimeLV : public LVBase {
   public:
    explicit TimeLV(ParameterHandler& p);
    explicit TimeLV(ParameterHandler& p, StoragePolicy preferred_policy);

    AddStatus add(uint64_t val);

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::vector<double> getWeights(pallas_timestamp_t start, pallas_timestamp_t end) const;
    [[nodiscard]] size_t getFirstOccurrenceBefore(pallas_timestamp_t ts) const;

    void write_to_file(FILE* info_file, FILE* data_file, const ParameterHandler* parameter_handler);

   protected:
    SubArrayBase* create_subarray(SubArrayBase* previous) const override;
};

class DurationLV : public LVBase {
   public:
    explicit DurationLV(ParameterHandler& p);
    explicit DurationLV(ParameterHandler& p, StoragePolicy preferred_policy);

    AddStatus add(uint64_t val);
    void final_update_mean();

    [[nodiscard]] pallas_duration_t weightedSum(std::vector<double>& weights) const;
    [[nodiscard]] pallas_duration_t computeDurationBetween(size_t start_index, size_t end_index) const;
    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] uint64_t min_value() const;
    [[nodiscard]] uint64_t max_value() const;
    [[nodiscard]] uint64_t mean_value() const;

    void write_to_file(FILE* info_file, FILE* data_file, const ParameterHandler* parameter_handler);

   protected:
    SubArrayBase* create_subarray(SubArrayBase* previous) const override;

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
