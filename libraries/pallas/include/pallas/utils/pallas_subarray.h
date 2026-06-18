/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Standalone subarray primitives used as a staging area for refactoring the
 * nested linked-vector subarray implementation.
 */
#pragma once

#include "pallas_timestamp.h"

#ifndef __cplusplus
#include <stdint.h>
#else

#include <cstddef>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pallas/utils/pallas_parameter_handler.h"

#ifndef DEFAULT_VECTOR_SIZE
#define DEFAULT_VECTOR_SIZE 1000
#endif

#ifndef DEFAULT_SMALL_SIZE
#define DEFAULT_SMALL_SIZE 32
#endif

/** Enum Domain, Policy and LossyPolicy Enums */
namespace pallas {

enum class ValueDomain : uint8_t {
    Timestamp = 0,
    Duration = 1,
};

enum class StoragePolicy : uint8_t {
    None = 0,
    Delta = 1,
    Lossy = 2,
};

enum class SubArrayPhase : uint8_t {
    RuntimeWrite = 0,
    AnalysisRead = 1,
};

enum class LossyPolicy : uint8_t {
    NormalSample = 0,
    PLA4 = 1,
    PLA8 = 2,
    PLA16 = 3,
    PLA32 = 4,
};

enum class AddStatus : uint8_t {
    Ok = 0,
    Outlier = 1,
    Full = 2,
};

[[nodiscard]] uint8_t encode_subarray_policy_byte(StoragePolicy policy, LossyPolicy lossy_policy);
void decode_subarray_policy_byte(uint8_t encoded_policy,
                                 StoragePolicy& storage_policy,
                                 LossyPolicy& lossy_policy,
                                 ValueDomain domain);

}

/** Manager Class - Handles the Internals of Subarray */
namespace pallas {

class SubArrayBase;
class TimeSubArray;

class Manager {
   public:
    virtual ~Manager();

    [[nodiscard]] virtual size_t recommended_capacity(ValueDomain domain, StoragePolicy policy, SubArrayPhase phase) const = 0;
    virtual AddStatus add(SubArrayBase& subarray, uint64_t val) = 0;

    [[nodiscard]] virtual uint64_t at(const SubArrayBase& subarray, size_t pos) const = 0;
    virtual void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const = 0;
    virtual void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) = 0;
    virtual void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) = 0;
    virtual void on_values_freed(SubArrayBase& subarray) = 0;
};

class NoneManager : public Manager {
   public:
    [[nodiscard]] size_t recommended_capacity(ValueDomain domain, StoragePolicy policy, SubArrayPhase phase) const override;
    AddStatus add(SubArrayBase& subarray, uint64_t val) override;
    [[nodiscard]] uint64_t at(const SubArrayBase& subarray, size_t pos) const override;
    void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const override;
    void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed(SubArrayBase& subarray) override;
};

class TimeDeltaManager : public Manager {
   public:
    [[nodiscard]] size_t recommended_capacity(ValueDomain domain, StoragePolicy policy, SubArrayPhase phase) const override;
    AddStatus add(SubArrayBase& subarray, uint64_t val) override;
    [[nodiscard]] uint64_t at(const SubArrayBase& subarray, size_t pos) const override;
    void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const override;
    void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed(SubArrayBase& subarray) override;

   private:
    struct Checkpoint {
        size_t logical_index = 0;
        size_t byte_offset = 0;
        uint64_t value = 0;
        uint64_t previous_delta = 0;
    };

    static constexpr size_t kCheckpointStride = 50;
    uint8_t* payload = nullptr;
    size_t payload_bytes = 0;
    size_t max_payload_bytes = 0;
    uint64_t previous_delta = 0;
    std::vector<Checkpoint> checkpoints;
};

class DurationDeltaManager : public Manager {
   public:
    [[nodiscard]] size_t recommended_capacity(ValueDomain domain, StoragePolicy policy, SubArrayPhase phase) const override;
    AddStatus add(SubArrayBase& subarray, uint64_t val) override;
    [[nodiscard]] uint64_t at(const SubArrayBase& subarray, size_t pos) const override;
    void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const override;
    void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed(SubArrayBase& subarray) override;

   private:
    struct Checkpoint {
        size_t logical_index = 0;
        size_t byte_offset = 0;
        uint64_t value = 0;
        int64_t previous_delta = 0;
    };

    static constexpr size_t kCheckpointStride = 50;
    uint8_t* payload = nullptr;
    size_t payload_bytes = 0;
    size_t max_payload_bytes = 0;
    uint64_t last_value = 0;
    int64_t previous_delta = 0;
    std::vector<Checkpoint> checkpoints;
};

class LinearTimeManager : public Manager {
   public:
    // Overridden base class methods
    [[nodiscard]] size_t recommended_capacity(ValueDomain domain, StoragePolicy policy, SubArrayPhase phase) const override;
    AddStatus add(SubArrayBase& subarray, uint64_t val) override;
    [[nodiscard]] uint64_t at(const SubArrayBase& subarray, size_t pos) const override;
    void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const override;
    void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed(SubArrayBase& subarray) override;
    // Epsilon
    void set_epsilon(uint64_t new_epsilon);

   private:
    static constexpr size_t kSeedValueCount = 16;
    static constexpr size_t kOutlierCapacity = 8;

    [[nodiscard]] bool model_active() const;
    [[nodiscard]] size_t representative_capacity() const;
    [[nodiscard]] size_t outlier_count(const SubArrayBase& subarray) const;
    [[nodiscard]] uint64_t prediction_from_fit(const uint64_t* fit_values, size_t fit_count, size_t logical_index) const;
    [[nodiscard]] uint64_t predict_value(size_t logical_index) const;
    [[nodiscard]] bool find_outlier(const SubArrayBase& subarray, size_t logical_index, uint64_t& value) const;
    void fit_model(const TimeSubArray& subarray, size_t fit_count);
    void sync_model_to_values(SubArrayBase& subarray) const;
    void refresh_model_from_values(const SubArrayBase& subarray);
    void activate_prediction_model(TimeSubArray& subarray, size_t fit_count);
    void clear_state();

    bool prediction_model_active = false;
    uint64_t epsilon = 64;
    uint64_t anchor_value = 0;
    double slope = 0.0;
};

[[nodiscard]] inline uint64_t zigzag_encode(int64_t x) {
    return (static_cast<uint64_t>(x) << 1) ^ static_cast<uint64_t>(x >> 63);
}

[[nodiscard]] inline int64_t zigzag_decode(uint64_t x) {
    return static_cast<int64_t>((x >> 1) ^ static_cast<uint64_t>(-static_cast<int64_t>(x & 1)));
}

inline void write_varint(uint64_t x, uint8_t*& out) {
    while (x >= 0x80) {
        *out++ = static_cast<uint8_t>((x & 0x7f) | 0x80);
        x >>= 7;
    }
    *out++ = static_cast<uint8_t>(x);
}

[[nodiscard]] inline uint64_t read_varint(const uint8_t*& p, const uint8_t* end) {
    uint64_t result = 0;
    int shift = 0;
    while (p < end) {
        const uint8_t byte = *p++;
        result |= static_cast<uint64_t>(byte & 0x7f) << shift;

        if ((byte & 0x80) == 0) {
            return result;
        }

        shift += 7;
        if (shift >= 64) {
            throw std::runtime_error("varint too long");
        }
    }
    throw std::runtime_error("truncated varint");
}

}

/** SubArray - Standalone Implementation of SubArray Class and its children Time and Duration SubArray */
namespace pallas {
class SubArrayBase {
   public:
    virtual ~SubArrayBase();

    virtual AddStatus add(uint64_t val) = 0;

    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    void copy_to_array(uint64_t* given_array) const;

    [[nodiscard]] ValueDomain domain() const;
    [[nodiscard]] StoragePolicy policy() const;
    [[nodiscard]] LossyPolicy lossy_policy() const;
    [[nodiscard]] SubArrayPhase phase() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t mem_size() const;
    [[nodiscard]] size_t capacity() const;
    [[nodiscard]] size_t starting_index() const;
    [[nodiscard]] size_t offset() const;
    [[nodiscard]] SubArrayBase* next_subarray() const;
    [[nodiscard]] SubArrayBase* previous_subarray() const;
    [[nodiscard]] bool has_values() const;
    void set_offset(size_t offset);
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler);

   protected:
    friend class Manager;
    friend class NoneManager;
    friend class TimeDeltaManager;
    friend class DurationDeltaManager;
    friend class LinearTimeManager;
    friend class LVBase;

    explicit SubArrayBase(ValueDomain domain,
                          StoragePolicy policy = StoragePolicy::None,
                          SubArrayBase* previous = nullptr,
                          const ParameterHandler* parameter_handler = nullptr);
    explicit SubArrayBase(FILE* info_file, ValueDomain domain, SubArrayBase* previous = nullptr);
    [[nodiscard]] bool contains(size_t pos) const;
    [[nodiscard]] size_t local_index(size_t pos) const;
    [[nodiscard]] uint64_t* raw_values();
    void free_values();
    void rebuild_manager();
    void write_common_header(FILE* info_file) const;
    void read_common_header(FILE* info_file);

    SubArrayBase* next = nullptr;
    SubArrayBase* prev = nullptr;
    ValueDomain value_domain;
    StoragePolicy storage_policy = StoragePolicy::None;
    LossyPolicy lossy_storage_policy = LossyPolicy::Linear;
    SubArrayPhase subarray_phase = SubArrayPhase::RuntimeWrite;
    std::unique_ptr<Manager> manager;
    size_t value_count = 0;
    size_t physical_size = 0;
    size_t allocated_count = DEFAULT_VECTOR_SIZE;
    uint64_t* values = nullptr;
    const ParameterHandler* configuration = nullptr;
    size_t first_index = 0;
    size_t file_offset = 0;
};

class TimeSubArray : public SubArrayBase {
   public:
    explicit TimeSubArray(StoragePolicy policy = StoragePolicy::None,
                          TimeSubArray* previous = nullptr,
                          const ParameterHandler* parameter_handler = nullptr);
    explicit TimeSubArray(FILE* info_file, TimeSubArray* previous = nullptr);

    AddStatus add(uint64_t val) override;
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    void write_header(FILE* info_file) const;
    void read_header(FILE* info_file);

    [[nodiscard]] uint64_t first_value() const;
    [[nodiscard]] uint64_t last_value() const;

   protected:
    friend class TimeDeltaManager;
    friend class LinearTimeManager;

    uint64_t first_timestamp = 0;
    uint64_t last_timestamp = 0;
};

class DurationSubArray : public SubArrayBase {
   public:
    explicit DurationSubArray(StoragePolicy policy = StoragePolicy::None,
                              DurationSubArray* previous = nullptr,
                              const ParameterHandler* parameter_handler = nullptr);
    explicit DurationSubArray(FILE* info_file, DurationSubArray* previous = nullptr);

    AddStatus add(uint64_t val) override;
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    void write_header(FILE* info_file) const;
    void read_header(FILE* info_file);
    void update_statistics(uint64_t current_value);
    void final_update_mean();

    [[nodiscard]] uint64_t min_value() const;
    [[nodiscard]] uint64_t max_value() const;
    [[nodiscard]] uint64_t mean_value() const;

   protected:
    uint64_t min_duration = UINT64_MAX;
    uint64_t max_duration = 0;
    uint64_t mean_duration = 0;
};

}  

#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
