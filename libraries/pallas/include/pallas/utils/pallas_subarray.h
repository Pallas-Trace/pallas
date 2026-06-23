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
#include "pallas/utils/pallas_pla.h"

#ifndef __cplusplus
#include <stdint.h>
#else

#include <cstddef>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include "pallas/utils/pallas_parameter_handler.h"

#ifndef DEFAULT_VECTOR_SIZE
#define DEFAULT_VECTOR_SIZE 1000
#endif


#ifndef VECTOR_SIZE_32 
#define VECTOR_SIZE_32 32
#endif

#ifndef VECTOR_SIZE_2048 
#define VECTOR_SIZE_2048 2048
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

constexpr LossyPolicy DEFAULT_LOSSY_TIME = LossyPolicy::PLA8;
constexpr LossyPolicy DEFAULT_LOSSY_DURATION = LossyPolicy::NormalSample;

enum class AddStatus : uint8_t {
    Ok = 0,
    Outlier = 1,
    Full = 2,
};

}

/** Manager Class - Handles the Internals of Subarray */
namespace pallas {

class SubArrayBase;
class TimeSubArray;
class LVBase;

class Manager {
   public:
    explicit Manager(SubArrayBase& parent)
        : parent(parent) {}
    virtual ~Manager();

    [[nodiscard]] virtual size_t _capacity() const = 0;
    virtual AddStatus add(uint64_t val) = 0;

    [[nodiscard]] virtual uint64_t at(size_t pos) const = 0;
    virtual void copy_to_array(uint64_t* given_array) const = 0;
    virtual void write_data(FILE* data_file, const ParameterHandler* parameter_handler) = 0;
    virtual void load_data(FILE* data_file, const ParameterHandler& parameter_handler) = 0;
    virtual void on_subarray_initialized();
    virtual void on_values_freed() = 0;

   protected:
    SubArrayBase& parent;
};

class NoneManager : public Manager {
   public:
    explicit NoneManager(SubArrayBase& parent)
        : Manager(parent) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed() override;
};

class DeltaManager : public Manager {
   public:
    DeltaManager(SubArrayBase& parent, ValueDomain value_domain)
        : Manager(parent), dom(value_domain) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed() override;

   private:
    union PrevDelta {
        uint64_t u;
        int64_t i;

        PrevDelta()
            : u(0) {}
    };

    struct Checkpoint {
        size_t idx = 0;
        size_t off = 0;
        uint64_t val = 0;
        PrevDelta prev;
    };

    struct State {
        uint64_t last = 0;
        PrevDelta prev;
    };

    static constexpr size_t kCheckpointStride = 50;
    [[nodiscard]] bool is_time_domain() const {
        return dom == ValueDomain::Timestamp;
    }
    AddStatus add_time(uint64_t val);
    AddStatus add_duration(uint64_t val);
    [[nodiscard]] uint64_t at_time(size_t pos) const;
    [[nodiscard]] uint64_t at_duration(size_t pos) const;
    void copy_time_to_array(uint64_t* given_array) const;
    void copy_duration_to_array(uint64_t* given_array) const;
    void load_time_data(FILE* data_file, const ParameterHandler& parameter_handler);
    void load_duration_data(FILE* data_file, const ParameterHandler& parameter_handler);

    ValueDomain dom;
    uint8_t* payload = nullptr;
    size_t bytes = 0;
    size_t cap_bytes = 0;
    State st;
    std::vector<Checkpoint> cps;
};

class PLAManager : public Manager {
   public:
    PLAManager(SubArrayBase& parent, uint8_t k_max)
        : Manager(parent), k_max(k_max) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_subarray_initialized() override;
    void on_values_freed() override;

   private:
    void ensure_staging();
    void finalize_block();
    void write_packed_payload();
    void load_packed_payload();
    [[nodiscard]] uint64_t interpolate_value(const TimeSubArray& subarray, size_t logical_index) const;
    void clear_state();

    uint8_t k_max = 0;
    uint8_t anchor_count = 0;
    bool compact_ready = false;
    GammaBlockStats stats{};
    PLAAnchor anchor_storage[kPLAMaxAnchors]{};
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
   /** Navigation and Storage State Metadata */
   protected:
    SubArrayBase* next = nullptr;
    SubArrayBase* prev = nullptr;
    size_t value_count = 0;
    size_t physical_size = 0;
    size_t first_index = 0;
    size_t file_offset = 0;

    /** Internal Logic Handler and Attached Buffer */
    std::unique_ptr<Manager> manager;
    uint64_t* buffer = nullptr;
    LVBase* parent_lv = nullptr;

   public:
    // Storage State Information
    [[nodiscard]] SubArrayBase* next_subarray() const;
    [[nodiscard]] SubArrayBase* previous_subarray() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t mem_size() const;
    [[nodiscard]] size_t starting_index() const;
    [[nodiscard]] size_t offset() const;
    void set_offset(size_t offset);
    
    // SubArray API Buffer Management, Manager 
    virtual ~SubArrayBase();
    virtual AddStatus add(uint64_t val) = 0;
    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    void copy_values(uint64_t* given_array) const;
    [[nodiscard]] size_t capacity() const;
    [[nodiscard]] bool has_values() const;
   protected:
    // Buffer Management and Accessor helpers
    [[nodiscard]] bool contains(size_t pos) const;
    [[nodiscard]] size_t local_index(size_t pos) const;
    [[nodiscard]] uint64_t* raw_buffer();
    void free_values();
    void rebuild_manager();

   protected:
    ValueDomain value_domain;
    StoragePolicy storage_policy = StoragePolicy::None;
    LossyPolicy lossy_storage_policy = DEFAULT_LOSSY_TIME;
    SubArrayPhase subarray_phase = SubArrayPhase::RuntimeWrite;
   
   public:
    [[nodiscard]] ValueDomain domain() const;
    [[nodiscard]] StoragePolicy policy() const;
    [[nodiscard]] LossyPolicy lossy_policy() const;
    [[nodiscard]] SubArrayPhase phase() const;

    [[nodiscard]] uint8_t pack_subarray_flags() const;
    void unpack_subarray_flags(uint8_t encoded_policy);    

   protected:
    /** Access control for managers and LVBase */
    friend class Manager;
    friend class NoneManager;
    friend class DeltaManager;
    friend class PLAManager;
    friend class LVBase;

    /** Construction and File time helpers */
    explicit SubArrayBase(ValueDomain domain,
                          StoragePolicy policy = StoragePolicy::None,
                          SubArrayBase* previous = nullptr,
                          const ParameterHandler* parameter_handler = nullptr,
                          LVBase* parent = nullptr);
    explicit SubArrayBase(FILE* info_file, ValueDomain domain, SubArrayBase* previous = nullptr);
    
    
    void write_common_header(FILE* info_file) const;
    void read_common_header(FILE* info_file);
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler);
};

class TimeSubArray : public SubArrayBase {
   public:
    // Runtime-write constructor and file-backed reconstruction constructor.
    explicit TimeSubArray(StoragePolicy policy = StoragePolicy::None,
                          TimeSubArray* previous = nullptr,
                          const ParameterHandler* parameter_handler = nullptr,
                          LVBase* parent = nullptr);
                          
    explicit TimeSubArray(FILE* info_file, TimeSubArray* previous = nullptr);

    // Timestamp-specific insertion and file serialization helpers.
    AddStatus add(uint64_t val) override;
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    void write_header(FILE* info_file) const;
    void read_header(FILE* info_file);

    // Cached timestamp bounds for this subarray.
    [[nodiscard]] uint64_t first_value() const;
    [[nodiscard]] uint64_t last_value() const;

   protected:
    friend class DeltaManager;
    friend class PLAManager;

    // First and last logical timestamps stored in this subarray.
    uint64_t first_timestamp = 0;
    uint64_t last_timestamp = 0;
};

class DurationSubArray : public SubArrayBase {
   public:
    // Runtime-write constructor and file-backed reconstruction constructor.
    explicit DurationSubArray(StoragePolicy policy = StoragePolicy::None,
                              DurationSubArray* previous = nullptr,
                              const ParameterHandler* parameter_handler = nullptr,
                              LVBase* parent = nullptr);
    explicit DurationSubArray(FILE* info_file, DurationSubArray* previous = nullptr);

    // Duration-specific insertion, file serialization, and statistics helpers.
    AddStatus add(uint64_t val) override;
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    void write_header(FILE* info_file) const;
    void read_header(FILE* info_file);
    void update_statistics(uint64_t current_value);
    void final_update_mean();

    // Aggregate duration statistics for this subarray.
    [[nodiscard]] uint64_t min_value() const;
    [[nodiscard]] uint64_t max_value() const;
    [[nodiscard]] uint64_t mean_value() const;

   protected:
    // Cached duration statistics for the logical values stored in this subarray.
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
