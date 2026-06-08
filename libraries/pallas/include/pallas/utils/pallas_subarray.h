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
#include <cstdint>
#include <cstdio>

#ifndef DEFAULT_VECTOR_SIZE
#define DEFAULT_VECTOR_SIZE 1000
#endif

#ifndef DEFAULT_SMALL_SIZE
#define DEFAULT_SMALL_SIZE 32
#endif

namespace pallas {

enum class ValueDomain : uint8_t {
    Timestamp = 0,
    Duration = 1,
};

enum class Policy : uint8_t {
    None = 0,
    Delta = 1,
    Lossy = 2,
};

enum class LossyPolicy : uint8_t {
    Linear = 0,  // ValueDomain must be Timestamp
    Normal = 1,  // ValueDomain must be Duration
};

enum class AddStatus : uint8_t {
    Ok = 0,
    Outlier = 1,
    Full = 2,
};

class Manager {
   public:
    explicit Manager(ValueDomain domain, Policy policy = Policy::None, size_t starting_index = 0);
    ~Manager();

    AddStatus add_raw(uint64_t val);

    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    void copy_to_array(uint64_t* given_array) const;

    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t capacity() const;
    [[nodiscard]] size_t starting_index() const;
    void set_starting_index(size_t starting_index);
    [[nodiscard]] size_t offset() const;
    void set_offset(size_t offset);
    [[nodiscard]] ValueDomain domain() const;
    [[nodiscard]] Policy policy() const;

    [[nodiscard]] uint64_t* data() const;

    void dump_runtime_state(FILE* file) const;
    void load_runtime_state(FILE* file);

    void reset_prediction_state();
    void note_prediction_sample(uint64_t value);

   protected:
    size_t value_count = 0;
    size_t allocated_count = DEFAULT_VECTOR_SIZE;
    uint64_t* values = nullptr;
    size_t first_index = 0;
    size_t file_offset = 0;
    ValueDomain value_domain = ValueDomain::Timestamp;
    Policy storage_policy = Policy::None;

    // Placeholders for upcoming on-the-fly prediction logic.
    bool dynamic_mode_enabled = false;
    size_t prediction_sample_count = 0;
};

class SubArrayBase {
   public:
    virtual ~SubArrayBase();

    virtual AddStatus add(uint64_t val) = 0;

    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    void copy_to_array(uint64_t* given_array) const;

    [[nodiscard]] ValueDomain domain() const;
    [[nodiscard]] Policy policy() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t capacity() const;
    [[nodiscard]] size_t starting_index() const;
    [[nodiscard]] size_t offset() const;
    void set_offset(size_t offset);

   protected:
    explicit SubArrayBase(ValueDomain domain, Policy policy = Policy::None, SubArrayBase* previous = nullptr);
    SubArrayBase* next = nullptr;
    SubArrayBase* prev = nullptr;
    ValueDomain value_domain;
    Manager manager;
};

class TimeSubArray : public SubArrayBase {
   public:
    explicit TimeSubArray(Policy policy = Policy::None, TimeSubArray* previous = nullptr);

    AddStatus add(uint64_t val) override;

    [[nodiscard]] uint64_t first_value() const;
    [[nodiscard]] uint64_t last_value() const;

   protected:
    uint64_t first_timestamp = 0;
    uint64_t last_timestamp = 0;
};

class DurationSubArray : public SubArrayBase {
   public:
    explicit DurationSubArray(Policy policy = Policy::None, DurationSubArray* previous = nullptr);

    AddStatus add(uint64_t val) override;
    void update_statistics();
    void final_update_mean();

    [[nodiscard]] uint64_t min_value() const;
    [[nodiscard]] uint64_t max_value() const;
    [[nodiscard]] uint64_t mean_value() const;

   protected:
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
