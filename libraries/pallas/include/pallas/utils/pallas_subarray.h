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
#include <memory>

#ifndef DEFAULT_VECTOR_SIZE
#define DEFAULT_VECTOR_SIZE 1000
#endif

#ifndef DEFAULT_SMALL_SIZE
#define DEFAULT_SMALL_SIZE 32
#endif

namespace pallas {

class ParameterHandler;

enum class ValueDomain : uint8_t {
    Timestamp = 0,
    Duration = 1,
};

enum class StoragePolicy : uint8_t {
    None = 0,
    Delta = 1,
    Lossy = 2,
};

enum class LossyPolicy : uint8_t {
    Linear = 0,
    NormalSample = 1,
};

enum class AddStatus : uint8_t {
    Ok = 0,
    Outlier = 1,
    Full = 2,
};

class SubArrayBase;

class Manager {
   public:
    virtual ~Manager();

    [[nodiscard]] virtual size_t recommended_capacity(ValueDomain domain, StoragePolicy policy) const = 0;
    virtual AddStatus add(SubArrayBase& subarray, uint64_t val) const = 0;

    [[nodiscard]] virtual uint64_t at(const SubArrayBase& subarray, size_t pos) const = 0;
    virtual void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const = 0;
    virtual void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) const = 0;
    virtual void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) const = 0;
};

class NoneManager : public Manager {
   public:
    [[nodiscard]] size_t recommended_capacity(ValueDomain domain, StoragePolicy policy) const override;
    AddStatus add(SubArrayBase& subarray, uint64_t val) const override;
    [[nodiscard]] uint64_t at(const SubArrayBase& subarray, size_t pos) const override;
    void copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const override;
    void write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) const override;
    void load_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler& parameter_handler) const override;
};

class SubArrayBase {
   public:
    virtual ~SubArrayBase();

    virtual AddStatus add(uint64_t val) = 0;

    [[nodiscard]] uint64_t at(size_t pos) const;
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    void copy_to_array(uint64_t* given_array) const;

    [[nodiscard]] ValueDomain domain() const;
    [[nodiscard]] StoragePolicy policy() const;
    [[nodiscard]] size_t size() const;
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

    explicit SubArrayBase(ValueDomain domain, StoragePolicy policy = StoragePolicy::None, SubArrayBase* previous = nullptr);
    explicit SubArrayBase(FILE* info_file, ValueDomain domain, StoragePolicy policy = StoragePolicy::None, SubArrayBase* previous = nullptr);
    [[nodiscard]] bool contains(size_t pos) const;
    [[nodiscard]] size_t local_index(size_t pos) const;
    [[nodiscard]] uint64_t* raw_values();
    void free_values();

    SubArrayBase* next = nullptr;
    SubArrayBase* prev = nullptr;
    ValueDomain value_domain;
    StoragePolicy storage_policy = StoragePolicy::None;
    std::unique_ptr<Manager> manager;
    size_t value_count = 0;
    size_t allocated_count = DEFAULT_VECTOR_SIZE;
    uint64_t* values = nullptr;
    size_t first_index = 0;
    size_t file_offset = 0;
};

class TimeSubArray : public SubArrayBase {
   public:
    explicit TimeSubArray(StoragePolicy policy = StoragePolicy::None, TimeSubArray* previous = nullptr);
    explicit TimeSubArray(FILE* info_file, StoragePolicy policy = StoragePolicy::None, TimeSubArray* previous = nullptr);

    AddStatus add(uint64_t val) override;
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    void write_header(FILE* info_file) const;
    void read_header(FILE* info_file);

    [[nodiscard]] uint64_t first_value() const;
    [[nodiscard]] uint64_t last_value() const;

   protected:
    uint64_t first_timestamp = 0;
    uint64_t last_timestamp = 0;
};

class DurationSubArray : public SubArrayBase {
   public:
    explicit DurationSubArray(StoragePolicy policy = StoragePolicy::None, DurationSubArray* previous = nullptr);
    explicit DurationSubArray(FILE* info_file, StoragePolicy policy = StoragePolicy::None, DurationSubArray* previous = nullptr);

    AddStatus add(uint64_t val) override;
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    void write_header(FILE* info_file) const;
    void read_header(FILE* info_file);
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
