/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_subarray.h"

/** Methods Pertaining to the memory manager of the SubArray */
namespace pallas {

namespace {

size_t resolve_manager_capacity(ValueDomain, Policy policy) {
    switch (policy) {
        case Policy::None:
        case Policy::Delta:
            return DEFAULT_VECTOR_SIZE;
        case Policy::Lossy:
            return DEFAULT_SMALL_SIZE;
    }

    return DEFAULT_VECTOR_SIZE;
}

}  // namespace

Manager::Manager(ValueDomain domain, Policy policy, size_t starting_index)
    : allocated_count(resolve_manager_capacity(domain, policy)),
      values(new uint64_t[allocated_count]),
      first_index(starting_index),
      value_domain(domain),
      storage_policy(policy) {}

Manager::~Manager() {
    delete[] values;
}

AddStatus Manager::add_raw(uint64_t val) {
    if (value_count >= allocated_count) {
        return AddStatus::Full;
    }

    values[value_count] = val;
    value_count++;
    return AddStatus::Ok;
}

uint64_t Manager::at(size_t pos) const {
    if (pos < first_index || pos >= first_index + value_count) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, first_index, value_count);
    }
    return values[pos - first_index];
}

uint64_t Manager::operator[](size_t pos) const {
    return values[pos - first_index];
}

void Manager::copy_to_array(uint64_t* given_array) const {
    std::memcpy(given_array, values, value_count * sizeof(uint64_t));
}

size_t Manager::size() const {
    return value_count;
}

size_t Manager::capacity() const {
    return allocated_count;
}

size_t Manager::starting_index() const {
    return first_index;
}

void Manager::set_starting_index(size_t starting_index) {
    first_index = starting_index;
}

size_t Manager::offset() const {
    return file_offset;
}

void Manager::set_offset(size_t offset) {
    file_offset = offset;
}

ValueDomain Manager::domain() const {
    return value_domain;
}

Policy Manager::policy() const {
    return storage_policy;
}

uint64_t* Manager::data() const {
    return values;
}

void Manager::dump_runtime_state(FILE* file) const {
    if (file == nullptr) {
        return;
    }
    std::fwrite(&value_count, sizeof(value_count), 1, file);
    std::fwrite(&allocated_count, sizeof(allocated_count), 1, file);
    std::fwrite(&first_index, sizeof(first_index), 1, file);
    std::fwrite(&file_offset, sizeof(file_offset), 1, file);
    std::fwrite(&dynamic_mode_enabled, sizeof(dynamic_mode_enabled), 1, file);
    std::fwrite(&prediction_sample_count, sizeof(prediction_sample_count), 1, file);
}

void Manager::load_runtime_state(FILE* file) {
    if (file == nullptr) {
        return;
    }
    std::fread(&value_count, sizeof(value_count), 1, file);
    std::fread(&allocated_count, sizeof(allocated_count), 1, file);
    std::fread(&first_index, sizeof(first_index), 1, file);
    std::fread(&file_offset, sizeof(file_offset), 1, file);
    std::fread(&dynamic_mode_enabled, sizeof(dynamic_mode_enabled), 1, file);
    std::fread(&prediction_sample_count, sizeof(prediction_sample_count), 1, file);
}

void Manager::reset_prediction_state() {
    dynamic_mode_enabled = false;
    prediction_sample_count = 0;
}

void Manager::note_prediction_sample(uint64_t) {
    prediction_sample_count++;
}

}

/** Methods Pertaining to the base SubArray Class */
namespace pallas {

SubArrayBase::SubArrayBase(ValueDomain domain, Policy policy, SubArrayBase* previous)
    : prev(previous), value_domain(domain), manager(domain, policy) {
    if (prev != nullptr) {
        prev->next = this;
        manager.set_starting_index(prev->manager.starting_index() + prev->manager.size());
    }
}

SubArrayBase::~SubArrayBase() = default;

uint64_t SubArrayBase::at(size_t pos) const {
    return manager.at(pos);
}

uint64_t SubArrayBase::operator[](size_t pos) const {
    return manager[pos];
}

void SubArrayBase::copy_to_array(uint64_t* given_array) const {
    manager.copy_to_array(given_array);
}

ValueDomain SubArrayBase::domain() const {
    return value_domain;
}

Policy SubArrayBase::policy() const {
    return manager.policy();
}

size_t SubArrayBase::size() const {
    return manager.size();
}

size_t SubArrayBase::capacity() const {
    return manager.capacity();
}

size_t SubArrayBase::starting_index() const {
    return manager.starting_index();
}

size_t SubArrayBase::offset() const {
    return manager.offset();
}

void SubArrayBase::set_offset(size_t offset) {
    manager.set_offset(offset);
}

}

/** Methods Peratining to the TimeSubArray Class */
namespace pallas {

TimeSubArray::TimeSubArray(Policy policy, TimeSubArray* previous)
    : SubArrayBase(ValueDomain::Timestamp, policy, previous) {}

AddStatus TimeSubArray::add(uint64_t val) {
    if (manager.size() == 0) {
        first_timestamp = val;
    }
    
    auto status = manager.add_raw(val);
    pallas_assert(status == AddStatus::Ok);
    last_timestamp = val;
    manager.note_prediction_sample(val);
    return status;
}

uint64_t TimeSubArray::first_value() const {
    return first_timestamp;
}

uint64_t TimeSubArray::last_value() const {
    return last_timestamp;
}

}

/** Methods Peratining to the DurationSubArray Class */
namespace pallas {

DurationSubArray::DurationSubArray(Policy policy, DurationSubArray* previous)
    : SubArrayBase(ValueDomain::Duration, policy, previous) {}

AddStatus DurationSubArray::add(uint64_t val) {
    auto status = manager.add_raw(val);
    pallas_assert(status == AddStatus::Ok);
    update_statistics();
    manager.note_prediction_sample(val);
    return status;
}

void DurationSubArray::update_statistics() {
    const uint64_t current_value = manager.data()[manager.size() - 1];
    min_duration = (current_value < min_duration) ? current_value : min_duration;
    max_duration = (current_value > max_duration) ? current_value : max_duration;
    mean_duration += current_value;
}

void DurationSubArray::final_update_mean() {
    if (manager.size() == 0) {
        return;
    }
    mean_duration /= manager.size();
    pallas_assert_inferior_equal(mean_duration, max_duration);
    pallas_assert_inferior_equal(min_duration, mean_duration);
}

uint64_t DurationSubArray::min_value() const {
    return min_duration;
}

uint64_t DurationSubArray::max_value() const {
    return max_duration;
}

uint64_t DurationSubArray::mean_value() const {
    return mean_duration;
}

}  // namespace pallas
