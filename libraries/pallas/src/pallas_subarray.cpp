/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_serialisation.h"
#include "pallas/utils/pallas_subarray.h"

extern size_t numberPreRawBytes;

/** Methods Pertaining to the policy manager of the SubArray */
namespace pallas {

namespace {

std::unique_ptr<Manager> make_manager(ValueDomain, StoragePolicy policy) {
    switch (policy) {
        case StoragePolicy::None:
        case StoragePolicy::Delta:
        case StoragePolicy::Lossy:
            return std::make_unique<NoneManager>();
    }

    return std::make_unique<NoneManager>();
}

}  // namespace

Manager::~Manager() = default;

void Manager::dump_runtime_state(const SubArrayBase& subarray, FILE* info_file) const {
    if (info_file == nullptr) {
        return;
    }
    std::fwrite(&subarray.value_count, sizeof(subarray.value_count), 1, info_file);
    std::fwrite(&subarray.allocated_count, sizeof(subarray.allocated_count), 1, info_file);
    std::fwrite(&subarray.first_index, sizeof(subarray.first_index), 1, info_file);
    std::fwrite(&subarray.file_offset, sizeof(subarray.file_offset), 1, info_file);
}

void Manager::load_runtime_state(SubArrayBase& subarray, FILE* info_file) const {
    if (info_file == nullptr) {
        return;
    }
    size_t loaded_allocated_count = 0;
    std::fread(&subarray.value_count, sizeof(subarray.value_count), 1, info_file);
    std::fread(&loaded_allocated_count, sizeof(loaded_allocated_count), 1, info_file);
    std::fread(&subarray.first_index, sizeof(subarray.first_index), 1, info_file);
    std::fread(&subarray.file_offset, sizeof(subarray.file_offset), 1, info_file);

    if (loaded_allocated_count != subarray.allocated_count) {
        delete[] subarray.values;
        subarray.allocated_count = loaded_allocated_count;
        subarray.values = (subarray.allocated_count == 0) ? nullptr : new uint64_t[subarray.allocated_count];
    }
}

}

/** Methods Peratining to NoneManger Class */
namespace pallas {

size_t NoneManager::recommended_capacity(ValueDomain, StoragePolicy policy) const {
    return DEFAULT_VECTOR_SIZE;
}

AddStatus NoneManager::add(SubArrayBase& subarray, uint64_t val) const {
    if (subarray.value_count >= subarray.allocated_count) {
        return AddStatus::Full;
    }

    subarray.values[subarray.value_count] = val;
    subarray.value_count++;
    return AddStatus::Ok;
}

uint64_t NoneManager::at(const SubArrayBase& subarray, size_t pos) const {
    if (!subarray.contains(pos)) {
        pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, subarray.first_index, subarray.value_count);
    }
    return subarray.values[subarray.local_index(pos)];
}

void NoneManager::copy_to_array(const SubArrayBase& subarray, uint64_t* given_array) const {
    std::memcpy(given_array, subarray.values, subarray.value_count * sizeof(uint64_t));
}

void NoneManager::write_data(SubArrayBase& subarray, FILE* data_file, const ParameterHandler* parameter_handler) const {
    if (data_file == nullptr || parameter_handler == nullptr || subarray.values == nullptr) {
        return;
    }

    const long current_offset = std::ftell(data_file);
    if (current_offset >= 0) {
        subarray.file_offset = static_cast<size_t>(current_offset);
    }

    numberPreRawBytes += subarray.size() * sizeof(uint64_t);
    _pallas_compress_write(subarray.values, subarray.allocated_count, data_file, parameter_handler);
    subarray.free_values();
}

} 

/** Methods Pertaining to the base SubArray Class */
namespace pallas {

SubArrayBase::SubArrayBase(ValueDomain domain, StoragePolicy policy, SubArrayBase* previous)
    : prev(previous),
      value_domain(domain),
      storage_policy(policy),
      manager(make_manager(domain, policy)),
      allocated_count(manager->recommended_capacity(domain, policy)),
      values(new uint64_t[allocated_count]) {
    if (prev != nullptr) {
        prev->next = this;
        first_index = prev->first_index + prev->value_count;
    }
}

SubArrayBase::~SubArrayBase() {
    free_values();
}

bool SubArrayBase::contains(size_t pos) const {
    return pos >= first_index && pos < first_index + value_count;
}

size_t SubArrayBase::local_index(size_t pos) const {
    return pos - first_index;
}

uint64_t* SubArrayBase::raw_values() {
    return values;
}

void SubArrayBase::free_values() {
    delete[] values;
    values = nullptr;
}

uint64_t SubArrayBase::at(size_t pos) const {
    return manager->at(*this, pos);
}

uint64_t SubArrayBase::operator[](size_t pos) const {
    return at(pos);
}

void SubArrayBase::copy_to_array(uint64_t* given_array) const {
    manager->copy_to_array(*this, given_array);
}

ValueDomain SubArrayBase::domain() const {
    return value_domain;
}

StoragePolicy SubArrayBase::policy() const {
    return storage_policy;
}

size_t SubArrayBase::size() const {
    return value_count;
}

size_t SubArrayBase::capacity() const {
    return allocated_count;
}

size_t SubArrayBase::starting_index() const {
    return first_index;
}

size_t SubArrayBase::offset() const {
    return file_offset;
}

SubArrayBase* SubArrayBase::next_subarray() const {
    return next;
}

SubArrayBase* SubArrayBase::previous_subarray() const {
    return prev;
}

void SubArrayBase::set_offset(size_t offset) {
    file_offset = offset;
}

void SubArrayBase::load_runtime_state(FILE* info_file) {
    manager->load_runtime_state(*this, info_file);
}

}

/** Methods Peratining to the TimeSubArray Class */
namespace pallas {

TimeSubArray::TimeSubArray(StoragePolicy policy, TimeSubArray* previous)
    : SubArrayBase(ValueDomain::Timestamp, policy, previous) {}

AddStatus TimeSubArray::add(uint64_t val) {
    if (value_count == 0) {
        first_timestamp = val;
    }

    auto status = manager->add(*this, val);
    pallas_assert(status == AddStatus::Ok);
    last_timestamp = val;
    return status;
}

void TimeSubArray::write_data(FILE* file, const ParameterHandler* parameter_handler) {
    manager->write_data(*this, file, parameter_handler);
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

DurationSubArray::DurationSubArray(StoragePolicy policy, DurationSubArray* previous)
    : SubArrayBase(ValueDomain::Duration, policy, previous) {}

AddStatus DurationSubArray::add(uint64_t val) {
    auto status = manager->add(*this, val);
    pallas_assert(status == AddStatus::Ok);
    update_statistics();
    return status;
}

void DurationSubArray::write_data(FILE* file, const ParameterHandler* parameter_handler) {
    manager->write_data(*this, file, parameter_handler);
}

void DurationSubArray::update_statistics() {
    const uint64_t current_value = values[value_count - 1];
    min_duration = (current_value < min_duration) ? current_value : min_duration;
    max_duration = (current_value > max_duration) ? current_value : max_duration;
    mean_duration += current_value;
}

void DurationSubArray::final_update_mean() {
    if (value_count == 0) {
        return;
    }
    mean_duration /= value_count;
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
