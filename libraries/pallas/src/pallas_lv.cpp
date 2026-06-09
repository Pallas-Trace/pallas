/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <algorithm>
#include <sstream>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_lv.h"

/** Methods Pertaining to base LV class */
namespace pallas {

LVBase::LVBase(ParameterHandler& p, ValueDomain domain, StoragePolicy preferred_policy)
    : parameter_handler(p), value_domain(domain), preferred_storage_policy(preferred_policy) {}

LVBase::~LVBase() {
    auto* current = first;
    while (current != nullptr) {
        auto* next = current->next_subarray();
        delete current;
        current = next;
    }
}

SubArrayBase* LVBase::find_subarray(size_t pos) {
    return const_cast<SubArrayBase*>(static_cast<const LVBase*>(this)->find_subarray(pos));
}

const SubArrayBase* LVBase::find_subarray(size_t pos) const {
    for (auto* subarray = last; subarray != nullptr; subarray = subarray->previous_subarray()) {
        const size_t begin = subarray->starting_index();
        const size_t end = begin + subarray->size();
        if (pos >= begin && pos < end) {
            return subarray;
        }
    }
    return nullptr;
}

uint64_t LVBase::at(size_t pos) const {
    auto* subarray = find_subarray(pos);
    if (subarray == nullptr) {
        pallas_error("Wrong index (%lu) compared to vector size (%lu)\n", pos, value_count);
    }
    return subarray->at(pos);
}

uint64_t LVBase::operator[](size_t pos) const {
    return at(pos);
}

uint64_t LVBase::front() const {
    if (empty()) {
        pallas_error("Trying to access the front of an empty vector\n");
    }
    return at(0);
}

uint64_t LVBase::back() const {
    if (empty()) {
        pallas_error("Trying to access the back of an empty vector\n");
    }
    return at(value_count - 1);
}

uint64_t* LVBase::as_flat_array() const {
    auto* flat_array = new uint64_t[value_count];
    size_t copied_values = 0;
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->copy_to_array(flat_array + copied_values);
        copied_values += subarray->size();
    }
    return flat_array;
}

std::string LVBase::values_to_string() const {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < value_count; ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << at(i);
    }
    stream << "]";
    return stream.str();
}

void LVBase::reset_offsets() {
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->set_offset(0);
    }
}

}

/** Methods Pertaining to TimeLV class */
namespace pallas {

TimeLV::TimeLV(ParameterHandler& p)
    : TimeLV(p, p.getStoragePolicy()) {}

TimeLV::TimeLV(ParameterHandler& p, StoragePolicy preferred_policy)
    : LVBase(p, ValueDomain::Timestamp, preferred_policy) {
    first = create_subarray(nullptr);
    last = first;
    subarray_total = 1;
}

AddStatus TimeLV::add(uint64_t val) {
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
    }

    auto status = last->add(val);
    if (status == AddStatus::Full) {
        last = create_subarray(last);
        subarray_total++;
        status = last->add(val);
    }

    if (status == AddStatus::Ok) {
        value_count++;
    }
    return status;
}

std::string TimeLV::to_string() const {
    return values_to_string();
}

std::vector<double> TimeLV::getWeights(pallas_timestamp_t, pallas_timestamp_t) const {
    return {};
}

size_t TimeLV::getFirstOccurrenceBefore(pallas_timestamp_t ts) const {
    if (empty()) {
        return 0;
    }

    size_t result = 0;
    for (size_t i = 0; i < value_count; ++i) {
        if (at(i) <= ts) {
            result = i;
        } else {
            break;
        }
    }
    return result;
}

SubArrayBase* TimeLV::create_subarray(SubArrayBase* previous) const {
    return new TimeSubArray(preferred_storage_policy, static_cast<TimeSubArray*>(previous));
}

}

/** Methods Pertaining to DurationLV class */
namespace pallas {

DurationLV::DurationLV(ParameterHandler& p)
    : DurationLV(p, p.getStoragePolicy()) {}

DurationLV::DurationLV(ParameterHandler& p, StoragePolicy preferred_policy)
    : LVBase(p, ValueDomain::Duration, preferred_policy) {
    first = create_subarray(nullptr);
    last = first;
    subarray_total = 1;
}

AddStatus DurationLV::add(uint64_t val) {
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
    }

    auto status = last->add(val);
    if (status == AddStatus::Full) {
        static_cast<DurationSubArray*>(last)->final_update_mean();
        last = create_subarray(last);
        subarray_total++;
        status = last->add(val);
    }

    if (status == AddStatus::Ok) {
        value_count++;
        min_duration = std::min(min_duration, val);
        max_duration = std::max(max_duration, val);
        mean_duration += val;
    }
    return status;
}

void DurationLV::final_update_mean() {
    if (value_count == 0) {
        return;
    }
    mean_duration /= value_count;
}

pallas_duration_t DurationLV::weightedSum(std::vector<double>& weights) const {
    if (weights.empty()) {
        return 0;
    }

    pallas_duration_t result = 0;
    size_t index = 0;
    for (auto* subarray = first; subarray != nullptr && index < weights.size(); subarray = subarray->next_subarray(), ++index) {
        auto* duration_subarray = static_cast<const DurationSubArray*>(subarray);
        result += static_cast<pallas_duration_t>(weights[index] * duration_subarray->mean_value());
    }
    return result;
}

pallas_duration_t DurationLV::computeDurationBetween(size_t start_index, size_t end_index) const {
    pallas_duration_t total = 0;
    for (size_t i = start_index; i < end_index && i < value_count; ++i) {
        total += at(i);
    }
    return total;
}

std::string DurationLV::to_string() const {
    std::ostringstream stream;
    stream << values_to_string() << " { " << min_duration << ", " << mean_duration << ", " << max_duration << " }";
    return stream.str();
}

uint64_t DurationLV::min_value() const {
    return min_duration;
}

uint64_t DurationLV::max_value() const {
    return max_duration;
}

uint64_t DurationLV::mean_value() const {
    return mean_duration;
}

SubArrayBase* DurationLV::create_subarray(SubArrayBase* previous) const {
    return new DurationSubArray(preferred_storage_policy, static_cast<DurationSubArray*>(previous));
}

}  // namespace pallas
