/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <algorithm>
#include <sstream>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_lv.h"

/** Methods Pertaining to LVBase class */
namespace pallas {

/** Constructor and Destructors */

LVBase::LVBase(ParameterHandler& p, ValueDomain domain, StoragePolicy _policy)
    : parameter_handler(p), value_domain(domain), storage_policy(_policy) {}

LVBase::~LVBase() { 
    free_data();
    auto* current = first;
    while (current != nullptr) {
        auto* next = current->next_subarray();
        delete current;
        current = next;
    }
    delete[] static_cast<uint8_t*>(hbuffer);
    hbuffer = nullptr;
    hbuffer_bytes = 0;
}

/** Internal Helpers */

void LVBase::ensure_hbuffer(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (hbuffer != nullptr && hbuffer_bytes >= bytes) {
        return;
    }
    delete[] static_cast<uint8_t*>(hbuffer);
    hbuffer = new uint8_t[bytes];
    hbuffer_bytes = bytes;
}

void LVBase::evict_loaded_subarrays() {
    while (parameter_handler.loaded_durations_size > parameter_handler.max_memory_durations &&
           !parameter_handler.subvector_queue.empty()) {
        auto* temp = static_cast<SubArrayBase*>(parameter_handler.subvector_queue.front());
        parameter_handler.subvector_queue.pop_front();
        if (temp != nullptr && temp->has_values()) {
            parameter_handler.loaded_durations_size -= temp->mem_size() * sizeof(uint64_t);
            temp->free_values();
            loaded_subarrays.erase(temp);
        }
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

/** Value Access and Materialization */

uint64_t LVBase::at(size_t pos) const {
    if (pos >= value_count) {
        pallas_error("Wrong index (%lu) compared to vector size (%lu)\n", pos, value_count);
    }
    return operator[](pos);
}

uint64_t LVBase::operator[](size_t pos) const {
    uint64_t cached_value = 0;
    if (recent_values.lookup(pos, cached_value)) {
        // pallas_log(DebugLevel::Error, "LV recent cache hit: pos=%zu value=%" PRIu64 "\n", pos, cached_value);
        return cached_value;
    }
    // pallas_log(DebugLevel::Error, "LV recent cache miss: pos=%zu\n", pos);

    auto* subarray = const_cast<SubArrayBase*>(find_subarray(pos));
    if (subarray == nullptr) {
        pallas_error("Wrong index (%lu) compared to vector size (%lu)\n", pos, value_count);
    }
    if (!subarray->has_values()) {
        if (value_domain == ValueDomain::Timestamp) {
            auto* time_subarray = static_cast<const TimeSubArray*>(subarray);
            if (pos == subarray->starting_index()) {
                const auto value = time_subarray->first_value();
                recent_values.push(pos, value);
                return value;
            }
            if (pos == subarray->starting_index() + subarray->size() - 1) {
                const auto value = time_subarray->last_value();
                recent_values.push(pos, value);
                return value;
            }
        }
        const_cast<LVBase*>(this)->evict_loaded_subarrays();
        const_cast<LVBase*>(this)->load_data(subarray);
        const_cast<LVBase*>(this)->loaded_subarrays.insert(subarray);
    }
    const auto value = subarray->at(pos);
    recent_values.push(pos, value);
    return value;
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
    const_cast<LVBase*>(this)->load_all();
    auto* flat_array = new uint64_t[value_count];
    size_t copied_values = 0;
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->copy_values(flat_array + copied_values);
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

/** Data Residency and Memory Management */

std::vector<StoragePolicy> LVBase::get_sub_array_policies() const {
    std::vector<StoragePolicy> policies;
    policies.reserve(subarray_total);
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        policies.push_back(subarray->policy());
    }
    return policies;
}

std::vector<StoragePolicy> LVBase::get_loaded_sub_array_policies() const {
    std::vector<StoragePolicy> policies;
    policies.reserve(loaded_subarrays.size());
    for (auto* subarray : loaded_subarrays) {
        if (subarray != nullptr && subarray->has_values()) {
            policies.push_back(subarray->policy());
        }
    }
    return policies;
}

void LVBase::load_all() {
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        if (!subarray->has_values()) {
            load_data(subarray);
            loaded_subarrays.insert(subarray);
        }
    }
}

/**
 * Performance note:
 * The previous shutdown path tried to erase every loaded subarray from
 * ParameterHandler::subvector_queue one by one. Since that queue is shared and
 * each erase required a linear search, cleanup could become quadratic and make
 * large traces take hours to shut down. During LV teardown we only need to free
 * the remaining loaded payloads owned by this LV and update the memory counter;
 * the queue itself is cleared later by ParameterHandler teardown.
 */
void LVBase::free_data() {
    if (first == nullptr) {
        return;
    }
    for (auto* subarray : loaded_subarrays) {
        if (subarray->has_values()) {
            const size_t subarray_bytes = subarray->mem_size() * sizeof(uint64_t);
            if (parameter_handler.loaded_durations_size >= subarray_bytes) {
                parameter_handler.loaded_durations_size -= subarray_bytes;
            } else {
                parameter_handler.loaded_durations_size = 0;
            }
            subarray->free_values();
        }
    }
    loaded_subarrays.clear();
}

void LVBase::reset_offsets() {
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->set_offset(0);
    }
}

/** Policy and Configuration Control */

bool LVBase::apply_storage_policy() {
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
        return true;
    }

    if (last->policy() == storage_policy) {
        return false;
    }

    if (last->size() == 0) {
        auto* previous = last->previous_subarray();
        delete last;
        last = create_subarray(previous);
        if (previous == nullptr) {
            first = last;
        }
        return true;
    }

    last = create_subarray(last);
    subarray_total++;
    return true;
}

}

/** Methods Pertaining to TimeLV class */
namespace pallas {

/** Constructors */

TimeLV::TimeLV(ParameterHandler& p)
    : TimeLV(p, p.getStoragePolicy()) {}

TimeLV::TimeLV(ParameterHandler& p, StoragePolicy _policy)
    : LVBase(p, ValueDomain::Timestamp, _policy) {
    first = create_subarray(nullptr);
    last = first;
    subarray_total = 1;
}

/** SubArray Creation */

SubArrayBase* TimeLV::create_subarray(SubArrayBase* previous) const {
    // This is only the runtime subarray-dispatch point for TimeLV.
    // The storage-policy encoding itself is handled separately in the subarray header path.
    if (storage_policy == StoragePolicy::Lossy) {
        switch (parameter_handler.getTimeLossyPolicy()) {
            case LossyPolicy::PLA4:
                const_cast<TimeLV*>(this)->ensure_hbuffer(GammaBlockStats::helper_buffer_bytes());
                break;
            case LossyPolicy::PLA8:
            case LossyPolicy::PLA16:
            case LossyPolicy::PLA32:
                break;
            case LossyPolicy::Spike4:
            case LossyPolicy::Spike8:
            case LossyPolicy::Spike16:
            case LossyPolicy::Spike32:
                pallas_error("Spike lossy policies are not supported for timestamp subarrays in the standalone LV path.\n");
                break;
        }
    }
    return new TimeSubArray(storage_policy,
                            static_cast<TimeSubArray*>(previous),
                            &parameter_handler,
                            const_cast<TimeLV*>(this));
}

/** Value Insertion */

AddStatus TimeLV::add(uint64_t val) {
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
    }

    const size_t insert_index = value_count;
    auto status = last->add(val);
    if (status == AddStatus::Full || status == AddStatus::Outlier) {
        last = create_subarray(last);
        subarray_total++;
        status = last->add(val);
    }

    if (status == AddStatus::Ok) {
        recent_values.push(insert_index, val);
        value_count++;
    }
    return status;
}

/** Queries and Stringification */

std::string TimeLV::to_string() const {
    return values_to_string();
}

std::vector<double> TimeLV::getWeights(pallas_timestamp_t start, pallas_timestamp_t end) const {
    auto output = std::vector<double>();
    auto* current = static_cast<TimeSubArray*>(first);
    double sum = 0;
    // While loop to go through all the SubVectors.
    // Legend:
    //   - : Time spent in current vector but NOT in the window
    //   # : Time spent in current vector AND in the window
    // We store in output the ratio of # / ( - + # )
    // i.e. the ratio of time spent in window over duration of current vector
    while (current != nullptr) {
        if (current->last_value() < start) {
            // first_value ... last_value ... [ start ... end ]
            // --------------------------
            // Completely outside of the range
            output.push_back(0.);
        } else if (end < current->first_value()) {
            // [ start ... end ] .. first_value ... last_value
            //                      --------------------------
            // We're past the boundaries, we can stop searching.
            break;
        } else if (start <= current->first_value() && current->last_value() <= end) {
            // [ start ... first_value ... last_value ... end ]
            //             ##########################
            // Completely inside the bounds
            output.push_back(1.0);
        } else if (current->first_value() < start && end < current->last_value()) {
            // first_value ... [ start ... end ] ... last_value
            // ----------------#################---------------
            // We have to compute the ratio of the two intervals to "guess" the weight of this vector in the total
            output.push_back(static_cast<double>(end - start) / (current->last_value() - current->first_value()));
        } else if (current->first_value() < start && current->last_value() < end) {
            // first_value ... [ start ... last_value ... end ]
            // ----------------######################
            // Same thing except the window ends in the current vector
            output.push_back(static_cast<double>(current->last_value() - start) / (current->last_value() - current->first_value()));
        } else if (current->first_value() <= end && end < current->last_value()) {
            // [ start ... first_value ... end ] ... last_value
            //             #####################---------------
            // Same thing except the window starts in the current vector and isn't entirely contained in it.
            output.push_back(static_cast<double>(end - current->first_value()) / (current->last_value() - current->first_value()));
        } else {
            pallas_error("This is not supposed to happen !\n");
            pallas_error("start=%lu, end=%lu\n", start, end);
        }
        sum += output.back();
        current = static_cast<TimeSubArray*>(current->next_subarray());
    }
    // Then we need to normalize the weight vector
    // UPDATE: We don't actually need to normalize the weight vector
    //
    // For example, a vector formatted like this:
    //          start                   end
    //          |                         |
    // A: [......##][########][#######][##......]
    // B:   [....############]
    // A would have a non-normalized weight of [ .25, 1, 1, .25 ] -> [ .1, .4, .4, 0.1 ]
    // B would have a non-normalized weight of [ .75 ] and that's that
    // if (sum > 1.0) {
    //     for (auto &i: output) {
    //         i /= sum;
    //     }
    // }
    return output;
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

}

/** Methods Pertaining to DurationLV class */
namespace pallas {

/** Constructors */

DurationLV::DurationLV(ParameterHandler& p)
    : DurationLV(p, p.getStoragePolicy()) {}

DurationLV::DurationLV(ParameterHandler& p, StoragePolicy _policy)
    : LVBase(p, ValueDomain::Duration, _policy) {
    first = create_subarray(nullptr);
    last = first;
    subarray_total = 1;
}

/** SubArray Creation */

SubArrayBase* DurationLV::create_subarray(SubArrayBase* previous) const {
    // DurationLV currently keeps the standalone LV path simple by materializing
    // lossy duration storage through the delta-backed duration subarray path.
    const auto effective_policy =
            (storage_policy == StoragePolicy::Lossy) ? StoragePolicy::Delta : storage_policy;
    return new DurationSubArray(effective_policy,
                                static_cast<DurationSubArray*>(previous),
                                &parameter_handler,
                                const_cast<DurationLV*>(this));
}

/** Value Insertion */

AddStatus DurationLV::add(uint64_t val) {
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
    }

    const size_t insert_index = value_count;
    auto status = last->add(val);
    
    if (status == AddStatus::Full || status == AddStatus::Outlier) {
        static_cast<DurationSubArray*>(last)->final_update_mean();
        last = create_subarray(last);
        subarray_total++;
        status = last->add(val);
    }

    if (status == AddStatus::Ok) {
        recent_values.push(insert_index, val);
        value_count++;
        min_duration = std::min(min_duration, val);
        max_duration = std::max(max_duration, val);
        mean_duration += val;
    }
    return status;
}

/** Aggregate Updates */

void DurationLV::final_update_mean() {
    if (value_count == 0) {
        return;
    }
    mean_duration /= value_count;
}

/** Queries and Stringification */

pallas_duration_t DurationLV::weightedSum(std::vector<double>& weights) const {
    if (weights.empty()) {
        return 0;
    }

    pallas_duration_t result = 0;
    size_t index = 0;
    for (auto* subarray = first; subarray != nullptr && index < weights.size(); subarray = subarray->next_subarray(), ++index) {
        auto* duration_subarray = static_cast<const DurationSubArray*>(subarray);
        result += static_cast<pallas_duration_t>(weights[index] * duration_subarray->mean_value() * duration_subarray->size());
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

}  // namespace pallas
