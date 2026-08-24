/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <algorithm>
#include <sstream>

#ifdef BMARK
#include "pallas/linked_vector/pallas_bmark.h"
#endif

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/linked_vector/pallas_linked_vector.h"

/** Methods Pertaining to LinkedVectorBase class */
namespace pallas {

const SubArrayBase* RecentSubArrayCache::lookup(size_t index, uint64_t& probes) const {
    for (auto& entry : entries) {
        if (entry.subarray == nullptr) {
            continue;
        }
        probes++;
        const size_t begin = entry.subarray->starting_index();
        const size_t end = begin + entry.subarray->size();
        if (index >= begin && index < end) {
            entry.hits++;
            entry.generation = ++generation;
            return entry.subarray;
        }
    }
    return nullptr;
}

void RecentSubArrayCache::remember(SubArrayBase* subarray) const {
    if (subarray == nullptr) {
        return;
    }

    Entry* eviction_candidate = nullptr;
    for (auto& entry : entries) {
        if (entry.subarray == subarray) {
            entry.hits++;
            entry.generation = ++generation;
            return;
        }
        if (entry.subarray == nullptr) {
            eviction_candidate = &entry;
            break;
        }
        if (eviction_candidate == nullptr || entry.hits < eviction_candidate->hits ||
            (entry.hits == eviction_candidate->hits && entry.generation < eviction_candidate->generation)) {
            eviction_candidate = &entry;
        }
    }

    if (eviction_candidate != nullptr) {
        eviction_candidate->subarray = subarray;
        eviction_candidate->hits = 1;
        eviction_candidate->generation = ++generation;
    }
}

/** Constructor and Destructors */

LinkedVectorBase::LinkedVectorBase(ParameterHandler& p, ValueDomain domain, StoragePolicy _policy)
    : parameter_handler(p), value_domain(domain), storage_policy(_policy) {}

/*
 * NOTE:
 * I intentionally make teardown faster by letting LinkedVectorBase directly free the
 * loaded SubArray payloads it owns, instead of asking
 * ParameterHandler::subvector_queue to search for and erase those references
 * one by one. This relies on the current lifecycle assumption that LinkedVectorBase
 * teardown only happens at the end, after analysis is done, so queue entries
 * referring to those subarrays may temporarily dangle until the queue itself is
 * destroyed shortly afterwards. If the API later grows a need for mid-lifetime
 * LV cleanup, we should reintroduce a separate destruction path that uses the
 * older queue-synchronized logic.
 */
LinkedVectorBase::~LinkedVectorBase() {
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

void LinkedVectorBase::ensure_hbuffer(size_t bytes) {
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

void LinkedVectorBase::append_subarray_index(SubArrayBase* subarray) {
    if (subarray == nullptr) {
        return;
    }
    subarray_index.push_back(subarray);
}

void LinkedVectorBase::rebuild_subarray_index() {
    subarray_index.clear();
    subarray_index.reserve(subarray_total);
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray_index.push_back(subarray);
    }
    recent_subarrays.clear();
}

void LinkedVectorBase::evict_loaded_subarrays() {
    while (parameter_handler.loaded_durations_size > parameter_handler.max_memory_durations &&
           !parameter_handler.subvector_queue.empty()) {
        auto* temp = static_cast<SubArrayBase*>(parameter_handler.subvector_queue.front());
        parameter_handler.subvector_queue.pop_front();
        if (temp != nullptr && temp->has_values()) {
            const auto evicted_bytes = static_cast<uint64_t>(temp->mem_size() * sizeof(uint64_t));
            parameter_handler.loaded_durations_size -= evicted_bytes;
#ifdef BMARK
            bmark_note_subarray_evict(temp->get_bmark_family(), evicted_bytes);
#endif
            temp->free_values();
            loaded_subarrays.erase(temp);
        }
    }
}

SubArrayBase* LinkedVectorBase::find_subarray(size_t pos) {
    return const_cast<SubArrayBase*>(static_cast<const LinkedVectorBase*>(this)->find_subarray(pos));
}

const SubArrayBase* LinkedVectorBase::find_subarray(size_t pos) const {
    uint64_t steps = 0;
    if (const auto* cached = recent_subarrays.lookup(pos, steps)) {
#ifdef BMARK
        bmark_note_find_subarray(benchmark_family, steps);
#endif
        return cached;
    }

    if (subarray_index.empty() && first != nullptr) {
        const_cast<LinkedVectorBase*>(this)->rebuild_subarray_index();
    }

    size_t left = 0;
    size_t right = subarray_index.size();
    while (left < right) {
        steps++;
        const size_t mid = left + (right - left) / 2;
        const auto* subarray = subarray_index[mid];
        const size_t begin = subarray->starting_index();
        const size_t end = begin + subarray->size();
        if (pos < begin) {
            right = mid;
            continue;
        }
        if (pos >= end) {
            left = mid + 1;
            continue;
        }
        recent_subarrays.remember(subarray_index[mid]);
#ifdef BMARK
        bmark_note_find_subarray(benchmark_family, steps);
#endif
        return subarray;
    }
#ifdef BMARK
    bmark_note_find_subarray(benchmark_family, steps);
#endif
    return nullptr;
}

/** Value Access and Materialization */

uint64_t LinkedVectorBase::at(size_t pos) const {
#ifdef BMARK
    BmarkScopedTimer timer(benchmark_family, BmarkMetric::At);
#endif
    if (pos >= value_count) {
        pallas_error("Wrong index (%lu) compared to vector size (%lu)\n", pos, value_count);
    }
    return operator[](pos);
}

uint64_t LinkedVectorBase::operator[](size_t pos) const {
#ifdef BMARK
    BmarkScopedTimer timer(benchmark_family, BmarkMetric::Operator);
#endif
    uint64_t cached_value = 0;
    if (recent_values.lookup(pos, cached_value)) {
#ifdef BMARK
        bmark_note_recent_value_lookup(benchmark_family, true);
#endif
        // pallas_log(DebugLevel::Error, "LV recent cache hit: pos=%zu value=%" PRIu64 "\n", pos, cached_value);
        return cached_value;
    }
#ifdef BMARK
    bmark_note_recent_value_lookup(benchmark_family, false);
#endif
    // pallas_log(DebugLevel::Error, "LV recent cache miss: pos=%zu\n", pos);

    auto* subarray = const_cast<SubArrayBase*>(find_subarray(pos));
    if (subarray == nullptr) {
        pallas_error("Wrong index (%lu) compared to vector size (%lu)\n", pos, value_count);
    }
    if (!subarray->has_values()) {
        if (value_domain == ValueDomain::Timestamp) {
            //auto* time_subarray = static_cast<const TimeSubArray*>(subarray);
            if (pos == subarray->starting_index()) {
                const auto value = subarray->first_value();
                recent_values.push(pos, value);
                return value;
            }
            if (pos == subarray->starting_index() + subarray->size() - 1) {
                const auto value = subarray->last_value();
                recent_values.push(pos, value);
                return value;
            }
        }
        const_cast<LinkedVectorBase*>(this)->evict_loaded_subarrays();
        const_cast<LinkedVectorBase*>(this)->load_data(subarray);
#ifdef BMARK
        const auto loaded_bytes = static_cast<uint64_t>(subarray->mem_size() * sizeof(uint64_t));
        bmark_note_subarray_load(benchmark_family, loaded_bytes, loaded_bytes);
#endif
        const_cast<LinkedVectorBase*>(this)->loaded_subarrays.insert(subarray);
    }
    const auto value = subarray->at(pos);
    recent_values.push(pos, value);
    return value;
}

uint64_t LinkedVectorBase::front() const {
    if (empty()) {
        pallas_error("Trying to access the front of an empty vector\n");
    }
    return at(0);
}

uint64_t LinkedVectorBase::back() const {
    if (empty()) {
        pallas_error("Trying to access the back of an empty vector\n");
    }
    return at(value_count - 1);
}

uint64_t* LinkedVectorBase::as_flat_array() const {
    const_cast<LinkedVectorBase*>(this)->load_all();
    auto* flat_array = new uint64_t[value_count];
    size_t copied_values = 0;
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->copy_values(flat_array + copied_values);
        copied_values += subarray->size();
    }
    return flat_array;
}

std::string LinkedVectorBase::values_to_string() const {
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

std::vector<StoragePolicy> LinkedVectorBase::get_sub_array_policies() const {
    std::vector<StoragePolicy> policies;
    policies.reserve(subarray_total);
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        policies.push_back(subarray->storage_policy());
    }
    return policies;
}

std::vector<StoragePolicy> LinkedVectorBase::get_loaded_sub_array_policies() const {
    std::vector<StoragePolicy> policies;
    policies.reserve(loaded_subarrays.size());
    for (auto* subarray : loaded_subarrays) {
        if (subarray != nullptr && subarray->has_values()) {
            policies.push_back(subarray->storage_policy());
        }
    }
    return policies;
}

void LinkedVectorBase::load_all() {
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
void LinkedVectorBase::free_data() {
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

void LinkedVectorBase::reset_offsets() {
    for (auto* subarray = first; subarray != nullptr; subarray = subarray->next_subarray()) {
        subarray->set_offset(0);
    }
}

/** Policy and Configuration Control */

bool LinkedVectorBase::apply_storage_policy() {
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
        append_subarray_index(last);
        return true;
    }

    if (last->storage_policy() == storage_policy) {
        return false;
    }

    if (last->size() == 0) {
        auto* previous = last->previous_subarray();
        delete last;
        last = create_subarray(previous);
        if (previous == nullptr) {
            first = last;
        }
        rebuild_subarray_index();
        return true;
    }

    last = create_subarray(last);
    subarray_total++;
    append_subarray_index(last);
    return true;
}

}

/** Methods Pertaining to TimeLinkedVector  class */
namespace pallas {

/** Constructors */

TimeLinkedVector ::TimeLinkedVector (ParameterHandler& p)
    : TimeLinkedVector (p, p.getStoragePolicy()) {}

TimeLinkedVector ::TimeLinkedVector (ParameterHandler& p, StoragePolicy _policy)
    : LinkedVectorBase(p, ValueDomain::Timestamp, _policy) {
    first = create_subarray(nullptr);
    last = first;
    subarray_total = 1;
    append_subarray_index(last);
}

/** SubArray Creation */

SubArrayBase* TimeLinkedVector ::create_subarray(SubArrayBase* previous) const {
    // This is only the runtime subarray-dispatch point for TimeLinkedVector .
    // The storage-policy encoding itself is handled separately in the subarray header path.
    #if 0
    if (storage_policy == StoragePolicy::Lossy) {
        switch (parameter_handler.getTimeLossyPolicy()) {
            case LossyPolicy::PLA4:
                const_cast<TimeLinkedVector *>(this)->ensure_hbuffer(GammaBlockStats::helper_buffer_bytes());
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
    #endif

    return SubArrayBase::create_subarray(previous,
                            &parameter_handler,
                            const_cast<TimeLinkedVector *>(this));
}

/** Value Insertion */

AddStatus TimeLinkedVector ::add(uint64_t val) {
#ifdef BMARK
    BmarkScopedTimer timer(benchmark_family, BmarkMetric::Add);
#endif
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
        append_subarray_index(last);
    }

    const size_t insert_index = value_count;
    auto status = last->add(val);
    if (status == AddStatus::Full || status == AddStatus::Outlier) {
        last = create_subarray(last);
        subarray_total++;
        append_subarray_index(last);
        status = last->add(val);
    }

    if (status == AddStatus::Ok) {
        recent_values.push(insert_index, val);
        value_count++;
    }
    return status;
}

/** Queries and Stringification */

std::string TimeLinkedVector ::to_string() const {
    return values_to_string();
}

std::vector<double> TimeLinkedVector ::getWeights(pallas_timestamp_t start, pallas_timestamp_t end) const {
    auto output = std::vector<double>();
    auto* current = first;
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
        current = current->next_subarray();
    }
    return output;
}

size_t TimeLinkedVector ::getFirstOccurrenceBefore(pallas_timestamp_t ts) const {
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

/** Methods Pertaining to DurationLinkedVector  class */
namespace pallas {

/** Constructors */

DurationLinkedVector ::DurationLinkedVector (ParameterHandler& p)
    : DurationLinkedVector (p, p.getStoragePolicy()) {}

DurationLinkedVector ::DurationLinkedVector (ParameterHandler& p, StoragePolicy _policy)
    : LinkedVectorBase(p, ValueDomain::Duration, _policy) {
    first = create_subarray(nullptr);
    last = first;
    subarray_total = 1;
    append_subarray_index(last);
}

/** SubArray Creation */

SubArrayBase* DurationLinkedVector ::create_subarray(SubArrayBase* previous) const {
    // Let SubArrayBase resolve the active duration lossy variant from the
    // parameter handler so Spike4/8/16/32 can instantiate their manager.
    return SubArrayBase::create_subarray(previous,
                                        &parameter_handler,
                                        const_cast<DurationLinkedVector *>(this));
}

/** Value Insertion */

AddStatus DurationLinkedVector ::add(uint64_t val) {
#ifdef BMARK
    BmarkScopedTimer timer(benchmark_family, BmarkMetric::Add);
#endif
    if (last == nullptr) {
        first = create_subarray(nullptr);
        last = first;
        subarray_total = 1;
        append_subarray_index(last);
    }

    const size_t insert_index = value_count;
    auto status = last->add(val);
    
    if (status == AddStatus::Full || status == AddStatus::Outlier) {
        //last->final_update_mean();
        last = create_subarray(last);
        subarray_total++;
        append_subarray_index(last);
        status = last->add(val);
    }

    if (status == AddStatus::Ok) {
        recent_values.push(insert_index, val);
        value_count++;
        min_duration = std::min(min_duration, val);
        max_duration = std::max(max_duration, val);
        if (mean_duration_is_finalized && value_count > 1) {
            mean_duration *= (value_count - 1);
        }
        mean_duration_is_finalized = false;
        mean_duration += val;
    }
    return status;
}

/** Aggregate Updates */

void DurationLinkedVector ::final_update_mean() {
    if (value_count == 0 || mean_duration_is_finalized) {
        return;
    }
    mean_duration /= value_count;
    mean_duration_is_finalized = true;
}

/** Queries and Stringification */

pallas_duration_t DurationLinkedVector ::weightedSum(std::vector<double>& weights) const {
    if (weights.empty()) {
        return 0;
    }

    pallas_duration_t result = 0;
    size_t index = 0;
    for (auto* subarray = first; subarray != nullptr && index < weights.size(); subarray = subarray->next_subarray(), ++index) {
        result += static_cast<pallas_duration_t>(weights[index] * subarray->subarray_stats().mean_duration() * subarray->size());
    }
    return result;
}

pallas_duration_t DurationLinkedVector ::computeDurationBetween(size_t start_index, size_t end_index) const {
    pallas_duration_t total = 0;
    for (size_t i = start_index; i < end_index && i < value_count; ++i) {
        total += at(i);
    }
    return total;
}

std::string DurationLinkedVector ::to_string() const {
    std::ostringstream stream;
    stream << values_to_string() << " { " << min_duration << ", " << mean_duration << ", " << max_duration << " }";
    return stream.str();
}

uint64_t DurationLinkedVector ::min_value() const {
    return min_duration;
}

uint64_t DurationLinkedVector ::max_value() const {
    return max_duration;
}

uint64_t DurationLinkedVector ::mean_value() const {
    return mean_duration;
}

}  // namespace pallas
