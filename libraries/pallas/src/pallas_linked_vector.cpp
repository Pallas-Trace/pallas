/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_linked_vector.h"
#include "pallas/utils/pallas_log.h"

#define SAME_FOR_BOTH_VECTORS(return_type, function_core) return_type LinkedVector::function_core return_type LinkedDurationVector::function_core


/** Functions pertaining to SubArrayCodec */

namespace pallas {

uint64_t SubArrayCodec::zigzag_encode(int64_t x) {
    return (static_cast<uint64_t>(x) << 1) ^ static_cast<uint64_t>(x >> 63);
}

int64_t SubArrayCodec::zigzag_decode(uint64_t x) {
    return static_cast<int64_t>((x >> 1) ^ static_cast<uint64_t>(-static_cast<int64_t>(x & 1)));
}

size_t NoneCodec::encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const {
    encoded_array = array;  // No encoding, so the encoded array is the same as the original array.
    return size;
}

void NoneCodec::decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, const ParameterHandler* parameter_handler) const {
    decoded_array = encoded_array;
}

void Delta2VintCodecBase::write_varint(uint64_t x, uint8_t*& out) {
    while (x >= 0x80) {
        *out++ = static_cast<uint8_t>((x & 0x7f) | 0x80);
        x >>= 7;
    }
    *out++ = static_cast<uint8_t>(x);
}

uint64_t Delta2VintCodecBase::read_varint(const uint8_t*& p, const uint8_t* end) {
    uint64_t result = 0;
    int shift = 0;
    while (p < end) {
        uint8_t byte = *p++;
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

size_t Delta2VintCodecBase::encode_timestamp(uint64_t* src, size_t size, uint64_t*& encoded_array) const {
    size_t enc_size = 0;

    if (size == 0) {
        encoded_array = nullptr;  // Should never encounter this case
        return enc_size;
    }

    // Worst case: uint64_t varint takes 10 bytes.
    const size_t max_bytes = 10 * size;
    const size_t max_words = (max_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    uint64_t* packed = new uint64_t[max_words];  // not zero-initialized
    uint8_t* begin = reinterpret_cast<uint8_t*>(packed);
    uint8_t* out = begin;
    encoded_array = packed;

    // base timestamp
    write_varint(src[0], out);

    if (size >= 2) {
        uint64_t prev_delta = src[1] - src[0];
        write_varint(prev_delta, out);
        for (size_t i = 2; i < size; ++i) {
            uint64_t cur_delta = src[i] - src[i - 1];
            int64_t ddelta = static_cast<int64_t>(cur_delta) - static_cast<int64_t>(prev_delta);
            write_varint(zigzag_encode(ddelta), out);
            prev_delta = cur_delta;
        }
    }

    const size_t used_bytes = static_cast<size_t>(out - begin);
    enc_size = (used_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    // zero the padding bytes in the final uint64_t word.
    const size_t padded_bytes = enc_size * sizeof(uint64_t);
    if (padded_bytes > used_bytes) {
        std::memset(begin + used_bytes, 0, padded_bytes - used_bytes);
    }

    return enc_size;
}

size_t Delta2VintCodecBase::encode_duration(uint64_t* src, size_t size, uint64_t*& encoded_array) const {
    size_t enc_size = 0;
    if (size == 0) {
        encoded_array = nullptr;  // Should never encounter this case
        return enc_size;
    }

    // Worst case: uint64_t varint takes 10 bytes.
    const size_t max_bytes = 10 * size;
    const size_t max_words = (max_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    uint64_t* packed = new uint64_t[max_words];  // not zero-initialized
    uint8_t* begin = reinterpret_cast<uint8_t*>(packed);
    uint8_t* out = begin;
    encoded_array = packed;

    // base duration
    write_varint(src[0], out);

    if (size >= 2) {
        int64_t prev_delta = static_cast<int64_t>(src[1]) - static_cast<int64_t>(src[0]);
        write_varint(zigzag_encode(prev_delta), out);

        for (size_t i = 2; i < size; ++i) {
            int64_t cur_delta = static_cast<int64_t>(src[i]) - static_cast<int64_t>(src[i - 1]);
            int64_t ddelta = cur_delta - prev_delta;
            write_varint(zigzag_encode(ddelta), out);
            prev_delta = cur_delta;
        }
    }

    const size_t used_bytes = static_cast<size_t>(out - begin);
    enc_size = (used_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    // zero the padding bytes in the final uint64_t word.
    const size_t padded_bytes = enc_size * sizeof(uint64_t);
    if (padded_bytes > used_bytes) {
        std::memset(begin + used_bytes, 0, padded_bytes - used_bytes);
    }

    return enc_size;
}

void Delta2VintCodecBase::decode_timestamp(const uint64_t* encoded_words, size_t enc_size, uint64_t* decoded_array, size_t size) {
    if (size == 0) {
        return;
    }

    pallas_assert(encoded_words != nullptr);
    pallas_assert(enc_size > 0);

    const uint8_t* p = reinterpret_cast<const uint8_t*>(encoded_words);
    const uint8_t* end = p + enc_size * sizeof(uint64_t);

    // base timestamp
    decoded_array[0] = read_varint(p, end);

    if (size == 1) {
        return;
    }

    // first delta, non-negative for timestamps
    uint64_t prev_delta = read_varint(p, end);
    decoded_array[1] = decoded_array[0] + prev_delta;

    for (size_t i = 2; i < size; ++i) {
        int64_t ddelta = zigzag_decode(read_varint(p, end));

        uint64_t cur_delta = static_cast<uint64_t>(static_cast<int64_t>(prev_delta) + ddelta);

        decoded_array[i] = decoded_array[i - 1] + cur_delta;
        prev_delta = cur_delta;
    }
}

void Delta2VintCodecBase::decode_duration(const uint64_t* encoded_words, size_t enc_size, uint64_t* out, size_t size) {
    if (size == 0) {
        return;
    }

    pallas_assert(encoded_words != nullptr);
    pallas_assert(enc_size > 0);

    const uint8_t* p = reinterpret_cast<const uint8_t*>(encoded_words);
    const uint8_t* end = p + enc_size * sizeof(uint64_t);

    // base duration
    out[0] = read_varint(p, end);

    if (size == 1) {
        return;
    }

    // first duration delta can be negative
    int64_t prev_delta = zigzag_decode(read_varint(p, end));
    out[1] = static_cast<uint64_t>(static_cast<int64_t>(out[0]) + prev_delta);

    for (size_t i = 2; i < size; ++i) {
        int64_t ddelta = zigzag_decode(read_varint(p, end));
        int64_t cur_delta = prev_delta + ddelta;

        out[i] = static_cast<uint64_t>(static_cast<int64_t>(out[i - 1]) + cur_delta);

        prev_delta = cur_delta;
    }
}

size_t TimestampDelta2VintCodec::encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const {
    return encode_timestamp(array, size, encoded_array);
}

void TimestampDelta2VintCodec::decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, const ParameterHandler* parameter_handler) const {
    decoded_array = new uint64_t[size];
    decode_timestamp(encoded_array, enc_size, decoded_array, size);
}

size_t DurationDelta2VintCodec::encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const {
    return encode_duration(array, size, encoded_array);
}

void DurationDelta2VintCodec::decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, const ParameterHandler* parameter_handler) const {
    decoded_array = new uint64_t[size];
    decode_duration(encoded_array, enc_size, decoded_array, size);
}

size_t MonotoneLossyCodec::kpercentile_anchor_index(size_t size, size_t anchor_id) {
    pallas_assert(size >= kKPercentileAnchorCount);
    pallas_assert(anchor_id < kKPercentileAnchorCount);
    return (anchor_id * (size - 1) + kKPercentileSegmentCount / 2) / kKPercentileSegmentCount;
}

uint64_t MonotoneLossyCodec::linear_interpolate(uint64_t start_value, uint64_t end_value, size_t offset, size_t span) {
    pallas_assert(span > 0);

    if (start_value <= end_value) {
        uint64_t delta = end_value - start_value;
        __uint128_t scaled_delta = static_cast<__uint128_t>(delta) * offset + span / 2;
        return start_value + static_cast<uint64_t>(scaled_delta / span);
    }

    uint64_t delta = start_value - end_value;
    __uint128_t scaled_delta = static_cast<__uint128_t>(delta) * offset + span / 2;
    return start_value - static_cast<uint64_t>(scaled_delta / span);
}

size_t MonotoneLossyCodec::encode_kpercentile_linear(uint64_t* array, size_t size, uint64_t*& encoded_array) {
    pallas_assert(size >= kKPercentileAnchorCount);

    encoded_array = new uint64_t[kLinearWordCount];

    for (size_t anchor_id = 0; anchor_id < kKPercentileAnchorCount; ++anchor_id) {
        size_t anchor_index = kpercentile_anchor_index(size, anchor_id);
        encoded_array[anchor_id] = array[anchor_index];
    }

    return kLinearWordCount;
}

void MonotoneLossyCodec::decode_kpercentile_linear(const uint64_t* encoded_array, size_t enc_size, uint64_t* decoded_array, size_t size) {
    pallas_assert(size >= kKPercentileAnchorCount);
    pallas_assert(enc_size == kLinearWordCount);
    const uint64_t* anchors = encoded_array;

    for (size_t anchor_id = 0; anchor_id + 1 < kKPercentileAnchorCount; ++anchor_id) {
        size_t start_index = kpercentile_anchor_index(size, anchor_id);
        size_t end_index = kpercentile_anchor_index(size, anchor_id + 1);
        uint64_t start_value = anchors[anchor_id];
        uint64_t end_value = anchors[anchor_id + 1];

        decoded_array[start_index] = start_value;
        for (size_t pos = start_index + 1; pos < end_index; ++pos) {
            decoded_array[pos] =
                linear_interpolate(start_value, end_value, pos - start_index, end_index - start_index);
        }
    }

    decoded_array[size - 1] = anchors[kKPercentileAnchorCount - 1];
}

size_t MonotoneLossyCodec::encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const {
    pallas_assert(parameter_handler != nullptr);

    switch (parameter_handler->getMonotoneLossyVariant()) {
        case MonotoneLossyVariant::Linear:
            return encode_kpercentile_linear(array, size, encoded_array);
        case MonotoneLossyVariant::LinearMeanRep:
        case MonotoneLossyVariant::LinearPchipMeanRep:
        case MonotoneLossyVariant::LinearPchipMeanRepAdaptive:
            pallas_error("MonotoneLossyVariant not yet implemented\n");
        default:
            pallas_error("Invalid MonotoneLossyVariant\n");
    }
}

bool MonotoneLossyCodec::can_encode(uint64_t* array, size_t size) const {
    return size >= kKPercentileAnchorCount;
}

void MonotoneLossyCodec::decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, const ParameterHandler* parameter_handler) const {
    pallas_assert(encoded_array != nullptr);
    pallas_assert(parameter_handler != nullptr);
    decoded_array = new uint64_t[size];

    switch (parameter_handler->getMonotoneLossyVariant()) {
        case MonotoneLossyVariant::Linear:
            decode_kpercentile_linear(encoded_array, enc_size, decoded_array, size);
            return;
        case MonotoneLossyVariant::LinearMeanRep:
        case MonotoneLossyVariant::LinearPchipMeanRep:
        case MonotoneLossyVariant::LinearPchipMeanRepAdaptive:
            pallas_error("MonotoneLossyVariant not yet implemented\n");
        default:
            pallas_error("Invalid MonotoneLossyVariant\n");
    }
}

const SubArrayCodec* get_subarray_codec(SubArrayEncoding encoding) {
    static const NoneCodec none_codec;
    static const TimestampDelta2VintCodec delta2_vint_timestamp_codec;
    static const DurationDelta2VintCodec delta2_vint_duration_codec;
    static const MonotoneLossyCodec monotone_lossy_codec;
    switch (encoding) {
        case SubArrayEncoding::None:
            return &none_codec;

        case SubArrayEncoding::Delta2VintTimestamp:
            return &delta2_vint_timestamp_codec;

        case SubArrayEncoding::Delta2VintDuration:
            return &delta2_vint_duration_codec;

        case SubArrayEncoding::MonotoneLossy:
            return &monotone_lossy_codec;

        default:
            pallas_error("Invalid SubArrayEncoding: %u\n",
                        static_cast<unsigned>(encoding));
            return nullptr;
    }
}

}
namespace pallas {

std::string LinkedVector::to_string() {
    if (size == 0)
        return "[ ]";
    std::ostringstream output;
    output << "[";
    for (size_t i = 0; i < size; i++) {
        if (i != size - 1) {
            output << this->at(i) << ", ";
        }
        else {
            output << this->at(i) << "]";
        }
    }
    return output.str();
}

std::string LinkedDurationVector::to_string() {
    if (size == 0)
        return "[ ]";
    std::ostringstream output;
    output << "[";
    for (size_t i = 0; i < size; i++) {
        if (i != size - 1) {
            output << this->at(i) << ", ";
        }
        else {
            output << this->at(i) << "]";
        }
    }
    if (size >= 2) {
        output << " { " << min << ", " << mean << ", " << max << " }";
    }
    return output.str();
}

LinkedVector::LinkedVector(ParameterHandler& p ) : parameter_handler(p) {
    preferred_sub_arr_encoding = parameter_handler.getTimestampSubArrayEncoding();
    first = new SubArray(DEFAULT_VECTOR_SIZE);
    first->sub_arr_encoding = preferred_sub_arr_encoding;
    last = first;
}

LinkedVector::LinkedVector(ParameterHandler& p, SubArrayEncoding preferred_encoding) : parameter_handler(p) {
    preferred_sub_arr_encoding = preferred_encoding;
    first = new SubArray(DEFAULT_VECTOR_SIZE);
    first->sub_arr_encoding = preferred_sub_arr_encoding;
    last = first;
}

LinkedDurationVector::LinkedDurationVector(ParameterHandler& p ) : parameter_handler(p) {
    preferred_sub_arr_encoding = parameter_handler.getDurationSubArrayEncoding();
    first = new SubArray(DEFAULT_VECTOR_SIZE);
    first->sub_arr_encoding = preferred_sub_arr_encoding;
    last = first;
}

LinkedDurationVector::LinkedDurationVector(ParameterHandler& p, SubArrayEncoding preferred_encoding) : parameter_handler(p) {
    preferred_sub_arr_encoding = preferred_encoding;
    first = new SubArray(DEFAULT_VECTOR_SIZE);
    first->sub_arr_encoding = preferred_sub_arr_encoding;
    last = first;
}

uint64_t* LinkedVector::SubArray::add(uint64_t val) {
    array[size] = val;
    return &array[size++];
}

uint64_t* LinkedDurationVector::SubArray::add(uint64_t val) {
    array[size++] = val;
    update_statistics();
    return &array[size-1];
}

SAME_FOR_BOTH_VECTORS(
  uint64_t&,
  SubArray::at(size_t pos) const {
      if (pos >= starting_index && pos < size + starting_index) {
          return array[pos - starting_index];
      }
      pallas_error("Wrong index (%lu) compared to starting index (%lu) and size (%lu)\n", pos, starting_index, size);
  })

SAME_FOR_BOTH_VECTORS(uint64_t&, SubArray::operator[](size_t pos) const { return array[pos - starting_index]; })

LinkedVector::SubArray::SubArray(size_t size, LinkedVector::SubArray* previous) {
    this->previous = previous;
    starting_index = 0;
    if (previous) {
        previous->next = this;
        starting_index = previous->starting_index + previous->size;
    }
    allocated = size;
    array = new uint64_t[size];
}

LinkedDurationVector::SubArray::SubArray(size_t size, LinkedDurationVector::SubArray* previous) {
    this->previous = previous;
    starting_index = 0;
    if (previous) {
        previous->next = this;
        starting_index = previous->starting_index + previous->size;
    }
    allocated = size;
    array = new uint64_t[size];
}


SAME_FOR_BOTH_VECTORS(, SubArray::~SubArray() { delete[] array; })

SAME_FOR_BOTH_VECTORS(void, SubArray::copy_to_array(uint64_t* given_array) const { memcpy(given_array, array, size * sizeof(uint64_t)); })

void LinkedDurationVector::update_statistics() {
    auto& val = at(size - 1);
    max = std::max(max, val);
    min = std::min(min, val);
    mean += val;
}

void LinkedDurationVector::final_update_mean() {
    mean /= size;
    pallas_assert_inferior_equal(mean, max);
    pallas_assert_inferior_equal(min, mean);
    last->final_update_mean();
}


void LinkedDurationVector::SubArray::update_statistics() {
    auto& val = at(size - 1 + starting_index);
        max = std::max(max, val);
        min = std::min(min, val);
        mean += val;
}

void LinkedDurationVector::SubArray::final_update_mean() {
    mean /= size;
    pallas_assert_inferior_equal(mean, max);
    pallas_assert_inferior_equal(min, mean);
}

uint64_t* LinkedDurationVector::add(uint64_t val) {
    if (this->last->size >= this->last->allocated) {
        last->final_update_mean();
        last = new SubArray(DEFAULT_VECTOR_SIZE, last);
        last->sub_arr_encoding = preferred_sub_arr_encoding;
        n_sub_array++;
    }
    size++;
    auto* out = last->add(val);
    update_statistics();
    return out;
}

uint64_t* LinkedVector::add(uint64_t val) {
    if (this->last->size >= this->last->allocated) {
        last = new SubArray(DEFAULT_VECTOR_SIZE, last);
        last->sub_arr_encoding = preferred_sub_arr_encoding;
        n_sub_array++;
    }
    size++;
    return last->add(val);
}

void LinkedVector::setPreferredSubArrayEncoding(SubArrayEncoding encoding) {
    preferred_sub_arr_encoding = encoding;
    if (last && last->size == 0) {
        last->sub_arr_encoding = encoding;
    }
}

SubArrayEncoding LinkedVector::getPreferredSubArrayEncoding() const {
    return preferred_sub_arr_encoding;
}

std::vector<SubArrayEncoding> LinkedVector::getSubArrayEncodings() const {
    std::vector<SubArrayEncoding> encodings;
    encodings.reserve(n_sub_array);
    for (auto* sub = first; sub != nullptr; sub = sub->next) {
        encodings.push_back(sub->sub_arr_encoding);
    }
    return encodings;
}

std::vector<SubArrayEncoding> LinkedVector::getLoadedSubArrayEncodings() const {
    std::vector<SubArrayEncoding> encodings;
    encodings.reserve(loaded_subarrays.size());
    for (auto* sub = first; sub != nullptr; sub = sub->next) {
        if (sub->array != nullptr) {
            encodings.push_back(sub->sub_arr_encoding);
        }
    }
    return encodings;
}

void LinkedDurationVector::setPreferredSubArrayEncoding(SubArrayEncoding encoding) {
    preferred_sub_arr_encoding = encoding;
    if (last && last->size == 0) {
        last->sub_arr_encoding = encoding;
    }
}

SubArrayEncoding LinkedDurationVector::getPreferredSubArrayEncoding() const {
    return preferred_sub_arr_encoding;
}

std::vector<SubArrayEncoding> LinkedDurationVector::getSubArrayEncodings() const {
    std::vector<SubArrayEncoding> encodings;
    encodings.reserve(n_sub_array);
    for (auto* sub = first; sub != nullptr; sub = sub->next) {
        encodings.push_back(sub->sub_arr_encoding);
    }
    return encodings;
}

std::vector<SubArrayEncoding> LinkedDurationVector::getLoadedSubArrayEncodings() const {
    std::vector<SubArrayEncoding> encodings;
    encodings.reserve(loaded_subarrays.size());
    for (auto* sub = first; sub != nullptr; sub = sub->next) {
        if (sub->array != nullptr) {
            encodings.push_back(sub->sub_arr_encoding);
        }
    }
    return encodings;
}

SAME_FOR_BOTH_VECTORS(void, load_all_data() {
    auto* v = first;
    while (v) {
        load_data(v);
        loaded_subarrays.insert(v);
        v = v->next;
    }
})


SAME_FOR_BOTH_VECTORS(
    uint64_t&,
    at(size_t pos) {
      if (pos >= size) {
          pallas_error("Getting an element whose index (%lu) is bigger than LinkedVector size (%lu)\n", pos, size);
      }
      return operator[](pos);
  })

uint64_t& LinkedVector::operator[](size_t pos) {
    SubArray* correct_sub = last;
    while (pos < correct_sub->starting_index) {
        correct_sub = correct_sub->previous;
    }
    if (correct_sub->array == nullptr) {
        // TODO We should not load data for small vectors ( <= 2 )
        //      This is a small temporary fix which should speed cleanup times
        if (pos == correct_sub->starting_index) {
            return correct_sub->first_value;
        }
        if (pos == correct_sub->starting_index + correct_sub->size - 1) {
            return correct_sub->last_value;
        }
        while (parameter_handler.loaded_durations_size > parameter_handler.max_memory_durations) {
            auto* temp = (SubArray*)parameter_handler.subvector_queue.front();
            parameter_handler.subvector_queue.pop_front();
            delete[] temp->array;
            temp->array = nullptr;
            parameter_handler.loaded_durations_size -= temp->size * sizeof(uint64_t);
        }
        load_data(correct_sub);
        loaded_subarrays.insert(correct_sub);
    }
    return (*correct_sub)[pos];
}

uint64_t& LinkedDurationVector::operator[](size_t pos) {
      SubArray* correct_sub = last;
      while (pos < correct_sub->starting_index) {
          correct_sub = correct_sub->previous;
      }
      if (correct_sub->array == nullptr) {
          while (parameter_handler.loaded_durations_size > parameter_handler.max_memory_durations) {
              auto * temp = (SubArray*) parameter_handler.subvector_queue.front();
              parameter_handler.subvector_queue.pop_front();
              delete[] temp->array;
              temp->array = nullptr;
              parameter_handler.loaded_durations_size -= temp->size * sizeof(uint64_t);
          }
          load_data(correct_sub);
          loaded_subarrays.insert(correct_sub);
      }
      return (*correct_sub)[pos];
}

size_t LinkedVector::getFirstOccurrenceBefore(pallas_timestamp_t ts) {
    if (ts <= front()) {
        return 0;
    }
    if (back() < ts) {
        return size - 1;
    }
    // TODO Infinite loop on ft.C.64 with 30 slices
    auto current_subarray = first;
    // First, we find the correct subarray
    while (current_subarray->last_value < ts) {
        current_subarray = current_subarray->next;
        if (current_subarray == nullptr) {
            pallas_warn("This shouldn't have happened\n");
            return -1;
        }
    }
    // We need first_value <= ts <= last_value
    if (ts < current_subarray->first_value) {
        if (current_subarray->starting_index > 0) {
            return current_subarray->starting_index - 1;
        }
        return 0;
    }
    if (current_subarray->array == nullptr) {
        load_data(current_subarray);
    }
    // Then we do a dichotomy.
    size_t start = 0;
    size_t end = current_subarray->size - 1;

    while (start < end) {
        size_t middle = (start + end ) / 2;
        if (current_subarray->array[middle] <= ts && current_subarray->array[middle + 1] > ts) {
            return current_subarray->starting_index + middle;
        }
        if (current_subarray->array[middle] < ts) {
            if (start == middle) {
                return end;
            }
            start = middle;
        } else {
            end = middle;
        }
    }
    pallas_error("This shouldn't have happened: Out of the Loop\n");
}

pallas_duration_t LinkedDurationVector::computeDurationBetween(size_t start_index, size_t end_index) {
    // Find the correct starting sub-array
    auto* start_subarray = first;
    while (start_subarray->starting_index + start_subarray->size < start_index) {
        start_subarray = start_subarray->next;
        if (start_subarray == nullptr)
            return 0;
    }

    pallas_duration_t sum = 0;
    if ( start_subarray->starting_index != start_index ) {
        // Load the sub_array
        size_t i = start_index;
        sum += at( i++ );
        for (; i < start_subarray->starting_index + start_subarray->size && i < end_index; i++) {
            sum += start_subarray->at(i);
        }
        start_subarray = start_subarray->next;
        if (start_subarray == nullptr)
            return sum;
    }

    while (start_subarray->starting_index + start_subarray->size < end_index) {
        sum += start_subarray->mean * start_subarray->size;
        start_subarray = start_subarray->next;
        if (start_subarray == nullptr)
            return sum;
    }

    size_t i = start_subarray->starting_index;
    sum += at(i ++);
    for (; i < end_index; i++) {
        sum += start_subarray->at(i);
    }
    return sum;
}

uint64_t& LinkedVector::front() {
    return first->first_value;
}

uint64_t& LinkedVector::back() {
    return last->last_value;
}


uint64_t& LinkedDurationVector::front() {
    return at(0);
}

uint64_t& LinkedDurationVector::back() {
    return at(size - 1);
}

void LinkedVector::free_data() {
    if (first == nullptr)
        return;
    auto& dq = parameter_handler.subvector_queue;
    for (auto* sub : loaded_subarrays) {
        // We need to remove the subvector from the global memory queue
        auto it = std::find(dq.begin(), dq.end(), sub);
        if (it != dq.end()) {
            dq.erase(it);
        }
        delete[] sub->array;
        sub->array = nullptr;
        parameter_handler.loaded_durations_size -= sub->size;
    }
}
void LinkedDurationVector::free_data() {
    if (first == nullptr)
        return;
    auto& dq = parameter_handler.subvector_queue;
    for (auto* sub : loaded_subarrays) {
        // We need to remove the subvector from the global memory queue
        auto it = std::find(dq.begin(), dq.end(), sub);
        if (it != dq.end()) {
            dq.erase(it);
        }
        delete[] sub->array;
        sub->array = nullptr;
        parameter_handler.loaded_durations_size -= sub->size;
    }
}

LinkedVector::~LinkedVector() {
    free_data();
    if (is_contiguous) {
        // All the subvectors were allocated using a single big calloc
#ifdef DEBUG
        auto* temp = first;
        auto& dq = parameter_handler.subvector_queue;
        for (int i = 0; i < n_sub_array; i ++, temp++) {
            // Check we've correctly cleared it
            // And cleared it from the queue
            pallas_assert(temp->array == nullptr);
            auto it = std::find(dq.begin(), dq.end(), temp);
            pallas_assert(it == dq.end());
        }
#endif
        free(first);
    } else {
        auto * sub = first;
        while (sub->next) {
            sub = sub->next;
            delete sub->previous;
        }
        delete sub;
    }
}

LinkedDurationVector::~LinkedDurationVector() {
    free_data();
    if (is_contiguous) {
        // All the subvectors were allocated using a single big calloc
#ifdef DEBUG
        auto* temp = first;
        auto& dq = parameter_handler.subvector_queue;
        for (int i = 0; i < n_sub_array; i ++, temp++) {
            // Check we've correctly cleared it
            // And cleared it from the queue
            pallas_assert_equals(temp->array, nullptr);
            auto it = std::find(dq.begin(), dq.end(), temp);
            pallas_assert(it == dq.end());
        }
#endif
        free(first);
    } else {
        auto * sub = first;
        while (sub->next) {
            sub = sub->next;
            delete sub->previous;
        }
        delete sub;
    }
}

SAME_FOR_BOTH_VECTORS(void, reset_offsets() {
    auto* v = first;
    while (v != nullptr) {
        v->offset = 0;
        v = v->next;
    }
})

SAME_FOR_BOTH_VECTORS(uint64_t*, as_flat_array() {
    load_all_data();
    auto * output = new uint64_t[size];
    auto * start = first;
    size_t i = 0;
    while (start != nullptr) {
        std::memcpy(&output[i], start->array, start->size * sizeof(uint64_t));
        i += start->size;
        start = start->next;
    }
    return output;
})


std::vector<double> LinkedVector::getWeights(pallas_timestamp_t start, pallas_timestamp_t end) {
    auto output = std::vector<double>();
    auto *current = first;
    double sum = 0;
    // While loop to go through all the SubVectors.
    // Legend:
    //   - : Time spent in current vector but NOT in the window
    //   # : Time spent in current vector AND in the window
    // We store in output the ratio of # / ( - + # )
    // i.e. the ratio of time spent in window over duration of current vector
    while (current != nullptr) {
        if (current->last_value < start) {
            // first_value ... last_value ... [ start ... end ]
            // --------------------------
            // Completely outside of the range
            output.push_back(0.);
        } else if (end < current->first_value) {
            // [ start ... end ] .. first_value ... last_value
            //                      --------------------------
            // We're past the boundaries, we can stop searching.
            break;
        } else if (start <= current->first_value && current->last_value <= end) {
            // [ start ... first_value ... last_value ... end ]
            //             ##########################
            // Completely inside the bounds
            output.push_back(1.0);
        } else if (current->first_value < start && end < current->last_value) {
            // first_value ... [ start ... end ] ... last_value
            // ----------------#################---------------
            // We have to compute the ratio of the two intervals to "guess" the weight of this vector in the total
            output.push_back(static_cast<double>(end - start) / (current->last_value - current->first_value));
        } else if (current->first_value < start && current->last_value < end) {
            // first_value ... [ start ... last_value ... end ]
            // ----------------######################
            // Same thing except the window ends in the current vector
            output.push_back(static_cast<double>(current->last_value - start) / (current->last_value - current->first_value));
        } else if (current->first_value <= end && end < current->last_value) {
            // [ start ... first_value ... end ] ... last_value
            //             #####################---------------
            // Same thing except the window starts in the current vector and isn't entirely contained in it.
            output.push_back(static_cast<double>(end - current->first_value) / (current->last_value - current->first_value));
        } else {
            pallas_error("This is not supposed to happen !\n");
            pallas_error("start=%lu, end=%lu\n", start, end);
        }
        sum += output.back();
        current = current->next;
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

pallas_duration_t LinkedDurationVector::weightedSum(std::vector<double>& weights) {
    double sum = 0;
    auto* current = first;
    for (auto w: weights) {
        sum += w * current->mean * current->size;
        current = current->next;
    }
    return sum;
}

}  // namespace pallas
