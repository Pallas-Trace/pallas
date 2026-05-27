/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * A custom type of Linked-List. This takes into account the fact that we never go remove anything from timestamps
 * vector.
 */
#pragma once

#include "pallas_timestamp.h"
#ifndef __cplusplus
#include <stdint.h>
#include <stdexcept>
#endif
#ifdef __cplusplus
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <set>
#include <vector>

#include "pallas_parameter_handler.h"
#include "pallas_dbg.h"
#include "pallas_log.h"
/** Default size for creating Vectors and SubVectors.*/
#define DEFAULT_VECTOR_SIZE 1000
#define DEFAULT_SUBARRAY_ENCODING 0
namespace pallas {

/**
 * Indicates the type of SubArray Encoding used for a LinkedVector::SubArray
 */
enum class SubArrayEncoding : uint8_t {
    None = 0,
    Delta2VintTimestamp = 1,
    Delta2VintDuration = 2,
    MonotoneLossy = 3,
};

class SubArrayCodec {
    protected:
        /** ZigZag Mapping of Negative Values to Non-Negative Values */
        static inline uint64_t zigzag_encode(int64_t x) {
            return (static_cast<uint64_t>(x) << 1) ^ static_cast<uint64_t>(x >> 63);
        }

        static inline int64_t zigzag_decode(uint64_t x) {
            return static_cast<int64_t>((x >> 1) ^ static_cast<uint64_t>(-static_cast<int64_t>(x & 1)));
        }
    public: 
        virtual ~SubArrayCodec() = default;
        virtual SubArrayEncoding encoding() const = 0;
        virtual bool can_encode(uint64_t* array, size_t size) const = 0;
        virtual size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const = 0;
        virtual void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size) const = 0;
};
class NoneCodec : public SubArrayCodec {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::None;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const override {
            encoded_array = array;  // No encoding, so the encoded array is the same as the original array.
            return size;
        }
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size) const override {
            decoded_array = encoded_array;
        }
};

class Delta2VintCodecBase : public SubArrayCodec {
    protected:
        /** Varint Helpers */
        static inline void write_varint(uint64_t x, uint8_t*& out) {
            while (x >= 0x80) {
                *out++ = static_cast<uint8_t>((x & 0x7f) | 0x80);
                x >>= 7;
            }
            *out++ = static_cast<uint8_t>(x);
        }

        static inline uint64_t read_varint(const uint8_t*& p, const uint8_t* end) {
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
    
        size_t encode_timestamp(uint64_t* src, size_t size, uint64_t*& encoded_array) const {
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

        size_t encode_duration(uint64_t* src, size_t size, uint64_t*& encoded_array) const {
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
    
        static void decode_timestamp(const uint64_t* encoded_words, size_t enc_size, uint64_t* decoded_array, size_t size) {
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

        static void decode_duration(const uint64_t* encoded_words, size_t enc_size, uint64_t* out, size_t size) {
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
};

class TimestampDelta2VintCodec : public Delta2VintCodecBase {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::Delta2VintTimestamp;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const override {
            return encode_timestamp(array, size, encoded_array);
        }
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size) const override {
            decoded_array = new uint64_t[size];
            decode_timestamp(encoded_array, enc_size, decoded_array, size);
        }
};

class DurationDelta2VintCodec : public Delta2VintCodecBase {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::Delta2VintDuration;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const override {
            return encode_duration(array, size, encoded_array);
        }
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size) const override {
            decoded_array = new uint64_t[size];
            decode_duration(encoded_array, enc_size, decoded_array, size);
        }
};

/**
 * A test encoding mechanism for SubArrays, which simply generates values (withing a given error rate ?) instead of actually storing them.
 */

enum class MonotoneLossyVariant : uint8_t {
    DecileLinear = 0,
    DecileLinearMeanRep = 1,
    DecilePchipMeanRep = 2,
    DecilePchipMeanRepAdaptive = 4, 
};

class MonotoneLossyCodec : public SubArrayCodec {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::MonotoneLossy;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, const ParameterHandler* parameter_handler) const override {
            pallas_error("Not yet implemented\n");
            return 0;
        }
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size) const override {
            pallas_error("Not yet implemented\n");
        }
};

inline const SubArrayCodec* get_subarray_codec(SubArrayEncoding encoding) {
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
/**
 * Classic linked array list. Sub-arrays are implemented as a subclass
 */
class LinkedVector {
   public:
    /** Number of element stored in the vector.  */
    size_t size = 0;
    /** Number of times the vector's data is linked somewhere. */
    size_t ref = 0;
    /** Number of Sub-arrays. */
    size_t n_sub_array = 1;
    /** Describes if the SubArrays were all defined contiguously or not. */
    bool is_contiguous = false;
    /**
     * Adds a new element at the end of the vector, after its current last element.
     *
     * @param val Value to be added.
     * @return Reference to the new element.
     */
    uint64_t* add(uint64_t val);

    /**
     * Returns a reference to the element at specified location `pos`, with bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& at(size_t pos);

    /**
     * Returns a reference to the element at specified location `pos`, without bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& operator[](size_t pos);

    /**
     * Returns a reference to the first element in the vector.
     * @return Reference to the first element.
     */
    [[nodiscard]] uint64_t& front();

    /**
     * Returns a reference to the last element in the vector.
     * @return Reference to the last element.
     */
    [[nodiscard]] uint64_t& back();

    /**
     * Frees the data contained in the vector, but keeps the references needed to load them again.
     */
    void free_data();

    /**
     * Returns a representation of the vector as a string, for example: "[10, 10000, 3141]"
     */
    std::string to_string();

    /**
     * Writes the vector to the given files.
     * @param infoFile File where information about the vector is stored.
     * @param dataFile  File where most of the data are stored.
     * @param parameter_handler Handler for the storage parameters.
     */
    void write_to_file(FILE* infoFile, FILE* dataFile, const ParameterHandler* parameter_handler);

    /**
     * Resets the offsets of all the subvectors.
     */
    void reset_offsets();

    /**
     * Given a starting and an ending timestamp, returns an array containing the ratio, for each subvector,
     * of the time spent between those two timestamps over the total duration of the subvector.
     */
    std::vector<double> getWeights(pallas_timestamp_t start, pallas_timestamp_t end);

   private:
    /** Path to the file storing this vector. */
    const char* filePath = nullptr;

    /** Parameter handler for the whole trace. */
    ParameterHandler& parameter_handler;
    /**
     * A fixed-sized array functioning as a node in a linked array list.
     */
    class SubArray {
       public:
        /** Number of elements stored in the vector. */
        size_t size = 0;

        /** Number of elements this vector has allocated. */
        size_t allocated = DEFAULT_VECTOR_SIZE;

        /** Subarray encoding used during the storage time */
        SubArrayEncoding sub_arr_encoding = static_cast<SubArrayEncoding>(DEFAULT_SUBARRAY_ENCODING);

        /** Encoded size : Will be calculated during the sub_array->write_to_file */

        size_t enc_size = 0;

        /** Array of elements. Currently only used on uint64_t */
        uint64_t* array = nullptr;

        /** Next SubArray in the Vector. nullptr if last. */
        SubArray* next = nullptr;

        /** Previous SubArray in the Vector. nullptr if first. */
        SubArray* previous = nullptr;

        /** Starting index of this SubVector. */
        size_t starting_index = 0;
        /** Value of the first element of that sub-array.*/
        uint64_t first_value = 0;
        /** Value of the last element of that sub-array.*/
        uint64_t last_value = 0;
        /** Offset where data is written. */
        size_t offset = 0;
        /**
         * Adds a new element at the end of the vector, after its current last element.
         *
         * @param val Value to be added.
         * @return Reference to the new element.
         */
        uint64_t* add(uint64_t val);

        /**
         * Returns a reference to the element at specified location `pos`, with bounds checking.
         * @param pos Position of the element in the array.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& at(size_t pos) const;

        /**
         * Returns a reference to the element at specified location `pos`, without bounds checking.
         * @param pos Position of the element in the LinkedVector.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& operator[](size_t pos) const;

        /**
         * Copies the values in array to given_array.
         * @param given_array An allocated array of correct size.
         */
        void copy_to_array(uint64_t* given_array) const;

        /**
         * Writes the content of this array to the file at the current offset.
         * Specifically, the first sizeof(size_t) bytes written will be the size of the data, then the data.
         * Then, sets up the "offset" field accordingly.
         * @param file File where the data is stored.
         * @param parameter_handler Handler for the storage parameters.
         */
        void write_to_file(FILE* file, const ParameterHandler* parameter_handler);

        ~SubArray();

        /**
         * Construct a SubArray of a given size.
         * @param size Size of the SubVector.
         * @param previous Previous SubArray.
         */
        explicit SubArray(size_t size, SubArray* previous = nullptr);
        /**
         * Load a SubArray's metadata from a file. Doesn't load the data.
         * @param file File where the metadata is stored.
         * @param previous Previous SubArray.
         */
        SubArray(FILE* file, SubArray* previous = nullptr);
    };

    std::set<SubArray*> loaded_subarrays;

    /** First array list in the linked array list structure.*/
    SubArray* first;
    /** Last array list in the linked array list structure.*/
    SubArray* last;

    /**
     * Loads the timestamps from filePath.
     */
    void load_data(SubArray* sub);

   public:
    /** Loads all the subvectors. */
    void load_all_data();
    /** Returns the index of the first value <= ts. If all values > ts, returns 0. */
    size_t getFirstOccurrenceBefore(pallas_timestamp_t ts);
    /**
     * Creates a new LinkedVector.
     */
    LinkedVector(ParameterHandler& p);

    /** Creates a new LinkedVector from a file. Doesn't actually load it until and element is accessed. */
    LinkedVector(FILE* vectorFile, const char* valueFilePath, ParameterHandler& parameter_handler, uint8_t abi_version);

    /**
     * Classic destructor. Calls free_data().
     */
    ~LinkedVector();
    /** Returns an array of size #size containing a copy of the values in this vector.*/
    [[nodiscard]] uint64_t* as_flat_array();
};

class LinkedDurationVector {
   public:
    /** Number of element stored in the vector.  */
    size_t size = 0;
    /** Number of times the vector's data is linked somewhere. */
    size_t ref = 0;
    /** Number of Sub-arrays. */
    size_t n_sub_array = 1;
    /** Describes if the SubArrays were all defined contiguously or not. */
    bool is_contiguous = false;
    /**
     * Adds a new element at the end of the vector, after its current last element.
     * Updates mean, min and max.
     *
     * @param val Value to be added.
     * @return Pointer to the new element.
     */
    uint64_t* add(uint64_t val);

    /**
     * Returns a reference to the element at specified location `pos`, with bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& at(size_t pos);

    /**
     * Returns a reference to the element at specified location `pos`, without bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& operator[](size_t pos);

    /**
     * Returns a reference to the first element in the vector.
     * @return Reference to the first element.
     */
    [[nodiscard]] uint64_t& front();

    /**
     * Returns a reference to the last element in the vector.
     * @return Reference to the last element.
     */
    [[nodiscard]] uint64_t& back();

    /**
     * Frees the data contained in the vector, but keeps the references needed to load them again.
     */
    void free_data();

    /**
     * Returns a representation of the vector as a string, for example: "[10, 10000, 3141] { min, mean, max }"
     */
    std::string to_string();

    /**
     * Writes the vector to the given files.
     * If size >= 4, we do the following:
     *    - To vectorFile, we write [size, min, max, mean, offset] in that order.
     *    - To valueFile, we write the array.
     * If size <= 3, we don't write anything to valueFile.
     * Instead, we write [size] + array to vectorFile.
     * @param infoFile File where metadata is stored.
     * @param dataFile File where data is stored (most of the time).
     * @param parameter_handler Handler for the storage parameters.
     */
    void write_to_file(FILE* infoFile, FILE* dataFile, const ParameterHandler* parameter_handler);

    /**
     * Returns the weighted mean over the subvectors.
     */
    pallas_duration_t weightedSum(std::vector<double>& weights);

    /**
     * Resets the offsets of all the subvectors.
     */
    void reset_offsets();

   private:
    /** Path to the file storing this vector. */
    const char* filePath = nullptr;
    /** Parameter handler for the whole trace. */
    ParameterHandler& parameter_handler;
    /**
     * A fixed-sized array functioning as a node in a linked array list.
     */
    class SubArray {
       public:
        /** Number of elements stored in the vector. */
        size_t size = 0;

        /** Number of elements this vector has allocated. */
        size_t allocated = DEFAULT_VECTOR_SIZE;

        /** Subarray Encoding Mechanism used */

        SubArrayEncoding sub_arr_encoding = static_cast<SubArrayEncoding>(DEFAULT_SUBARRAY_ENCODING);

        /** Encoded size : Will be calculated during the sub_array->write_to_file */

        size_t enc_size = 0;

        /** Array of elements. Currently only used on uint64_t */
        uint64_t* array = nullptr;

        /** Next SubArray in the Vector. nullptr if last. */
        SubArray* next = nullptr;

        /** Previous SubArray in the Vector. nullptr if first. */
        SubArray* previous = nullptr;

        /** Starting index of this SubVector. */
        size_t starting_index = 0;

        /** Offset where data is written. */
        size_t offset = 0;

        /**
         * Updates the min/max/mean.
         */
        void update_statistics();

       public:
        /** Replace the sum (being stored in the mean) by the actual mean. */
        void final_update_mean();
        /** Max element stored in the array. */
        uint64_t min = UINT64_MAX;

        /** Min element stored in the array. */
        uint64_t max = 0;

        /** Mean of all the elements in the array. */
        uint64_t mean = 0;

        /**
         * Adds a new element at the end of the vector, after its current last element.
         * Updates mean, min and max.
         *
         * @param val Value to be added.
         * @return Pointer to the new element.
         */
        uint64_t* add(uint64_t val);

        /**
         * Returns a reference to the element at specified location `pos`, with bounds checking.
         * @param pos Position of the element in the array.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& at(size_t pos) const;

        /**
         * Returns a reference to the element at specified location `pos`, without bounds checking.
         * @param pos Position of the element in the LinkedVector.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& operator[](size_t pos) const;

        /**
         * Copies the values in array to given_array.
         * @param given_array An allocated array of correct size.
         */
        void copy_to_array(uint64_t* given_array) const;

        /**
         * Writes the content of this array to the file at the current offset.
         * Specifically, the first sizeof(size_t) bytes written will be the size of the data, then the data.
         * Then, sets up the "offset" field accordingly.
         *  @param file File where the data is stored.
         * @param parameter_handler Handler for the storage parameters.
         */
        void write_to_file(FILE* file, const ParameterHandler* parameter_handler);

        ~SubArray();

        /**
         * Construct a SubArray of a given size.
         * @param size Size of the SubVector.
         * @param previous Previous SubArray.
         */
        explicit SubArray(size_t size, SubArray* previous = nullptr);

        /**
         * Load a SubArray's metadata from a file. Doesn't load the data.
         * @param file File where the metadata is stored.
         * @param previous Previous SubArray.
         */
        SubArray(FILE* file, SubArray* previous = nullptr);
    };
    /** Set of loaded subarrays indexes. */
    std::set<SubArray*> loaded_subarrays;

    /** First array list in the linked array list structure.*/
    SubArray* first;
    /** Last array list in the linked array list structure.*/
    SubArray* last;

    /**
     * Loads the durations from filePath.
     */
    void load_data(SubArray* sub);
    /**
     * Updates the min/max/mean.
     */
    void update_statistics();

   public:
    /**
     * Loads all the subvectors.
     */
    void load_all_data();
    /** Replace the sum (being stored in the mean) by the actual mean. */
    void final_update_mean();
    /** Returns the sum of the durations between [start, end[. */
    pallas_duration_t computeDurationBetween(size_t start_index, size_t end_index);

    ~LinkedDurationVector();
    /** Returns an array of size #size containing a copy of the values in this vector.*/
    [[nodiscard]] uint64_t* as_flat_array();

    /** Max element stored in the vector. */
    uint64_t min = UINT64_MAX;
    /** Min element stored in the vector. */
    uint64_t max = 0;
    /** Mean of all the elements in the vector. */
    uint64_t mean = 0;
    /**
     * Loads a LinkedDurationVector from a file.
     * Only loads the statistics, doesn't load the timestamps until they're accessed.
     */
    LinkedDurationVector(FILE* vectorFile, const char* valueFilePath, ParameterHandler& parameter_handler, uint8_t abi_version);

    /**
     * Creates a new LinkedDurationVector.
     */
    LinkedDurationVector(ParameterHandler& p);
};
}  // namespace pallas

#else
typedef struct LinkedVector {
} LinkedVector;

typedef struct LinkedDurationVector {
} LinkedDurationVector;
#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
