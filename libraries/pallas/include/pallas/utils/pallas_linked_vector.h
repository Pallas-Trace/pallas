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

#endif
#ifdef __cplusplus
#include <cstdint>
#include <set>
#include <vector>

#include "pallas_parameter_handler.h"

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
    DurationLossy = 4,
};

class SubArrayCodec {
    protected:
        /** ZigZag Mapping of Negative Values to Non-Negative Values */
        static uint64_t zigzag_encode(int64_t x);
        static int64_t zigzag_decode(uint64_t x);
    public: 
        virtual ~SubArrayCodec() = default;
        virtual SubArrayEncoding encoding() const = 0;
        virtual bool can_encode(uint64_t* array, size_t size) const = 0;
        /** caller_kind is 0 for LinkedVector::SubArray and 1 for LinkedDurationVector::SubArray. */
        virtual size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const = 0;
        /** caller_kind is 0 for LinkedVector::SubArray and 1 for LinkedDurationVector::SubArray. */
        virtual void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const = 0;
};
class NoneCodec : public SubArrayCodec {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::None;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
};

class Delta2VintCodecBase : public SubArrayCodec {
    protected:
        /** Varint Helpers */
        static void write_varint(uint64_t x, uint8_t*& out);
        static uint64_t read_varint(const uint8_t*& p, const uint8_t* end);
        size_t encode_timestamp(uint64_t* src, size_t size, uint64_t*& encoded_array) const;
        size_t encode_duration(uint64_t* src, size_t size, uint64_t*& encoded_array) const;
        static void decode_timestamp(const uint64_t* encoded_words, size_t enc_size, uint64_t* decoded_array, size_t size);
        static void decode_duration(const uint64_t* encoded_words, size_t enc_size, uint64_t* out, size_t size);
};

class TimestampDelta2VintCodec : public Delta2VintCodecBase {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::Delta2VintTimestamp;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
};

class DurationDelta2VintCodec : public Delta2VintCodecBase {
    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::Delta2VintDuration;
        }
        bool can_encode(uint64_t* array, size_t size) const override {
            return true;
        }
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
};

/**
 * A test encoding mechanism for SubArrays, which simply generates values (withing a given error rate ?) instead of actually storing them.
 */

enum class MonotoneLossyVariant : uint8_t {
    QLinear = 0,
    QLinearMeanRep = 1,
    QLinearPchipMeanRep = 2,
    QLinearPchipMeanRepAdaptive = 4, 
};

enum class DurationLossyVariant : uint8_t {
    QLinear = 0,
    NormalSample = 1,
};

class MonotoneLossyCodec : public SubArrayCodec {
    protected:
        static constexpr size_t kKPercentileAnchorCount = 11;
        static constexpr size_t kKPercentileSegmentCount = kKPercentileAnchorCount - 1;
        static constexpr size_t kQLinearWordCount = kKPercentileAnchorCount;

        static size_t kpercentile_anchor_index(size_t size, size_t anchor_id);
        static uint64_t linear_interpolate(uint64_t start_value, uint64_t end_value, size_t offset, size_t span);
        static size_t encode_kpercentile_linear(uint64_t* array, size_t size, uint64_t*& encoded_array);
        static void decode_kpercentile_linear(const uint64_t* encoded_array, size_t enc_size, uint64_t* decoded_array, size_t size);

    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::MonotoneLossy;
        }
        bool can_encode(uint64_t* array, size_t size) const override;
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
};

class DurationLossyCodec : public SubArrayCodec {
    protected:
        static constexpr size_t kNormalSampleWordCount = 1;
        static constexpr size_t kQLinearAnchorCount = 11;
        static constexpr size_t kQLinearSegmentCount = kQLinearAnchorCount - 1;
        static constexpr size_t kQLinearStoredWordCount = kQLinearAnchorCount - 2;
        static constexpr uint64_t kShuffleSeed = 0xD071A110ULL;

        static uint64_t linear_interpolate(uint64_t start_value, uint64_t end_value, size_t offset, size_t span);
        static uint64_t compute_shuffle_seed(size_t size, size_t starting_index);
        static uint64_t pack_double(double value);
        static double unpack_double(uint64_t value);
        static size_t qlinear_anchor_index(size_t size, size_t anchor_id);
        static size_t encode_qlinear(uint64_t* array, size_t size, uint64_t*& encoded_array);
        static void decode_qlinear(const uint64_t* encoded_array, size_t enc_size, uint64_t* decoded_array, size_t size, void* caller_sub_array, int caller_kind);
        static size_t encode_normal_sample(uint64_t* array, size_t size, uint64_t*& encoded_array);
        static void decode_normal_sample(const uint64_t* encoded_array, size_t enc_size, uint64_t* decoded_array, size_t size, void* caller_sub_array, int caller_kind);

    public:
        SubArrayEncoding encoding() const override {
            return SubArrayEncoding::DurationLossy;
        }
        bool can_encode(uint64_t* array, size_t size) const override;
        size_t encode(FILE* file, uint64_t* array, size_t size, uint64_t*& encoded_array, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
        void decode(uint64_t* encoded_array, size_t enc_size, uint64_t*& decoded_array, size_t size, void* caller_sub_array, int caller_kind, const ParameterHandler* parameter_handler) const override;
};

const SubArrayCodec* get_subarray_codec(SubArrayEncoding encoding);

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

    /** Sets the preferred encoding for future subarrays of this vector. */
    void setPreferredSubArrayEncoding(SubArrayEncoding encoding);
    /** Returns the preferred encoding for future subarrays of this vector. */
    [[nodiscard]] SubArrayEncoding getPreferredSubArrayEncoding() const;
    /** Returns the stored encoding of each subarray in linked-list order. */
    [[nodiscard]] std::vector<SubArrayEncoding> getSubArrayEncodings() const;
    /** Returns the stored encoding of currently loaded subarrays in linked-list order. */
    [[nodiscard]] std::vector<SubArrayEncoding> getLoadedSubArrayEncodings() const;

   private:
    /** Path to the file storing this vector. */
    const char* filePath = nullptr;

    /** Parameter handler for the whole trace. */
    ParameterHandler& parameter_handler;
    /** Preferred encoding for newly created subarrays. */
    SubArrayEncoding preferred_sub_arr_encoding = static_cast<SubArrayEncoding>(DEFAULT_SUBARRAY_ENCODING);
    /**
     * A fixed-sized array functioning as a node in a linked array list.
     */
    class SubArray {
        friend class NoneCodec;
        friend class TimestampDelta2VintCodec;
        friend class DurationDelta2VintCodec;
        friend class MonotoneLossyCodec;
        friend class DurationLossyCodec;

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
    /** Returns the starting index of a LinkedVector subarray passed through the codec callback API. */
    static size_t codec_subarray_starting_index(const void* caller_sub_array);
    /**
     * Creates a new LinkedVector.
     */
    LinkedVector(ParameterHandler& p);
    LinkedVector(ParameterHandler& p, SubArrayEncoding preferred_encoding);

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

    /** Sets the preferred encoding for future subarrays of this vector. */
    void setPreferredSubArrayEncoding(SubArrayEncoding encoding);
    /** Returns the preferred encoding for future subarrays of this vector. */
    [[nodiscard]] SubArrayEncoding getPreferredSubArrayEncoding() const;
    /** Returns the stored encoding of each subarray in linked-list order. */
    [[nodiscard]] std::vector<SubArrayEncoding> getSubArrayEncodings() const;
    /** Returns the stored encoding of currently loaded subarrays in linked-list order. */
    [[nodiscard]] std::vector<SubArrayEncoding> getLoadedSubArrayEncodings() const;

   private:
    /** Path to the file storing this vector. */
    const char* filePath = nullptr;
    /** Parameter handler for the whole trace. */
    ParameterHandler& parameter_handler;
    /** Preferred encoding for newly created subarrays. */
    SubArrayEncoding preferred_sub_arr_encoding = static_cast<SubArrayEncoding>(DEFAULT_SUBARRAY_ENCODING);
    /**
     * A fixed-sized array functioning as a node in a linked array list.
     */
    class SubArray {
        friend class NoneCodec;
        friend class TimestampDelta2VintCodec;
        friend class DurationDelta2VintCodec;
        friend class MonotoneLossyCodec;
        friend class DurationLossyCodec;

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
    /** Returns the starting index of a LinkedDurationVector subarray passed through the codec callback API. */
    static size_t codec_subarray_starting_index(const void* caller_sub_array);
    /** Returns the minimum value of a LinkedDurationVector subarray passed through the codec callback API. */
    static uint64_t codec_subarray_min(const void* caller_sub_array);
    /** Returns the maximum value of a LinkedDurationVector subarray passed through the codec callback API. */
    static uint64_t codec_subarray_max(const void* caller_sub_array);
    /** Returns the mean value of a LinkedDurationVector subarray passed through the codec callback API. */
    static uint64_t codec_subarray_mean(const void* caller_sub_array);

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
    LinkedDurationVector(ParameterHandler& p, SubArrayEncoding preferred_encoding);
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
