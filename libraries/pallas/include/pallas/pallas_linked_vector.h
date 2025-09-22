/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * A custom type of Linked-List. This takes into account the fact that we never go remove anything from timestamps
 * vector.
 */
#pragma once

#include "pallas_dbg.h"
#include "pallas_timestamp.h"
#ifndef __cplusplus
#include <stdint.h>
#endif
#ifdef __cplusplus
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "pallas_parameter_handler.h"
/** Default size for creating Vectors and SubVectors.*/
#define DEFAULT_VECTOR_SIZE 1000

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
    LinkedVector(FILE* vectorFile, const char* valueFilePath, ParameterHandler& parameter_handler);

    /**
     * Classic destructor. Calls free_data().
     */
    ~LinkedVector();
    /* Returns an array of size #size containing a copy of the values in this vector.*/
    [[nodiscard]] uint64_t* as_flat_array();
};

class LinkedDurationVector {
   public:
    /** Number of element stored in the vector.  */
    size_t size = 0;
    /** Number of times the vector's data is linked somewhere. */
    size_t ref = 0;
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
         * Updates the min/max/mean, taking into account all the items from 0 to size-1.
         *
         * This is because we assume the last element isn't a duration, but a timestamp.
         */
        void update_statistics();

       public:
        /** Replace the sum (being stored in the mean) by the actual mean. */
        void final_update_mean() { mean /= size; }
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
    void final_update_mean() { mean /= size; }
 /** Returns the sum of the durations between the two indexes. */
    pallas_duration_t computeDurationBetween(size_t start_index, size_t end_index);

    ~LinkedDurationVector();
    /* Returns an array of size #size containing a copy of the values in this vector.*/
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
    LinkedDurationVector(FILE* vectorFile, const char* valueFilePath, ParameterHandler& parameter_handler);

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
