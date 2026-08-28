/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Standalone subarray primitives used as a staging area for refactoring the
 * nested linked-vector subarray implementation.
 */
#pragma once

#include "pallas/utils/pallas_timestamp.h"
#include "pallas/utils/pallas_parameter_handler.h"

#ifndef __cplusplus
#include <stdint.h>
#else

#include <cstddef>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>



/** Domain, storage-policy, and SubArray state enums used by linked-vector storage. */
namespace pallas {

    /** Identifies the logical kind of values stored in a SubArray or manager. */
    enum class ValueDomain : uint8_t {
        /** Event or sequence timestamps. */
        Timestamp = 0,
        /** Inclusive or exclusive durations. */
        Duration = 1,
    };

    /** Selects the exact or lossy encoding family used by a SubArray manager. */
    enum class StoragePolicy : uint8_t {
        /** Store values without an internal encoding transform. */
        None = 0,
        /** Store values using delta-based exact encoding. */
        Delta = 1,
        /** Store values using one of the lossy prediction-based schemes. */
        Lossy = 2,
    };

    /** Describes whether a SubArray is being filled online or replayed during analysis. */
    enum class SubArrayPhase : uint8_t {
        /** Runtime path used while recording values into the archive. */
        RuntimeWrite = 0,
        /** Read-side path used when materialising values from stored data. */
        AnalysisRead = 1,
    };

    /** Result returned by manager add paths while filling a SubArray online. */
    enum class AddStatus : uint8_t {
        /** Value was accepted and encoded normally. */
        Ok = 0,
        /** Value could not be represented in the current model and must be handled separately. */
        Outlier = 1,
        /** Current SubArray is full and the caller must rotate to a new one. */
        Full = 2,
    };

}


/** SubArray storage shells shared by timestamp and duration linked-vector paths. */
namespace pallas {
    class File;

    /** A SubArrayStats object contains statistics on a SubArray
     */
    class SubArrayStats {
        protected:
            ValueDomain _value_domain;
            size_t _count = 0;

        public:
            SubArrayStats(ValueDomain value_domain);
            virtual ~SubArrayStats() = default;
            /** Update the subArray statistics */
            virtual void add(uint64_t value) = 0;

            /** Finalize the subArray statistics (eg. compute the mean duration) */
            virtual void finalize() = 0;

            [[nodiscard]] ValueDomain value_domain() { return _value_domain; }
            [[nodiscard]] size_t count() { return _count; }

            [[nodiscard]] virtual uint64_t first_timestamp() { return -1; }
            [[nodiscard]] virtual uint64_t last_timestamp () { return -1; }

            [[nodiscard]] virtual uint64_t min_duration() { return -1; }
            [[nodiscard]] virtual uint64_t max_duration() { return -1; }
            [[nodiscard]] virtual uint64_t mean_duration() { return -1; }

            virtual void write_summary(File* info_file) = 0;
            virtual void read_summary(File* info_file) = 0;
    };

    class SubArrayTimestampStats: public SubArrayStats {
        /** First logical timestamp covered by this SubArray. */
        uint64_t _first_timestamp = 0;
        /** Last logical timestamp covered by this SubArray. */
        uint64_t _last_timestamp = 0;
        public:
            SubArrayTimestampStats(ValueDomain value_domain);
            /** Update the subArray statistics */
            void add(uint64_t value) {
                if(_count == 0) { _first_timestamp = value; }
                _last_timestamp = value;
                _count++;
            }

            /** Finalize the subArray statistics (eg. compute the mean duration) */
            void finalize() {/* nothing to do */};

            [[nodiscard]] virtual uint64_t first_timestamp() { return _first_timestamp;} 
            [[nodiscard]] virtual uint64_t last_timestamp () { return _last_timestamp; }
            void write_summary(File* info_file);
            void read_summary(File* info_file);
    };

    class SubArrayDurationStats: public SubArrayStats {
        /** Minimum duration observed among the logical values stored here. */
        uint64_t _min_duration = UINT64_MAX;
        /** Maximum duration observed among the logical values stored here. */
        uint64_t _max_duration = 0;
        /** Mean duration cached for this SubArray. */
        uint64_t _mean_duration = 0;
        /** Tracks whether `mean_duration` currently stores a finalized mean or a running sum. */
        bool _mean_duration_is_finalized = false;

        public:
            SubArrayDurationStats(ValueDomain value_domain);
            /** Update the subArray statistics */
            void add(uint64_t value) {
                if(_count == 0) {
                _min_duration = value;
                _max_duration = value;
                } else {
                    if(value < _min_duration) {_min_duration = value;}
                    if(value > _max_duration) {_max_duration = value;}
                    // the actual mean duration will be computed later. For now, we just sum the durations
                    _mean_duration += value; 
                }
                _count++;
            }

            /** Finalize the subArray statistics (eg. compute the mean duration) */
            void finalize() {
                _mean_duration /= _count;
                _mean_duration_is_finalized = true;
            };
            
            [[nodiscard]] uint64_t min_duration() { return _min_duration; }
            [[nodiscard]] uint64_t max_duration() { return _max_duration; }
            [[nodiscard]] uint64_t mean_duration() {
                if(!_mean_duration_is_finalized) finalize();
                 return _mean_duration;
            }

            void write_summary(File* info_file);
            void read_summary(File* info_file);
    };

    // forward declaration
    class LinkedVectorBase;

    /**
     * Common physical storage unit used underneath `LinkedVectorBase`.
     *
     * A `SubArrayBase` represents one contiguous logical range inside a linked vector. It owns the navigation links, logical-range 
     * metadata, the backing buffer used while values are resident in memory, and the policy state needed to recreate its manager.
     * The actual storage algorithm is delegated to the attached `Manager`, so this class stays as the common shell shared by
     * `TimeSubArray` and `DurationSubArray`.
     */
    class SubArrayBase {
    protected:
        /** Next SubArray in the owning linked-vector chain. */
        SubArrayBase* _next_subarray = nullptr;
        /** Previous SubArray in the owning linked-vector chain. */
        SubArrayBase* _prev_subarray = nullptr;

        /** Owning linked vector, used for policy context and benchmark attribution. */
        LinkedVectorBase* _parent_linked_vector = nullptr;

    protected:
        /** Value domain served by this SubArray: timestamps or durations. */
        ValueDomain _value_domain;

        /** Exact or lossy storage policy selected for this SubArray. */
        StoragePolicy _storage_policy = StoragePolicy::None;
        /** Whether the SubArray is being used for runtime writes or analysis reads. */
        SubArrayPhase _subarray_phase = SubArrayPhase::RuntimeWrite; // TODO: is it usefull ?

        
        /** Number of logical values represented by this SubArray. */
        size_t _size = 0;
        /** Physical occupancy used inside the backing representation. */
        //size_t physical_size = 0; // TODO: useless ?
        /** Global logical index of the first value stored in this SubArray. */
        size_t _starting_index = 0;
        
        /** Offset of the subarray in the data file, or 0 if the subarray has not been written to disk yet */
        //size_t _file_offset = 0;
        off_t _details_offset = 0;
        /** Size of the subarray in the data file, or 0 if the subarray has not been written to disk yet */
        size_t _details_size = 0;

        SubArrayStats* _subarray_stats = nullptr;

    public:

    //TODO: make this constructor protected to force using create_subarray
        /** Runtime-write constructor used when appending to a live linked vector. */
        explicit SubArrayBase(ValueDomain domain,
                            StoragePolicy policy = StoragePolicy::None,
                            SubArrayBase* previous = nullptr,
                            const ParameterHandler* parameter_handler = nullptr,
                            LinkedVectorBase* parent = nullptr);

        /** File-backed constructor used while reconstructing archived SubArrays. */
        explicit SubArrayBase(File* info_file, ValueDomain domain, StoragePolicy policy = StoragePolicy::None, SubArrayBase* previous = nullptr,
                            const ParameterHandler* parameter_handler = nullptr, LinkedVectorBase* parent=nullptr);

        /** Virtual destructor for polymorphic timestamp/duration SubArray ownership. */
        virtual ~SubArrayBase();        

        /** Write/read the SubArray summary (eg. stats) */
        //void write_summary(File* summary_file) const;
        void read_summary(File* summary_file);

        /** Write/read the SubArray data (eg. timestamps) and calls the child class write_values */
        //void write_data(File* data_file);
        void read_data(File* data_file);
        void write_details(File* details_file, size_t* data_size = nullptr, off_t *data_offset = nullptr);


        static SubArrayBase* create_subarray(SubArrayBase* previous = nullptr,
                                        const ParameterHandler* parameter_handler = nullptr,
                                        LinkedVectorBase* parent = nullptr);
        static SubArrayBase* load_subarray(File* info_file,
                                        SubArrayBase* previous = nullptr,
                                        const ParameterHandler* parameter_handler = nullptr,
                                        LinkedVectorBase* parent = nullptr);
    protected:
        virtual void write_values(File* info_file) const = 0;
        virtual void load_values(File* info_file) = 0;
    public:
        /********* Getters/Setter functions *********/

        /** @returns Next SubArray in the linked chain, or `nullptr` at the tail. */
        [[nodiscard]] SubArrayBase* next_subarray() const;
        /** @returns Previous SubArray in the linked chain, or `nullptr` at the head. */
        [[nodiscard]] SubArrayBase* previous_subarray() const;
        /** @returns Parent linked vector. */
        [[nodiscard]] LinkedVectorBase* parent_linked_vector() const;
        
        /** @returns The domain of logical values stored in this SubArray. */
        [[nodiscard]] ValueDomain value_domain() const;
        /** @returns The storage policy for this SubArray. */
        [[nodiscard]] StoragePolicy storage_policy() const;

        /** @returns The number of logical values stored in this SubArray. */
        size_t size() const;
        /** @returns Global logical index of the first value stored here. */
        size_t starting_index() const;
        /** @returns The byte offset of this SubArray in the archive data file */
        size_t offset() const;
        
        /** @returns Whether absolute logical position `pos` belongs to this SubArray. */
        [[nodiscard]] bool contains(size_t pos) const;
        /** Convert an absolute logical index into a SubArray-local position. */
        [[nodiscard]] size_t local_index(size_t pos) const;
        

        /** @returns Physical occupancy of the current in-memory representation. */
        [[nodiscard]] virtual size_t mem_size() const = 0;

        /** Release the resident payload and let the manager drop transient state. */
        virtual void free_values() = 0; // useless ?
        
        /** Update the persisted payload offset recorded for this SubArray. */
        void set_offset(size_t offset); // useless ?


        /********* Functions for accessing the SubArray data *********/
        
        /** Reconstruct the logical value stored at absolute index `pos`. */
        [[nodiscard]] virtual uint64_t at(size_t pos) const = 0;
        /** Convenience alias for `at()` used by linked-vector call sites. */
        [[nodiscard]] uint64_t operator[](size_t pos) const;
        /** Materialise all logical values into `given_array` in logical order. */
        virtual void copy_values(uint64_t* given_array) const = 0;
        /** @returns Maximum physical occupancy allowed by the attached manager. */
        [[nodiscard]] virtual size_t capacity() const = 0;
        /** @retval true - The SubArray payload is currently resident in memory.
         *  @retval false - The SubArray payload has been freed and must be reloaded. */
        [[nodiscard]] virtual bool has_values() const = 0;
        /** @returns The statistics for this SubArray. */
        SubArrayStats& subarray_stats() const;

        /** @returns First logical timestamp stored in this SubArray. */
        [[nodiscard]] uint64_t first_value() const;
        /** @returns Last logical timestamp stored in this SubArray. */
        [[nodiscard]] uint64_t last_value() const;

        /********* Functions for adding data to the SubArray *********/
        /** Append one logical value to the manager-controlled representation
         *  and updates statistics.
         */
        AddStatus add(uint64_t val);

        /** Finalize the current subArray once it is full. */
        virtual void finalize_block() = 0 ; // todo: return statistics on the data block to decide if we could use a lossy encoding or not
    protected:
        /** Append one logical value to the manager-controlled representation.
         * A child class should implement this function to actually store val
         * This function is called by the SubArrayBase::add() function before
         * updating the statistics and the size of the SubArray. 
         * */
        virtual AddStatus add_value(uint64_t val) = 0;
        

#if 0
    protected:
        /** @returns Direct access to the backing buffer used by the manager. */
        //[[nodiscard]] uint64_t* raw_buffer(); // TODO: useless ?
        /** Recreate the policy manager after file-backed reconstruction. */
        //void rebuild_manager(); // useless ?

    

    public:
        /** @returns Value domain served by this SubArray. */
        //[[nodiscard]] ValueDomain domain() const; // renamed value_domain()
        /** @returns Storage policy encoded in this SubArray. */
//        [[nodiscard]] StoragePolicy policy() const; // renamed storage_policy()
        /** @returns Concrete lossy policy variant associated with this SubArray. */
//        [[nodiscard]] LossyPolicy lossy_policy() const; // useless ?
        /** @returns Current lifecycle phase of this SubArray instance. */
        //[[nodiscard]] SubArrayPhase phase() const;  useless ?

        /** Pack the persisted storage and lossy policy flags into one byte. */
        //[[nodiscard]] uint8_t pack_subarray_flags() const; //useless
        /** Decode persisted policy flags from the compact on-disk header byte. */
        // void unpack_subarray_flags(uint8_t encoded_policy); // useless
#endif
    protected:
   
        /** Write the common SubArray header shared by timestamp and duration variants. */
        //void write(File* info_file) const; // TODO
        //virtual write_values(File* info_file) const; // TODO

        /** Read the common SubArray header before rebuilding the manager state. */
        //void read_common_header(File* info_file); // TODO
        /** Reload the persisted payload for this SubArray from the data file. */
        //void load_data(File* data_file, const ParameterHandler& parameter_handler); // TODO
    };

    #if 0

    /**
     * Timestamp-specialised SubArray shell used by `TimeLinkedVector`.
     *
     * This subclass keeps the common `SubArrayBase` structure but adds the small amount of timestamp-specific 
     * metadata needed by exact and lossy managers, namely cached first/last logical timestamps and timestamp 
     * header helpers for archive I/O.
     */
    class TimeSubArray : public SubArrayBase {
    public:
        /** Runtime-write constructor used while appending timestamp values. */
        explicit TimeSubArray(StoragePolicy policy = StoragePolicy::None,
                            TimeSubArray* previous = nullptr,
                            const ParameterHandler* parameter_handler = nullptr,
                            LinkedVectorBase* parent = nullptr);
        /** File-backed constructor used while reconstructing archived timestamp subarrays. */
        explicit TimeSubArray(File* info_file, TimeSubArray* previous = nullptr);

        /** Append one timestamp value and refresh timestamp-specific cached bounds. */
        AddStatus add(uint64_t val) override;
        /** Persist the timestamp payload handled by the attached manager. */
        void write_data(File* file, const ParameterHandler* parameter_handler);
        /** Write timestamp-specific header fields after the common SubArray header. */
        void write_header(File* info_file) const;
        /** Read timestamp-specific header fields after the common SubArray header. */
        void read_header(File* info_file);

        

    protected:
        friend class DeltaManager;
        friend class PLAManager;
        friend class DurationSpikeManager;

        /** First logical timestamp covered by this SubArray. */
        uint64_t first_timestamp = 0;
        /** Last logical timestamp covered by this SubArray. */
        uint64_t last_timestamp = 0;
    };

    /**
     * Duration-specialised SubArray shell used by `DurationLinkedVector`.
     *
     * In addition to the common `SubArrayBase` metadata, this subclass tracks the per-SubArray duration aggregates needed for duration
     * -specific headers and quick summary queries. These statistics are maintained while values are appended and are persisted alongside 
     * the common SubArray metadata.
     */
    class DurationSubArray : public SubArrayBase {
    public:
        /** Runtime-write constructor used while appending duration values. */
        explicit DurationSubArray(StoragePolicy policy = StoragePolicy::None,
                                DurationSubArray* previous = nullptr,
                                const ParameterHandler* parameter_handler = nullptr,
                                LinkedVectorBase* parent = nullptr);
        /** File-backed constructor used while reconstructing archived duration subarrays. */
        explicit DurationSubArray(File* info_file, DurationSubArray* previous = nullptr);

        /** Append one duration value and refresh the cached aggregate statistics. */
        AddStatus add(uint64_t val) override;
        /** Persist the duration payload handled by the attached manager. */
        void write_data(File* file, const ParameterHandler* parameter_handler);
        /** Write duration-specific header fields after the common SubArray header. */
        void write_header(File* info_file) const;
        /** Read duration-specific header fields after the common SubArray header. */
        void read_header(File* info_file);
        /** Update min/max/running-mean state after accepting one duration value. */
        void update_statistics(uint64_t current_value);
        /** Finalise the mean value once runtime accumulation is complete. */
        void final_update_mean();

        /** @returns Minimum logical duration stored in this SubArray. */
        [[nodiscard]] uint64_t min_value() const;
        /** @returns Maximum logical duration stored in this SubArray. */
        [[nodiscard]] uint64_t max_value() const;
        /** @returns Mean logical duration stored in this SubArray. */
        [[nodiscard]] uint64_t mean_value() const;

    protected:
        /** Minimum duration observed among the logical values stored here. */
        uint64_t min_duration = UINT64_MAX;
        /** Maximum duration observed among the logical values stored here. */
        uint64_t max_duration = 0;
        /** Mean duration cached for this SubArray. */
        uint64_t mean_duration = 0;
        /** Tracks whether `mean_duration` currently stores a finalized mean or a running sum. */
        bool mean_duration_is_finalized = false;
    };

    #endif

    
    /** Append one unsigned integer to the payload using variable-length encoding. */
    inline void write_varint(uint64_t x, uint8_t*& out) {
        while (x >= 0x80) {
            *out++ = static_cast<uint8_t>((x & 0x7f) | 0x80);
            x >>= 7;
        }
        *out++ = static_cast<uint8_t>(x);
    }

    /** Decode one variable-length unsigned integer from the payload cursor. */
    [[nodiscard]] inline uint64_t read_varint(const uint8_t*& p, const uint8_t* end) {
        uint64_t result = 0;
        int shift = 0;
        while (p < end) {
            const uint8_t byte = *p++;
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

}

namespace pallas {
// TODO: move this to subarray_pla.h
/** Chooses the concrete lossy predictor used inside a lossy manager. */
    enum class LossyPolicy : uint8_t {
        /** Piecewise linear approximation with 4-sample segments. */
        PLA4 = 0,
        /** Piecewise linear approximation with 8-sample segments. */
        PLA8 = 1,
        /** Piecewise linear approximation with 16-sample segments. */
        PLA16 = 2,
        /** Piecewise linear approximation with 32-sample segments. */
        PLA32 = 3,
        /** Duration-spike predictor with a small spike budget (`k_max = 4`). */
        Spike4 = 4,
        /** Duration-spike predictor with a moderate spike budget (`k_max = 8`). */
        Spike8 = 5,
        /** Duration-spike predictor with a larger spike budget (`k_max = 16`). */
        Spike16 = 6,
        /** Duration-spike predictor with the largest spike budget (`k_max = 32`). */
        Spike32 = 7,
    };

    /** Default lossy policy for timestamp-oriented linked vectors. */
    constexpr LossyPolicy DEFAULT_LOSSY_TIME = LossyPolicy::PLA8;
    /** Default lossy policy for duration-oriented linked vectors. */
    constexpr LossyPolicy DEFAULT_LOSSY_DURATION = LossyPolicy::Spike8;
}


//#include "pallas_subarray_delta.h"
//#include "pallas_subarray_pla.h"
#include "pallas_subarray_raw.h"

#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
