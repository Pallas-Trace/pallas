/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#pragma once

#include "pallas_subarray.h"

namespace pallas {

    /**
     * Exact delta-encoding manager shared by timestamp and duration SubArrays.
     *
     * Timestamp streams only generate non-negative deltas, while duration streams can produce signed delta behaviour around the previous value.
     * These paths originally existed as separate implementations, but were unified behind one manager to avoid duplicating the same payload,
     * checkpointing, and serialization machinery. The manager keeps a common outer structure and dispatches to small domain-specific helpers 
     * where the encoding details differ.
     */
    class DeltaManager : public Manager {
    public:
        DeltaManager(SubArrayBase& parent, ValueDomain value_domain)
            : Manager(parent), dom(value_domain) {}

        [[nodiscard]] size_t _capacity() const override;
        AddStatus add(uint64_t val) override;
        [[nodiscard]] uint64_t at(size_t pos) const override;
        void copy_to_array(uint64_t* given_array) const override;
        void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
        void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
        void on_values_freed() override;

    private:
        /** Interprets the previous delta as signed or unsigned without copying. */
        union PrevDelta {
            uint64_t u;
            int64_t i;

            PrevDelta()
                : u(0) {}
        };

        /** Restart point used to bound decode work during random access. */
        struct Checkpoint {
            /** Logical index covered by this checkpoint. */
            size_t idx = 0;
            /** Encoded payload byte offset at `idx`. */
            size_t off = 0;
            /** Reconstructed value at `idx`. */
            uint64_t val = 0;
            /** Previous delta state needed to resume decoding. */
            PrevDelta prev;
        };

        /** Rolling state maintained while online delta-encoding one SubArray. */
        struct State {
            /** Last reconstructed logical value. */
            uint64_t last = 0;
            /** Previous first-order delta used by the codec. */
            PrevDelta prev;
        };

        /** Spacing between checkpoints inserted into the encoded payload. */
        static constexpr size_t kCheckpointStride = 50;
        /** Select the timestamp-specialised delta path. */
        [[nodiscard]] bool is_time_domain() const {
            return dom == ValueDomain::Timestamp;
        }
        /** Append one timestamp value using the timestamp delta codec. */
        AddStatus add_time(uint64_t val);
        /** Append one duration value using the duration delta codec. */
        AddStatus add_duration(uint64_t val);
        /** Decode one timestamp value, resuming from the nearest checkpoint. */
        [[nodiscard]] uint64_t at_time(size_t pos) const;
        /** Decode one duration value, resuming from the nearest checkpoint. */
        [[nodiscard]] uint64_t at_duration(size_t pos) const;
        /** Materialise the full timestamp payload into a flat array. */
        void copy_time_to_array(uint64_t* given_array) const;
        /** Materialise the full duration payload into a flat array. */
        void copy_duration_to_array(uint64_t* given_array) const;
        /** Reload timestamp payload state from the persisted SubArray bytes. */
        void load_time_data(FILE* data_file, const ParameterHandler& parameter_handler);
        /** Reload duration payload state from the persisted SubArray bytes. */
        void load_duration_data(FILE* data_file, const ParameterHandler& parameter_handler);

        /** Whether this manager is serving timestamp or duration values. */
        ValueDomain dom;
        /** Byte-oriented payload view used by the packed delta codec. */
        uint8_t* payload = nullptr;
        /** Number of payload bytes currently occupied. */
        size_t bytes = 0;
        /** Total payload capacity in bytes for the current SubArray buffer. */
        size_t cap_bytes = 0;
        /** Rolling encoder state for the current append position. */
        State st;
        /** Sparse decode checkpoints used to avoid replaying the full payload. */
        std::vector<Checkpoint> cps;
    #ifdef BMARK
        /** Exact logical values retained only for benchmark-side error accounting. */
        std::vector<uint64_t> shadow_values;
    #endif
    };

}
