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
#include "pallas_pla.h"

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

#include "pallas/utils/pallas_parameter_handler.h"
#ifdef BMARK
#include "pallas/utils/pallas_bmark.h"
#endif

#ifndef DEFAULT_VECTOR_SIZE
#define DEFAULT_VECTOR_SIZE 1000
#endif


#ifndef VECTOR_SIZE_32 
#define VECTOR_SIZE_32 32
#endif

#ifndef VECTOR_SIZE_2048 
#define VECTOR_SIZE_2048 2048
#endif


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

/** Policy-manager interfaces used by SubArray implementations. */
namespace pallas {

class SubArrayBase;
class TimeSubArray;
class DurationSubArray;
class LinkedVectorBase;

/**
 * Smart policy object attached to a SubArray shell.
 *
 * `SubArrayBase` is intentionally a lightweight shell: it owns common metadata, the backing buffer, and the archive-facing lifetime hooks, but it
 * does not itself implement the storage algorithm. A concrete `Manager` supplies that behaviour by deciding how values are appended, reconstructed,
 * materialised, and serialised. This split keeps the SubArray shape stable while allowing different `StoragePolicy` implementations to plug in behind
 * the same outer container.
 */
class Manager {
   public:
    /** Bind the manager to the owning SubArray instance. */
    explicit Manager(SubArrayBase& parent)
        : parent(parent) {}
    /** Virtual destructor for polymorphic policy objects. */
    virtual ~Manager();

    /** Return the physical capacity used by this policy inside one SubArray. */
    [[nodiscard]] virtual size_t _capacity() const = 0;
    /** Append one logical value to the current SubArray representation. */
    virtual AddStatus add(uint64_t val) = 0;

    /** Reconstruct the logical value stored at absolute position `pos`. */
    [[nodiscard]] virtual uint64_t at(size_t pos) const = 0;
    /** Materialise all logical values into `given_array` in order. */
    virtual void copy_to_array(uint64_t* given_array) const = 0;
    /** Persist the current SubArray payload to the archive data file. */
    virtual void write_data(FILE* data_file, const ParameterHandler* parameter_handler) = 0;
    /** Reload the persisted SubArray payload from the archive data file. */
    virtual void load_data(FILE* data_file, const ParameterHandler& parameter_handler) = 0;
    /** Optional hook run once the parent SubArray has finished its common setup. */
    virtual void on_subarray_initialized();
    /** Release policy-owned transient state after SubArray values are freed. */
    virtual void on_values_freed() = 0;

   protected:
    /** Owning SubArray whose metadata and buffers this manager operates on. */
    SubArrayBase& parent;
};

/**
 * Exact manager that stores values in the SubArray buffer without applying an internal encoding transform.
 *
 * This is the simplest policy: logical order and physical order match, random access is direct, and serialisation forwards 
 * the raw buffer to the shared compression helpers. It acts as the baseline implementation for `StoragePolicy::None`.
 */
class NoneManager : public Manager {
   public:
    explicit NoneManager(SubArrayBase& parent)
        : Manager(parent) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_values_freed() override;
};

/**
 * Exact delta-encoding manager shared by timestamp and duration SubArrays.
 *
 * Timestamp streams only generate non-negative deltas, while duration streams can produce signed delta behaviour around the previous value.
 * These paths originally existed as separate implementations, but were unified behind one manager to avoid duplicating the same payload,
 * checkpointing, and serialisation machinery. The manager keeps a common outer structure and dispatches to small domain-specific helpers 
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

/**
 * Lossy timestamp manager based on blockwise PLA-style approximation.
 *
 * This path is only pseudo-online: values are accepted through `add()` as if they were being encoded incrementally, but the actual PLA compaction 
 * happens once a full SubArray-sized block has been collected. That design keeps the public linked-vector interface online while still letting the 
 * manager run a stronger block analysis (some time) before serialisation. The tradeoff is that `add()` carries extra staging and bookkeeping overhead 
 * compared to exact policies.
 */
class PLAManager : public Manager {
   public:
    PLAManager(SubArrayBase& parent, uint8_t k_max)
        : Manager(parent), k_max(k_max) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_subarray_initialized() override;
    void on_values_freed() override;

   private:
    /** Allocate and bind the temporary helper buffers needed by block compaction. */
    void ensure_staging();
    /** Build the compact PLA representation once the staging block is full. */
    void finalize_block();
    /** Pack the compact anchor representation into the SubArray buffer. */
    void write_packed_payload();
    /** Reload anchors and helper state from the packed SubArray payload. */
    void load_packed_payload();
    /** Reconstruct one logical timestamp from the compact anchor representation. */
    [[nodiscard]] uint64_t interpolate_value(const TimeSubArray& subarray, size_t logical_index) const;
    /** Drop transient staging and compaction state after values are freed. */
    void clear_state();

    /** Policy knob selecting the maximum anchor budget used by the PLA builder. */
    uint8_t k_max = 0;
    /** Number of anchors emitted into `anchor_storage` for the current block. */
    uint8_t anchor_count = 0;
    /** Whether the current full block has already been compacted. */
    bool compact_ready = false;
    /** ScratchPad views and statistics used while analysing one full PLA block. */
    GammaBlockStats stats{};
    /** Fixed storage for the anchors produced by the block compaction step. */
    PLAAnchor anchor_storage[kPLAMaxAnchors]{};
};

/**
 * Lossy duration manager that models a block around a clipped baseline and a bounded set of spike exceptions.
 *
 * Like `PLAManager`, this path is only pseudo-online. Values are accepted through `add()` during runtime, but the actual model fitting and spike
 * selection happen as a post-processing step once the current SubArray block is full. This keeps the outer linked-vector API online while allowing the
 * manager to inspect the complete block before deciding which values become exact spikes, grouped spikes, or baseline-covered samples.
 */
class DurationSpikeManager : public Manager {
   public:
    DurationSpikeManager(SubArrayBase& parent, uint8_t k_max)
        : Manager(parent), k_max(k_max) {}

    [[nodiscard]] size_t _capacity() const override;
    AddStatus add(uint64_t val) override;
    [[nodiscard]] uint64_t at(size_t pos) const override;
    void copy_to_array(uint64_t* given_array) const override;
    void write_data(FILE* data_file, const ParameterHandler* parameter_handler) override;
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler) override;
    void on_subarray_initialized() override;
    void on_values_freed() override;

   private:
    /** One logical sample kept exactly because it does not fit the baseline well. */
    struct ExactSpike {
        /** Logical index of the exact exception. */
        uint16_t idx = 0;
        /** Exact stored duration value. */
        uint64_t value = 0;
    };

    /** Small group of samples that share the same representative spike value. */
    struct SpikeGroup {
        /** Representative value assigned to every member of the group. */
        uint64_t value = 0;
        /** Number of logical indices stored in `indices`. */
        uint8_t index_count = 0;
        /** Logical indices that reconstruct to `value`. */
        std::array<uint16_t, 64> indices{};
    };

    /** Hard cap on individually stored exact spikes per compacted block. */
    static constexpr uint8_t kMaxExactSpikes = 8;
    /** Hard cap on grouped spike buckets per compacted block. */
    static constexpr uint8_t kMaxSpikeGroups = 4;
    /** Minimum population required before a spike group is worth keeping. */
    static constexpr uint8_t kMinGroupSize = 2;
    /** Relative tolerance used when clustering residual spikes into groups. */
    static constexpr double kRelativeGroupTolerance = 0.20;
    /** Absolute tolerance used when clustering residual spikes into groups. */
    static constexpr double kAbsoluteGroupTolerance = 96.0;
    /** Minimum residual magnitude required before a value is treated as a spike. */
    static constexpr double kMinSpikeResidual = 64.0;
    /** Sigma-based clipping factor used while estimating the baseline window. */
    static constexpr double kBaselineClipSigma = 2.5;
    /** Minimum absolute clipping radius retained around the baseline mean. */
    static constexpr double kBaselineMinClipRadius = 32.0;

    /** Analyse the staged block and build the baseline-plus-spikes model. */
    void finalize_block();
    /** Reset transient block-analysis state after values are freed. */
    void clear_state();
    /** Pack the compact baseline and spike metadata into the SubArray buffer. */
    void write_packed_payload();
    /** Reload the packed baseline and spike metadata from persisted bytes. */
    void load_packed_payload();
    /** Return the exact payload size needed for the packed compact form. */
    [[nodiscard]] size_t packed_payload_bytes() const;
    /** Reconstruct one logical duration from the baseline and spike metadata. */
    [[nodiscard]] uint64_t reconstructed_value(size_t logical_index) const;

    /** Whether the current staged block has already been compacted. */
    bool compact_ready = false;
    /** Whether packed payload bytes are ready to be written or replayed. */
    bool packed_payload_ready = false;
    /** Policy knob controlling the spike budget chosen by `LossyPolicy::Spike*`. */
    uint8_t k_max = 0;
    /** Mean of the clipped baseline used for non-spike samples. */
    uint64_t baseline_mean = 0;
    /** Baseline spread encoded alongside the compacted block. */
    uint32_t baseline_stddev = 0;
    /** Number of populated entries in `exact_spikes`. */
    uint8_t exact_count = 0;
    /** Number of populated entries in `spike_groups`. */
    uint8_t group_count = 0;
    /** Individually stored exceptions for values that remain too important to merge. */
    std::array<ExactSpike, kMaxExactSpikes> exact_spikes{};
    /** Grouped spike buckets used to represent repeated exceptional values compactly. */
    std::array<SpikeGroup, kMaxSpikeGroups> spike_groups{};
};

/** Shared delta-codec helpers used by the packed exact and lossy payload paths. */
/** Map a signed delta to an unsigned integer with small magnitudes near zero. */
[[nodiscard]] inline uint64_t zigzag_encode(int64_t x) {
    return (static_cast<uint64_t>(x) << 1) ^ static_cast<uint64_t>(x >> 63);
}

/** Recover the original signed delta from its zigzag-encoded unsigned form. */
[[nodiscard]] inline int64_t zigzag_decode(uint64_t x) {
    return static_cast<int64_t>((x >> 1) ^ static_cast<uint64_t>(-static_cast<int64_t>(x & 1)));
}

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

/** SubArray storage shells shared by timestamp and duration linked-vector paths. */
namespace pallas {
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
    SubArrayBase* next = nullptr;
    /** Previous SubArray in the owning linked-vector chain. */
    SubArrayBase* prev = nullptr;
    /** Number of logical values represented by this SubArray. */
    size_t value_count = 0;
    /** Physical occupancy used inside the backing representation. */
    size_t physical_size = 0;
    /** Global logical index of the first value stored in this SubArray. */
    size_t first_index = 0;
    /** Byte offset of this SubArray payload in the archive data file. */
    size_t file_offset = 0;

    /** Policy object that implements append, decode, and I/O behaviour. */
    std::unique_ptr<Manager> manager;
    /** Backing buffer used while the SubArray payload is resident in memory. */
    uint64_t* buffer = nullptr;
    /** Owning linked vector, used for policy context and benchmark attribution. */
    LinkedVectorBase* parent_lv = nullptr;

   public:
    /** @returns Next SubArray in the linked chain, or `nullptr` at the tail. */
    [[nodiscard]] SubArrayBase* next_subarray() const;
    /** @returns Previous SubArray in the linked chain, or `nullptr` at the head. */
    [[nodiscard]] SubArrayBase* previous_subarray() const;
    /** @returns Number of logical values represented by this SubArray. */
    [[nodiscard]] size_t size() const;
    /** @returns Physical occupancy of the current in-memory representation. */
    [[nodiscard]] size_t mem_size() const;
    /** @returns Global logical index of the first value stored here. */
    [[nodiscard]] size_t starting_index() const;
    /** @returns Persisted byte offset of this SubArray payload in the data file. */
    [[nodiscard]] size_t offset() const;
    /** Update the persisted payload offset recorded for this SubArray. */
    void set_offset(size_t offset);
#ifdef BMARK
    /** @returns Benchmark family inherited from the owning linked vector. */
    [[nodiscard]] BmarkFamily get_bmark_family() const;
#endif

    /** Virtual destructor for polymorphic timestamp/duration SubArray ownership. */
    virtual ~SubArrayBase();
    /** Append one logical value to the manager-controlled representation. */
    virtual AddStatus add(uint64_t val) = 0;
    /** Reconstruct the logical value stored at absolute index `pos`. */
    [[nodiscard]] uint64_t at(size_t pos) const;
    /** Convenience alias for `at()` used by linked-vector call sites. */
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    /** Materialise all logical values into `given_array` in logical order. */
    void copy_values(uint64_t* given_array) const;
    /** @returns Maximum physical occupancy allowed by the attached manager. */
    [[nodiscard]] size_t capacity() const;
    /** @retval true - The SubArray payload is currently resident in memory.
     *  @retval false - The SubArray payload has been freed and must be reloaded. */
    [[nodiscard]] bool has_values() const;
   protected:
    /** @returns Whether absolute logical position `pos` belongs to this SubArray. */
    [[nodiscard]] bool contains(size_t pos) const;
    /** Convert an absolute logical index into a SubArray-local position. */
    [[nodiscard]] size_t local_index(size_t pos) const;
    /** @returns Direct access to the backing buffer used by the manager. */
    [[nodiscard]] uint64_t* raw_buffer();
    /** Release the resident payload and let the manager drop transient state. */
    void free_values();
    /** Recreate the policy manager after file-backed reconstruction. */
    void rebuild_manager();

   protected:
    /** Value domain served by this SubArray: timestamps or durations. */
    ValueDomain value_domain;
    /** Exact or lossy storage policy selected for this SubArray. */
    StoragePolicy storage_policy = StoragePolicy::None;
    /** Concrete lossy variant recorded when `storage_policy` is lossy. */
    LossyPolicy lossy_storage_policy = DEFAULT_LOSSY_TIME;
    /** Whether the SubArray is being used for runtime writes or analysis reads. */
    SubArrayPhase subarray_phase = SubArrayPhase::RuntimeWrite;

   public:
    /** @returns Value domain served by this SubArray. */
    [[nodiscard]] ValueDomain domain() const;
    /** @returns Storage policy encoded in this SubArray. */
    [[nodiscard]] StoragePolicy policy() const;
    /** @returns Concrete lossy policy variant associated with this SubArray. */
    [[nodiscard]] LossyPolicy lossy_policy() const;
    /** @returns Current lifecycle phase of this SubArray instance. */
    [[nodiscard]] SubArrayPhase phase() const;

    /** Pack the persisted storage and lossy policy flags into one byte. */
    [[nodiscard]] uint8_t pack_subarray_flags() const;
    /** Decode persisted policy flags from the compact on-disk header byte. */
    void unpack_subarray_flags(uint8_t encoded_policy);

   protected:
    /** Access control for manager implementations and the owning linked vector. */
    friend class Manager;
    friend class NoneManager;
    friend class DeltaManager;
    friend class PLAManager;
    friend class DurationSpikeManager;
    friend class LinkedVectorBase;

    /** Runtime-write constructor used when appending to a live linked vector. */
    explicit SubArrayBase(ValueDomain domain,
                          StoragePolicy policy = StoragePolicy::None,
                          SubArrayBase* previous = nullptr,
                          const ParameterHandler* parameter_handler = nullptr,
                          LinkedVectorBase* parent = nullptr);
    /** File-backed constructor used while reconstructing archived SubArrays. */
    explicit SubArrayBase(FILE* info_file, ValueDomain domain, SubArrayBase* previous = nullptr);

    /** Write the common SubArray header shared by timestamp and duration variants. */
    void write_common_header(FILE* info_file) const;
    /** Read the common SubArray header before rebuilding the manager state. */
    void read_common_header(FILE* info_file);
    /** Reload the persisted payload for this SubArray from the data file. */
    void load_data(FILE* data_file, const ParameterHandler& parameter_handler);
};

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
    explicit TimeSubArray(FILE* info_file, TimeSubArray* previous = nullptr);

    /** Append one timestamp value and refresh timestamp-specific cached bounds. */
    AddStatus add(uint64_t val) override;
    /** Persist the timestamp payload handled by the attached manager. */
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    /** Write timestamp-specific header fields after the common SubArray header. */
    void write_header(FILE* info_file) const;
    /** Read timestamp-specific header fields after the common SubArray header. */
    void read_header(FILE* info_file);

    /** @returns First logical timestamp stored in this SubArray. */
    [[nodiscard]] uint64_t first_value() const;
    /** @returns Last logical timestamp stored in this SubArray. */
    [[nodiscard]] uint64_t last_value() const;

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
    explicit DurationSubArray(FILE* info_file, DurationSubArray* previous = nullptr);

    /** Append one duration value and refresh the cached aggregate statistics. */
    AddStatus add(uint64_t val) override;
    /** Persist the duration payload handled by the attached manager. */
    void write_data(FILE* file, const ParameterHandler* parameter_handler);
    /** Write duration-specific header fields after the common SubArray header. */
    void write_header(FILE* info_file) const;
    /** Read duration-specific header fields after the common SubArray header. */
    void read_header(FILE* info_file);
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

}  

#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
