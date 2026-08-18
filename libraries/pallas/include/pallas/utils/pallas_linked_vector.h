/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * Clean linked-vector scaffold built on top of the standalone subarray layer.
 */
#pragma once

#ifndef __cplusplus

typedef struct TimeLinkedVector {
} TimeLinkedVector;

typedef struct DurationLinkedVector {
} DurationLinkedVector;

#else

#include <cstddef>
#include <cstdint>
#include <array>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#ifdef BMARK
#include "pallas/utils/pallas_bmark.h"
#endif

#include "pallas_parameter_handler.h"
#include "pallas_subarray.h"

namespace pallas {

/**
 * @brief Small fixed-size cache of recently accessed logical values.
 *
 * This ring buffer stores `(logical_index, value)` pairs from the most recent
 * linked-vector values pushed through two paths: newly appended values recorded
 * by `TimeLinkedVector::add()` / `DurationLinkedVector::add()`, and values materialized during
 * indexed reads in `LinkedVectorBase::at()`. It is used as a fast path for repeated reads
 * of very recent positions, avoiding an additional subarray lookup or decode
 * step when the value is still in the cache.
 */
class RecentValueRingBuffer {
   public:
    /** Maximum number of recent values remembered at once. */
    static constexpr size_t kCapacity = 64;

    /**
     * @brief Clears all remembered entries from the cache.
     */
    void clear() {
        next_slot = 0;
        entry_count = 0;
    }

    /**
     * @brief Records one logical value in the ring buffer.
     * @param index Logical index of the cached value.
     * @param value Value observed at that logical index.
     */
    void push(size_t index, uint64_t value) {
        entries[next_slot] = Entry{index, value};
        next_slot = (next_slot + 1) % kCapacity;
        if (entry_count < kCapacity) {
            entry_count++;
        }
    }

    /**
     * @brief Searches the cache for a recently seen logical index.
     * @param index Logical index being queried.
     * @param value Output parameter filled with the cached value on success.
     * @retval true  A cached value for `index` was found.
     * @retval false No cached value for `index` is currently stored.
     */
    [[nodiscard]] bool lookup(size_t index, uint64_t& value) const {
        for (size_t i = 0; i < entry_count; ++i) {
            const size_t offset = (next_slot + kCapacity - 1 - i) % kCapacity;
            const auto& entry = entries[offset];
            if (entry.index == index) {
                value = entry.value;
                return true;
            }
        }
        return false;
    }

   private:
    struct Entry {
        size_t index = 0;
        uint64_t value = 0;
    };

    std::array<Entry, kCapacity> entries{};
    size_t next_slot = 0;
    size_t entry_count = 0;
};

/**
 * @brief Small fixed-size cache of recently useful subarrays.
 *
 * This cache stores pointers to subarrays that were recently involved in
 * indexed lookups. It acts as a fast path before the linked vector falls back
 * to its broader subarray index and binary-search lookup path.
 */
class RecentSubArrayCache {
   public:
    /** Maximum number of cached subarray pointers remembered at once. */
    static constexpr size_t kCapacity = 8;

    /**
     * @brief Clears all cached subarray entries and resets their generation state.
     */
    void clear() {
        generation = 0;
        for (auto& entry : entries) {
            entry = {};
        }
    }

    /**
     * @brief Searches the cache for a subarray that contains the queried logical index.
     * @param index Logical value index being resolved.
     * @param probes Output counter used to accumulate how many cache entries were inspected.
     * @returns Pointer to the cached subarray that contains `index`, or `nullptr` if no cache hit occurs.
     */
    [[nodiscard]] const SubArrayBase* lookup(size_t index, uint64_t& probes) const;
    /**
     * @brief Inserts or refreshes a subarray entry in the recent-subarray cache.
     * @param subarray Subarray that was just used during a lookup and should be remembered.
     */
    void remember(SubArrayBase* subarray) const;

   private:
    struct Entry {
        SubArrayBase* subarray = nullptr;
        uint32_t hits = 0;
        uint64_t generation = 0;
    };

    mutable std::array<Entry, kCapacity> entries{};
    mutable uint64_t generation = 0;
};

/**
 * @brief Common linked-vector base class shared by timestamp and duration vectors.
 *
 * `LinkedVectorBase` exposes the logical vector interface used by the rest of the runtime
 * while hiding the physical subarray layout underneath. It owns the linked list
 * of subarrays, the auxiliary lookup structures used by the read path, and the
 * policy state needed to create or reconstruct storage-policy-specific subarrays.
 *
 * Derived classes such as `TimeLinkedVector` and `DurationLinkedVector` specialize this base by
 * fixing the value domain and by creating the appropriate `SubArrayBase`
 * subclass for that domain.
 */
class LinkedVectorBase {
   protected:
    /** Number of logical values currently stored in the linked vector. */
    size_t value_count = 0;
    /** Reference count preserved for compatibility with the older linked-vector interface. */
    size_t reference_count = 0;
    /** Total number of subarrays currently chained into this vector. */
    size_t subarray_total = 0;
    /** Indicates whether the vector currently has a fully contiguous representation. */
    bool is_contiguous = false;
    /** Shared runtime configuration used for policy resolution, memory limits, and file I/O helpers. */
    ParameterHandler& parameter_handler;
    /** Path to the value file used when subarray payloads are reloaded on demand. */
    const char* file_path = nullptr;

    /** Set of subarrays whose payloads are currently resident in memory. */
    std::set<SubArrayBase*> loaded_subarrays;
    /** Small cache of recent logical values used by the read fast path. */
    mutable RecentValueRingBuffer recent_values;
    /** Small cache of recent subarray hits used before broader indexed lookup. */
    mutable RecentSubArrayCache recent_subarrays;
    /** Auxiliary lookup array used to resolve logical positions to subarrays quickly. */
    mutable std::vector<SubArrayBase*> subarray_index;

    /** Value domain served by this linked vector. */
    ValueDomain value_domain;
    /** Default storage policy used when new subarrays are created. */
    StoragePolicy storage_policy = StoragePolicy::None;
    /** Shared scratch buffer used by selected codecs and manager implementations. */
    void* hbuffer = nullptr;
    /** Size in bytes of the currently allocated helper buffer. */
    size_t hbuffer_bytes = 0;
#ifdef BMARK
    /** Benchmark family used when BMARK instrumentation is enabled. */
    BmarkFamily benchmark_family = BmarkFamily::Unknown;
#endif

    /** First subarray in the linked-vector chain. */
    SubArrayBase* first = nullptr;
    /** Tail subarray used as the append target for runtime writes. */
    SubArrayBase* last = nullptr;

   public:
    /** Virtual destructor for polymorphic linked-vector ownership. */
    virtual ~LinkedVectorBase();

    /** @returns Number of logical values stored in the vector. */
    [[nodiscard]] size_t size() const {
        return value_count;
    }
    /** @returns Compatibility reference count associated with this vector. */
    [[nodiscard]] size_t ref_count() const {
        return reference_count;
    }
    /** @returns Number of subarrays chained into this vector. */
    [[nodiscard]] size_t subarray_count() const {
        return subarray_total;
    }
    /** @returns Number of subarrays whose payloads are currently loaded in memory. */
    [[nodiscard]] size_t loaded_subarray_count() const {
        return loaded_subarrays.size();
    }
    /** @retval true - The vector contains no logical values.
     *  @retval false - The vector contains at least one logical value. */
    [[nodiscard]] bool empty() const {
        return value_count == 0;
    }
    /** @returns Whether the vector is currently represented contiguously. */
    [[nodiscard]] bool contiguous() const {
        return is_contiguous;
    }
    /**
     * @brief Clears persisted file offsets recorded by each subarray.
     *
     */
    void reset_offsets();

    /**
     * @brief Returns the logical value stored at index `pos`.
     * @param pos Logical value index.
     * @returns Value stored at `pos`.
     */
    [[nodiscard]] uint64_t at(size_t pos) const;
    /**
     * @brief Fast indexed access helper for the logical value at `pos`.
     * @param pos Logical value index.
     * @returns Value stored at `pos`.
     */
    [[nodiscard]] uint64_t operator[](size_t pos) const;
    /** @returns First logical value stored in the vector. */
    [[nodiscard]] uint64_t front() const;
    /** @returns Last logical value stored in the vector. */
    [[nodiscard]] uint64_t back() const;
    /**
     * @brief Materializes the full logical stream into a freshly allocated flat array.
     * @returns Heap-allocated array containing all logical values in order.
     */
    [[nodiscard]] uint64_t* as_flat_array() const;
    /** @returns Human-readable string representation of the logical values. */
    [[nodiscard]] std::string values_to_string() const;

    /** @returns Value domain served by this linked vector. */
    [[nodiscard]] ValueDomain domain() const {
        return value_domain;
    }

    /**
     * @brief Updates the preferred storage policy used for future subarray creation.
     * @param policy Storage policy to apply to newly created subarrays.
     */
    void set_storage_policy(StoragePolicy policy) {
        storage_policy = policy;
    }
    /**
     * @brief Compatibility alias for `set_storage_policy()`.
     * @param policy Storage policy to apply to newly created subarrays.
     */
    void setPreferredStoragePolicy(StoragePolicy policy) {
        set_storage_policy(policy);
    }

    /** Loads every subarray payload in memory. */
    void load_all();
    /** Frees all currently loaded subarray payloads owned by this vector. */
    void free_data();
    /**
     * @brief Applies the current storage-policy preference to subsequent runtime writes.
     * @returns `true` if the policy state is valid and can be applied, `false` otherwise.
     */
    bool apply_storage_policy();

#ifdef BMARK
    /** @brief Sets the benchmark family associated with this vector. */
    void set_bmark_family(BmarkFamily family) {
        benchmark_family = family;
    }
    /** @returns Benchmark family associated with this vector. */
    [[nodiscard]] BmarkFamily get_bmark_family() const {
        return benchmark_family;
    }
#endif

   protected:
    friend class PLAManager;

    /**
     * @brief Runtime-write constructor for a fresh linked vector.
     * @param p Shared runtime parameter handler.
     * @param domain Value domain served by the vector.
     * @param _policy Initial storage policy used for subarray creation.
     */
    explicit LinkedVectorBase(ParameterHandler& p, ValueDomain domain, StoragePolicy _policy);
    /**
     * @brief File-backed reconstruction constructor.
     * @param vector_file Metadata stream from which the vector header is read.
     * @param value_file_path Path to the data stream used for lazy subarray reload.
     * @param p Shared runtime parameter handler.
     * @param domain Value domain served by the vector.
     * @param _policy Fallback policy associated with the reconstructed vector.
     * @param abi_version Trace ABI version used to interpret stored metadata.
     */
    explicit LinkedVectorBase(FILE* vector_file,
                    const char* value_file_path,
                    ParameterHandler& p,
                    ValueDomain domain,
                    StoragePolicy _policy,
                    uint8_t abi_version);

    /** Ensures that the shared helper buffer is at least `bytes` large. */
    void ensure_hbuffer(size_t bytes);
    /** Appends one subarray pointer to the auxiliary lookup index. */
    void append_subarray_index(SubArrayBase* subarray);
    /** Rebuilds the full auxiliary lookup index from the linked subarray chain. */
    void rebuild_subarray_index();
    /** Writes the linked-vector metadata common to both `TimeLinkedVector` and `DurationLinkedVector`. */
    void write_common_header(FILE* vector_file) const;
    /** Evicts loaded subarray payloads when runtime memory constraints require it. */
    void evict_loaded_subarrays();
    /** Loads one subarray payload from the backing value file. */
    void load_data(SubArrayBase* subarray);
    /** @returns Pointer to the shared helper buffer used by codec-specific logic. */
    [[nodiscard]] void* helper_buffer() const {
        return hbuffer;
    }
    /** @returns Size in bytes of the currently allocated helper buffer. */
    [[nodiscard]] size_t helper_buffer_size() const {
        return hbuffer_bytes;
    }

    /** Finds the mutable subarray that contains logical position `pos`. */
    [[nodiscard]] SubArrayBase* find_subarray(size_t pos);
    /** Finds the read-only subarray that contains logical position `pos`. */
    [[nodiscard]] const SubArrayBase* find_subarray(size_t pos) const;

    /**
     * @brief Creates the next subarray for the derived linked-vector type.
     * @param previous Previous tail subarray in the chain, or `nullptr` for the first one.
     * @returns Newly created domain-specific subarray.
     */
    virtual SubArrayBase* create_subarray(SubArrayBase* previous) const = 0;
    
   #if 1
    public:
    /** @returns Current default storage policy associated with this vector. */
    [[nodiscard]] StoragePolicy get_storage_policy() const {
        return storage_policy;
    }
    /** @returns Storage policy of every subarray currently chained into the vector. */
    [[nodiscard]] std::vector<StoragePolicy> get_sub_array_policies() const;
    /** @returns Storage policy of the currently loaded subarrays only. */
    [[nodiscard]] std::vector<StoragePolicy> get_loaded_sub_array_policies() const;
   #endif
};

/**
 * @brief Linked-vector specialization for timestamp-domain values.
 *
 * `TimeLinkedVector` stores ordered timestamp streams such as event timestamps and
 * sequence timestamps. It keeps the generic linked-vector mechanics from
 * `LinkedVectorBase` and specializes them by creating `TimeSubArray` instances and by
 * exposing timestamp-specific query helpers used during analysis.
 */
class TimeLinkedVector : public LinkedVectorBase {
   public:
    /** @brief Creates a fresh timestamp vector using the default policy from the parameter handler. */
    explicit TimeLinkedVector(ParameterHandler& p);
    /**
     * @brief Creates a fresh timestamp vector with an explicit storage policy.
     * @param p Shared runtime parameter handler.
     * @param _policy Initial storage policy used for newly created timestamp subarrays.
     */
    explicit TimeLinkedVector(ParameterHandler& p, StoragePolicy _policy);
    /**
     * @brief Reconstructs a timestamp vector from persisted metadata.
     * @param vector_file Metadata stream containing the linked-vector header.
     * @param value_file_path Backing value file used for lazy payload reload.
     * @param p Shared runtime parameter handler.
     * @param abi_version Trace ABI version used to interpret persisted metadata.
     */
    TimeLinkedVector(FILE* vector_file, const char* value_file_path, ParameterHandler& p, uint8_t abi_version);

    /**
     * @brief Appends one timestamp value to the logical stream.
     * @param val Timestamp value to append.
     * @returns Status reported by its manager.
     */
    AddStatus add(uint64_t val);

    /** @returns Human-readable representation of the logical timestamp stream. */
    [[nodiscard]] std::string to_string() const;
    /**
     * @brief Computes one per-subarray weight for the time window `[start, end]`.
     * @param start Window start timestamp.
     * @param end Window end timestamp.
     * @returns Ratio-based weights describing how much of each timestamp subarray overlaps the window.
     */
    [[nodiscard]] std::vector<double> getWeights(pallas_timestamp_t start, pallas_timestamp_t end) const;
    /**
     * @brief Finds the first logical occurrence that appears before `ts`.
     * @param ts Timestamp threshold.
     * @returns Logical index of the first occurrence preceding the threshold.
     */
    [[nodiscard]] size_t getFirstOccurrenceBefore(pallas_timestamp_t ts) const;

    /** Writes the timestamp-vector header to the metadata stream. */
    void write_header(FILE* info_file);
    /**
     * @brief Writes the timestamp vector to the metadata and value streams.
     * @param info_file Metadata output stream.
     * @param data_file Value output stream.
     * @param parameter_handler Runtime parameter handler used by subarray serialization.
     */
    void write_to_file(FILE* info_file, FILE* data_file, const ParameterHandler* parameter_handler);

   protected:
    /** Creates the next timestamp-domain subarray in the chain. */
    SubArrayBase* create_subarray(SubArrayBase* previous) const override;
};

/**
 * @brief Linked-vector specialization for duration-domain values.
 *
 * `DurationLinkedVector ` stores inclusive or exclusive duration streams. In addition to
 * the generic linked-vector mechanics inherited from `LinkedVectorBase`, it maintains
 * aggregate duration statistics at the vector level and creates
 * `DurationSubArray` instances for the physical storage layer.
 */
class DurationLinkedVector  : public LinkedVectorBase {
   public:
    /** @brief Creates a fresh duration vector using the default policy from the parameter handler. */
    explicit DurationLinkedVector(ParameterHandler& p);
    /**
     * @brief Creates a fresh duration vector with an explicit storage policy.
     * @param p Shared runtime parameter handler.
     * @param _policy Initial storage policy used for newly created duration subarrays.
     */
    explicit DurationLinkedVector(ParameterHandler& p, StoragePolicy _policy);
    /**
     * @brief Reconstructs a duration vector from persisted metadata.
     * @param vector_file Metadata stream containing the linked-vector header.
     * @param value_file_path Backing value file used for lazy payload reload.
     * @param p Shared runtime parameter handler.
     * @param abi_version Trace ABI version used to interpret persisted metadata.
     */
    DurationLinkedVector (FILE* vector_file, const char* value_file_path, ParameterHandler& p, uint8_t abi_version);

    /**
     * @brief Appends one duration value to the logical stream.
     * @param val Duration value to append.
     * @returns Status reported by the active tail subarray or its manager.
     */
    AddStatus add(uint64_t val);

    /**
     * @brief Finalizes the vector-level running mean after all values have been accumulated.
     */
    void final_update_mean();

    /**
     * @brief Computes a weighted sum of durations using one weight per subarray.
     * @param weights Weight vector aligned with the logical subarray order.
     * @returns Weighted duration sum across the vector.
     */
    [[nodiscard]] pallas_duration_t weightedSum(std::vector<double>& weights) const;
    /**
     * @brief Computes the duration sum between two logical indices.
     * @param start_index Inclusive starting logical index.
     * @param end_index Exclusive ending logical index.
     * @returns Sum of logical duration values in that range.
     */
    [[nodiscard]] pallas_duration_t computeDurationBetween(size_t start_index, size_t end_index) const;
    /** @returns Human-readable representation of the logical duration stream and its aggregate statistics. */
    [[nodiscard]] std::string to_string() const;

    /** @returns Minimum duration tracked at the linked-vector level. */
    [[nodiscard]] uint64_t min_value() const;
    /** @returns Maximum duration tracked at the linked-vector level. */
    [[nodiscard]] uint64_t max_value() const;
    /** @returns Mean duration tracked at the linked-vector level. */
    [[nodiscard]] uint64_t mean_value() const;

    /** Writes the duration-vector header to the metadata stream. */
    void write_header(FILE* info_file);
    /**
     * @brief Writes the duration vector to the metadata and value streams.
     * @param info_file Metadata output stream.
     * @param data_file Value output stream.
     * @param parameter_handler Runtime parameter handler used by subarray serialization.
     */
    void write_to_file(FILE* info_file, FILE* data_file, const ParameterHandler* parameter_handler);

   protected:
    /** Creates the next duration-domain subarray in the chain. */
    SubArrayBase* create_subarray(SubArrayBase* previous) const override;

    /** Minimum logical duration observed in this vector. */
    uint64_t min_duration = UINT64_MAX;
    /** Maximum logical duration observed in this vector. */
    uint64_t max_duration = 0;
    /** Running or finalized mean duration tracked at the vector level. */
    uint64_t mean_duration = 0;
    /** Tracks whether `mean_duration` currently stores a finalized mean or a running sum. */
    bool mean_duration_is_finalized = false;
};

}  // namespace pallas

#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
