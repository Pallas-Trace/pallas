/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** \file
 * A header file that defines various enums
 * and a class that stores all the different parameters used during the execution.
 */
#pragma once
#ifdef __cplusplus
#include <cstddef>
#include <queue>
#include <string>

#ifdef WITH_SZ
#undef SZ
#endif

namespace pallas {
/** A set of various compression algorithms supported by Pallas.*/
enum class CompressionAlgorithm {
  /** No Compression.*/
  None = 0,
  /** Compression using ZSTD (lossless). */
  ZSTD = 1,
  /**Compression using Histogram (lossy). */
  Histogram = 2,
#ifdef WITH_SZ
  /** Compression using SZ (lossy). */
  SZ = 3,
#endif
#ifdef WITH_ZFP
  /**Compression using ZFP (lossy). */
  ZFP = 4,
#endif
  ZSTD_Histogram = 5,
  Invalid
};

const enum CompressionAlgorithm CompressionAlgorithmDefault = CompressionAlgorithm::None;
const size_t zstdCompressionLevelDefault = 3;

/**
 * Converts a compression algorithm to its string name.
 * @param alg Algorithm to compress.
 * @return String such that it shall be parsed to that algorithm's enum.
 */
std::string toString(CompressionAlgorithm alg);

/**
 * Converts a string to a  compression algorithm.
 * @param str the string.
 * @return Compression Algorithm that corresponds to the string.
 */
CompressionAlgorithm compressionAlgorithmFromString(const std::string& str);

/** Returns whether a compression algorithm is lossy or not. */
inline bool isLossy(CompressionAlgorithm alg) {
  return alg != CompressionAlgorithm::None && alg != CompressionAlgorithm::ZSTD;
}

/** A set of various encoding algorithms supported by Pallas */
enum class EncodingAlgorithm {
  /** No encoding. */
  None,
  /** Masking encoding: the first byte of an array indicates the size of the rest of the elements in the array.
   * This is done to reduce the number of leading zeroes.*/
  Masking,
  /** LeadingZeroes encoding: the first byte of each element in the array indicates the size of that element.
   * This is done to nullify the number of leading zeroes */
  LeadingZeroes,
  Invalid
};

const enum EncodingAlgorithm EncodingAlgorithmDefault = EncodingAlgorithm::None;

/**
 * Converts an EncodingAlgorithm to its string name.
 * @param alg the EncodingAlgorithm.
 * @return String such that it shall be parsed to that algorithm's enum.
 */
std::string toString(EncodingAlgorithm alg);

/**
 * Converts a string to an EncodingAlgorithm.
 * @param str the string.
 * @return EncodingAlgorithm that corresponds to the string.
 */
EncodingAlgorithm encodingAlgorithmFromString(const std::string& str);

/** A set of various loop-finding algorithms used by Pallas */
enum class LoopFindingAlgorithm {
  /** No loop finding */
  None,
  /** Basic, quadratic loop finding algorithm. */
  Basic,
  /** Basic, quadratic loop finding algorithm.
   * The algorithm doesn't search for any pattern longer than ParameterHandler::maxLoopLength */
  BasicTruncated,
  Invalid
};
const enum LoopFindingAlgorithm LoopFindingAlgorithmDefault = LoopFindingAlgorithm::BasicTruncated;
const size_t maxLoopLengthDefault = 100;

/**
 * Converts a LoopFindingAlgorithm to its string name.
 * @param alg the LoopFindingAlgorithm.
 * @return String such that it shall be parsed to that algorithm's enum.
 */
std::string toString(LoopFindingAlgorithm alg);

/**
 * Converts a string to an LoopFindingAlgorithm.
 * @param str the string.
 * @return LoopFindingAlgorithm that corresponds to the string.
 */
LoopFindingAlgorithm loopFindingAlgorithmFromString(const std::string& str);

/** A set of various encoding algorithms supported by Pallas */
enum class TimestampStorage {
  /** Do not store timestamps. */
  None,
  /** Store event durations (default). */
  Delta,
  /** Store event timestamps (not implemented yet). */
  Timestamp,

  Invalid,
};
const enum TimestampStorage TimestampStorageDefault = TimestampStorage::Delta;

/**
 * Converts a TimestampStorage to its string name.
 * @param alg the TimestampStorage.
 * @return String such that it shall be parsed to that TimestampStorage's enum.
 */
std::string toString(TimestampStorage alg);

/**
 * Converts a string to an TimestampStorage.
 * @param str the string.
 * @return TimestampStorage that corresponds to the string.
 */
TimestampStorage timestampStorageFromString(const std::string& str);

/**
 * A simple data class that contains information on different parameters.
 */
class ParameterHandler {
   public:
    /** The compression algorithm used during the execution. */
    CompressionAlgorithm compressionAlgorithm{CompressionAlgorithmDefault};
    /** The ZSTD compression level. */
    size_t zstdCompressionLevel{zstdCompressionLevelDefault};
    /** The encoding algorithm used during the execution. */
    EncodingAlgorithm encodingAlgorithm{EncodingAlgorithmDefault};
    /** The compression algorithm used during the execution. */
    LoopFindingAlgorithm loopFindingAlgorithm{LoopFindingAlgorithmDefault};
    /** The max length the LoopFindingAlgorithm::BasicTruncated will go to.*/
    size_t maxLoopLength{maxLoopLengthDefault};

    /** Timestamp storage method. */
    TimestampStorage timestampStorage{TimestampStorageDefault};
    /** Amount of durations loaded in memory, in bytes. */
    size_t loaded_durations_size = 0;
    /** Max amount of memory taken by timestamps / durations. */
    size_t max_memory_durations = sizeof(size_t) * 1024 * 1024;
    /** Least recently loaded queue for the durations and timestamps subvectors. */
    std::deque<void*> subvector_queue;

   public:
    /** Getter for #maxLoopLength. Error if you're not supposed to have a maximum loop length.
     * @returns Value of #maxLoopLength. */
    [[nodiscard]] size_t getMaxLoopLength() const;
    /** Getter for #zstdCompressionLevel. Error if you're not using ZSTD.
     * @returns Value of #zstdCompressionLevel. */
    [[nodiscard]] u_int8_t getZstdCompressionLevel() const;
    /** Getter for #compressionAlgorithm.
     * @returns Value of #compressionAlgorithm. */
    [[nodiscard]] CompressionAlgorithm getCompressionAlgorithm() const;
    /**
     * Getter for #encodingAlgorithm.
     * @returns Value of #encodingAlgorithm. If the #compressionAlgorithm is lossy, returns EncodingAlgorithm::None. */
    [[nodiscard]] EncodingAlgorithm getEncodingAlgorithm() const;
    /**
     * Getter for #loopFindingAlgorithm.
     * @returns Value of #loopFindingAlgorithm.
     */
    [[nodiscard]] LoopFindingAlgorithm getLoopFindingAlgorithm() const;
    /** Creates a ParameterHandler from a config file loaded from PALLAS_CONFIG_PATH or pallas.config.
     */

    /**
     * Getter for #timestampStorage.
     * @returns Value of #timestampStorage. */
    [[nodiscard]] TimestampStorage getTimestampStorage() const;

    void writeToFile(FILE* file) const;
    void readFromFile(FILE* file);

    ParameterHandler();
    ParameterHandler(const std::string& stringConfig);
    ParameterHandler(FILE* file);

    /**
     * Prints the config of the ParameterHandler. That string is a valid Pallas configuration file.
     * @return String containing itself.
     */
    [[nodiscard]] std::string to_string() const;
};

}  // namespace pallas
#else
/** Struct for the CPP ParameterHandler class. Don't manipulate in C. */
typedef struct {} ParameterHandler;
#endif