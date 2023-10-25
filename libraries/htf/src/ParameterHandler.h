//
// Created by khatharsis on 24/10/23.
//
#pragma once
/** \file
 * A header file that defines various enums
 * and a class that stores all the different parameters used during the execution.
 */
#include <cstddef>
#include <string>
namespace htf {
/** A set of various compression algorithms supported by HTF.*/
enum class CompressionAlgorithm {
  /** No Compression.*/
  None,
  /** Compression using ZSTD (lossless). */
  ZSTD,
  /** Compression using SZ (lossy). */
  SZ,
  /**Compression using ZFP (lossy). */
  ZFP
};
/** Returns whether a compression algorithm is lossy or not. */
inline bool isLossy(CompressionAlgorithm alg) {
  return alg == CompressionAlgorithm::SZ || alg == CompressionAlgorithm::ZFP;
}
/** A set of various encoding algorithms supported by HTF */
enum class EncodingAlgorithm {
  /** No encoding. */
  None,
  /** Masking encoding: the first byte of an array indicates the size of the rest of the elements in the array.
   * This is done to reduce the number of leading zeroes.*/
  Masking,
  /** LeadingZeroes encoding: the first byte of each element in the array indicates the size of that element.
   * This is done to nullify the number of leading zeroes */
  LeadingZeroes
};

/** A set of various loop-finding algorithms used by HTF */
enum class LoopFindingAlgorithm {
  /** No loop finding */
  None,
  /** Basic, quadratic loop finding algorithm. */
  Basic,
  /** Basic, quadratic loop finding algorithm.
   * The algorithm doesn't search for any pattern longer than ParameterHandler::maxLoopLength */
  BasicTruncated
};

/**
 * @brief A simple data class that contains information on different parameters.
 */
class ParameterHandler {
  /** The compression algorithm used during the execution. */
  CompressionAlgorithm compressionAlgorithm{CompressionAlgorithm::None};
  /** The ZSTD compression level. */
  size_t zstdCompressionLevel{1};
  /** The encoding algorithm used during the execution. */
  EncodingAlgorithm encodingAlgorithm{EncodingAlgorithm::None};
  /** The compression algorithm used during the execution. */
  LoopFindingAlgorithm loopFindingAlgorithm{LoopFindingAlgorithm::BasicTruncated};
  /** The max length the LoopFindingAlgorithm::BasicTruncated will go to.*/
  size_t maxLoopLength{100};

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
  /**
   * Creates a ParameterHandler from a config file.
   * @param configFileName Path to a valid JSON config file.
   */
  explicit ParameterHandler(const std::string& configFileName);
  /**
   * Default constructor.
   */
  ParameterHandler() = default;
};

/** Global ParameterHandler. This is supposed to be the only instance of that class. */
extern const ParameterHandler parameterHandler;
}  // namespace htf
