//
// Created by khatharsis on 24/10/23.
//

#include "ParameterHandler.h"
#include <json/json.h>
#include <json/value.h>
#include <fstream>
#include <iostream>
#include "htf/htf_dbg.h"

/** Given a parameter and an enumValue, will assign the parameter's value to enumValue if the config file says so.*/
#define MATCH_ENUM(parameterName, enumValue)  \
  if (config[#parameterName] == #enumValue) { \
    parameterName = enumValue;                \
    goto parameterName##LoadingIfStatement;   \
  }
/** Small macro to load a field that is supposed to be an enum, have some code that does the matching,
 * and then throw a warning if nothing has been matched. */
#define LOAD_FIELD_ENUM(parameterName, loadingCode)                                                              \
  if (config[#parameterName]) {                                                                                  \
    if (config[#parameterName].isString()) {                                                                     \
      loadingCode;                                                                                               \
    }                                                                                                            \
    htf_warn("Parameter in \"" #parameterName "\" field was invalid: %s\n", config[#parameterName].asCString()); \
    parameterName##LoadingIfStatement :;                                                                         \
  }
/** Small macro to load a field that's supposed to be a UINT64, and throw a warning in case something goes wrong. */
#define LOAD_FIELD_UINT64(parameterName)                                                                           \
  if (config[#parameterName]) {                                                                                    \
    if (config[#parameterName].isUInt64()) {                                                                       \
      parameterName = config[#parameterName].asUInt64();                                                           \
    } else {                                                                                                       \
      htf_warn("Parameter in \"" #parameterName "\" field was invalid: %s\n", config[#parameterName].asCString()); \
    }                                                                                                              \
  }

/** Small macro that is used when getting parameters from environment variables.*/
#define GET_ENV_FIELD(parameterName, enumName, enumSpecific) \
  if (parameterName##String == #enumSpecific)                \
  parameterName##Algorithm = enumName##enumSpecific

namespace htf {
// TODO Find a way to deal with that because it's sure as hell not a good idea.
const std::string defaultPath = "/home/khatharsis/Documents/Stage/Hierarchical-Trace-Format/config.json";
const ParameterHandler parameterHandler = ParameterHandler(defaultPath);

ParameterHandler::ParameterHandler(const std::string& configFileName) {
  htf_log(Debug, "Loading configuration file from %s\n", configFileName.data());
  std::ifstream configFile(configFileName, std::ifstream::binary);
  if (!configFile.good()) {
    htf_warn("Given config file didn't exist: %s.\n", configFileName.data());
    return;
  }
  Json::Value config;
  configFile >> config;
  configFile.close();
  /* Load from file */
  LOAD_FIELD_ENUM(compressionAlgorithm, {
    MATCH_ENUM(compressionAlgorithm, CompressionNone);
    MATCH_ENUM(compressionAlgorithm, CompressionZSTD);
    MATCH_ENUM(compressionAlgorithm, CompressionZFP);
    MATCH_ENUM(compressionAlgorithm, CompressionSZ);
  });

  LOAD_FIELD_ENUM(encodingAlgorithm, {
    MATCH_ENUM(encodingAlgorithm, EncodingNone);
    MATCH_ENUM(encodingAlgorithm, EncodingMasking);
    MATCH_ENUM(encodingAlgorithm, EncodingLeadingZeroes);
  });
  LOAD_FIELD_ENUM(loopFindingAlgorithm, {
    MATCH_ENUM(loopFindingAlgorithm, LoopFindingNone);
    MATCH_ENUM(loopFindingAlgorithm, LoopFindingBasic);
    MATCH_ENUM(loopFindingAlgorithm, LoopFindingBasicTruncated);
  });
  LOAD_FIELD_UINT64(maxLoopLength);
  LOAD_FIELD_UINT64(zstdCompressionLevel);

  /* Override from Environment Variables */

#define GET_COMP_FIELD(value) GET_ENV_FIELD(compression, Compression, value)
  char* compressionChar = std::getenv("COMPRESSION");
  if (compressionChar) {
    std::string compressionString = compressionChar;
    GET_COMP_FIELD(None);
    GET_COMP_FIELD(ZSTD);
    GET_COMP_FIELD(SZ);
    GET_COMP_FIELD(ZFP);
  }

#define GET_ENCO_FIELD(value) GET_ENV_FIELD(encoding, Encoding, value)
  char* encodingChar = std::getenv("ENCODING");
  if (encodingChar) {
    std::string encodingString = encodingChar;
    GET_ENCO_FIELD(None);
    GET_ENCO_FIELD(Masking);
    GET_ENCO_FIELD(LeadingZeroes);
  }

#define GET_LOOP_FIELD(value) GET_ENV_FIELD(loopFinding, LoopFinding, value)
  char* loopFindingChar = std::getenv("LOOP_FINDING");
  if (loopFindingChar) {
    std::string loopFindingString = loopFindingChar;
    GET_LOOP_FIELD(None);
    GET_LOOP_FIELD(Basic);
    GET_LOOP_FIELD(BasicTruncated);
  }
}

size_t ParameterHandler::getMaxLoopLength() const {
  if (loopFindingAlgorithm == LoopFindingBasicTruncated)
    return maxLoopLength;
  htf_error("Asked for the max loop length but wasn't using a LoopFindingBasicTruncated algorithm.\n");
}
u_int8_t ParameterHandler::getZstdCompressionLevel() const {
  if (compressionAlgorithm == CompressionZSTD) {
    return zstdCompressionLevel;
  }
  htf_error("Asked for ZSTD Compression Level but wasn't using a CompressionZSTD algorithm.\n");
}
CompressionAlgorithm ParameterHandler::getCompressionAlgorithm() const {
  return compressionAlgorithm;
}
EncodingAlgorithm ParameterHandler::getEncodingAlgorithm() const {
  if (isLossy(compressionAlgorithm)) {
    htf_warn("Encoding algorithm isn't None even though the compression algorithm is lossy.\n");
    return EncodingNone;
  }
  return encodingAlgorithm;
}
LoopFindingAlgorithm ParameterHandler::getLoopFindingAlgorithm() const {
  return loopFindingAlgorithm;
}

}  // namespace htf