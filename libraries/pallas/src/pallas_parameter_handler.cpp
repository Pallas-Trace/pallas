/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_parameter_handler.h"

#include <json/json.h>
#include <json/value.h>
#include <pallas_config.h>
#include <fstream>
#include <iostream>
#include "pallas/pallas_dbg.h"


namespace pallas {
std::string loadStringFromConfig(Json::Value& config, std::string fieldName) {
  if (config[fieldName]) {
    if (config[fieldName].isString()) {
      return config[fieldName].asString();
    }
  }
  return "";
}

uint64_t loadUInt64FromConfig(Json::Value& config, std::string fieldName) {
  if (config[fieldName]) {
    if (config[fieldName].isUInt64()) {
      return config[fieldName].asUInt64();
    }
  }
  return UINT64_MAX;
}

std::string loadStringFromEnv(std::string envName) {
  const char* env_value = getenv(envName.c_str());
  if (env_value)
    return std::string(env_value);
  return "";
}

uint64_t loadUInt64FromEnv(std::string envName) {
  const char* env_value = getenv(envName.c_str());
  if (env_value) {
    return std::stoull(env_value);
  }
  return UINT64_MAX;
}

std::map<CompressionAlgorithm, std::string> CompressionAlgorithmMap = {
  {CompressionAlgorithm::None, "None"},
  {CompressionAlgorithm::ZSTD, "ZSTD"},
  {CompressionAlgorithm::Histogram, "Histogram"},
  {CompressionAlgorithm::ZSTD_Histogram, "ZSTD_Histogram"},
#ifdef WITH_SZ
  {CompressionAlgorithm::SZ, "SZ"},
#endif
#ifdef WITH_ZFP
  {CompressionAlgorithm::ZFP, "ZFP"},
#endif
  {CompressionAlgorithm::Invalid, "Invalid"},
};

std::string toString(CompressionAlgorithm alg) {
  return CompressionAlgorithmMap[alg];
}

CompressionAlgorithm compressionAlgorithmFromString(std::string str) {
  for (auto& it : CompressionAlgorithmMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return CompressionAlgorithm::Invalid;
}

CompressionAlgorithm loadCompressionAlgorithmConfig(Json::Value& config) {
  CompressionAlgorithm ret = CompressionAlgorithm::None;

  std::string value = loadStringFromEnv("PALLAS_COMPRESSION");
  if (value.empty() && ! config.empty()) {
    value = loadStringFromConfig(config, "compressionAlgorithm");
  }
  if (!value.empty())
    ret = compressionAlgorithmFromString(value);

  return ret;
}

std::map<EncodingAlgorithm, std::string> EncodingAlgorithmMap = {
  {EncodingAlgorithm::None, "None"},
  {EncodingAlgorithm::Masking, "Masking"},
  {EncodingAlgorithm::LeadingZeroes, "LeadingZeroes"},
  {EncodingAlgorithm::Invalid, "Invalid"},
};

std::string toString(EncodingAlgorithm alg) {
  return EncodingAlgorithmMap[alg];
}

EncodingAlgorithm encodingAlgorithmFromString(std::string str) {
  for (auto& it : EncodingAlgorithmMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return EncodingAlgorithm::Invalid;
}

EncodingAlgorithm loadEncodingAlgorithmConfig(Json::Value& config) {
  EncodingAlgorithm ret = EncodingAlgorithm::None;

  std::string value = loadStringFromEnv("PALLAS_ENCODING");
  if (value.empty() && ! config.empty()) {
    value = loadStringFromConfig(config, "encodingAlgorithm");
  }
  if (!value.empty())
    ret = encodingAlgorithmFromString(value);
  return ret;
}

std::map<LoopFindingAlgorithm, std::string> LoopFindingAlgorithmMap = {
  {LoopFindingAlgorithm::None, "None"},
  {LoopFindingAlgorithm::Basic, "Basic"},
  {LoopFindingAlgorithm::BasicTruncated, "BasicTruncated"},
  {LoopFindingAlgorithm::Filter, "Filter"},
  {LoopFindingAlgorithm::Invalid, "Invalid"},
};

std::string toString(LoopFindingAlgorithm alg) {
  return LoopFindingAlgorithmMap[alg];
}

LoopFindingAlgorithm loopFindingAlgorithmFromString(std::string str) {
  for (auto& it : LoopFindingAlgorithmMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return LoopFindingAlgorithm::Invalid;
}

LoopFindingAlgorithm loadLoopFindingAlgorithmConfig(Json::Value& config) {
  LoopFindingAlgorithm ret = LoopFindingAlgorithm::BasicTruncated;

  std::string value = loadStringFromEnv("PALLAS_LOOP_FINDING");
  if (value.empty() && ! config.empty()) {
    value = loadStringFromConfig(config, "loopFindingAlgorithm");
  }
  if (!value.empty())
    ret = loopFindingAlgorithmFromString(value);
  return ret;
}

uint64_t loadMaxLoopLength(Json::Value& config) {

  uint64_t value = loadUInt64FromEnv("PALLAS_LOOP_LENGTH");
  if (value == UINT64_MAX && ! config.empty()) {
    value = loadUInt64FromConfig(config, "maxLoopLength");
  }

  if (value == UINT64_MAX) {
    return 100;
  }
  return value;
}

uint64_t loadZSTDCompressionLevel(Json::Value& config) {

  uint64_t value = loadUInt64FromEnv("PALLAS_ZSTD_LVL");
  if (value == UINT64_MAX && ! config.empty()) {
    value = loadUInt64FromConfig(config, "zstdCompressionLevel");
  }
  if (value == UINT64_MAX) {
    return 3;
  }
  return value;
}

std::map<TimestampStorage, std::string> TimestampStorageMap = {
  {TimestampStorage::None, "None"},
  {TimestampStorage::Delta, "Delta"},
  {TimestampStorage::Timestamp, "Timestamp"},
  {TimestampStorage::Invalid, "Invalid"}};

std::string toString(TimestampStorage alg) {
  return TimestampStorageMap[alg];
}

TimestampStorage timestampStorageFromString(std::string str) {
  for (auto& it : TimestampStorageMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return TimestampStorage::Invalid;
}

TimestampStorage loadTimestampStorageConfig(Json::Value& config) {
  TimestampStorage ret = TimestampStorage::Delta;

  std::string value = loadStringFromEnv("PALLAS_TIMESTAMP_STORAGE");
  if (value.empty() && ! config.empty()) {
    value = loadStringFromConfig(config, "timestampStorage");
  }
  if (!value.empty())
    ret = timestampStorageFromString(value);
  return ret;
}

const char* defaultConfigFile = PALLAS_CONFIG_PATH;
ParameterHandler* parameterHandler = nullptr;
ParameterHandler::ParameterHandler(const std::string &stringConfig) {
  Json::Reader reader;
  Json::Value config;
  reader.parse(stringConfig, config);

  compressionAlgorithm = loadCompressionAlgorithmConfig(config);
  encodingAlgorithm = loadEncodingAlgorithmConfig(config);
  loopFindingAlgorithm = loadLoopFindingAlgorithmConfig(config);
  maxLoopLength = loadMaxLoopLength(config);
  zstdCompressionLevel = loadZSTDCompressionLevel(config);
  timestampStorage = loadTimestampStorageConfig(config);

  pallas_log(DebugLevel::Normal, "%s\n", to_string().c_str());
}

ParameterHandler::ParameterHandler() {
  std::ifstream configFile;
  if (const char* givenConfigFile = getenv("PALLAS_CONFIG_PATH"); givenConfigFile) {
    pallas_log(DebugLevel::Debug, "Loading configuration file from %s\n", givenConfigFile);
    configFile.open(givenConfigFile);
    if (!configFile.good()) {
      pallas_warn("Provided config file didn't exist, or couldn't be read: %s.\n", givenConfigFile);
      goto elseJump;
    }
  } else {
    elseJump:
    pallas_log(DebugLevel::Debug, "No config file provided, using default: %s\n", defaultConfigFile);
    configFile.open(defaultConfigFile);
    if (!configFile.good()) {
      pallas_warn("No config file found at default install path ! Check your installation.\n");
      return;
    }
  }

  Json::Value config;
  if (configFile.good()) {
    configFile >> config;
  }
  configFile.close();
  /* Load from file */

  compressionAlgorithm = loadCompressionAlgorithmConfig(config);
  encodingAlgorithm = loadEncodingAlgorithmConfig(config);
  loopFindingAlgorithm = loadLoopFindingAlgorithmConfig(config);
  maxLoopLength = loadMaxLoopLength(config);
  zstdCompressionLevel = loadZSTDCompressionLevel(config);
  timestampStorage = loadTimestampStorageConfig(config);

  pallas_log(DebugLevel::Debug, "%s\n", to_string().c_str());
}

size_t ParameterHandler::getMaxLoopLength() const {
  if (loopFindingAlgorithm == LoopFindingAlgorithm::BasicTruncated)
    return maxLoopLength;
  pallas_error("Asked for the max loop length but wasn't using a LoopFindingBasicTruncated algorithm.\n");
}
u_int8_t ParameterHandler::getZstdCompressionLevel() const {
  if (compressionAlgorithm == CompressionAlgorithm::ZSTD || compressionAlgorithm == CompressionAlgorithm::ZSTD_Histogram) {
    return zstdCompressionLevel;
  }
  pallas_error("Asked for ZSTD Compression Level but wasn't using a CompressionZSTD algorithm.\n");
}
CompressionAlgorithm ParameterHandler::getCompressionAlgorithm() const {
  return compressionAlgorithm;
}
EncodingAlgorithm ParameterHandler::getEncodingAlgorithm() const {
  if (isLossy(compressionAlgorithm) && encodingAlgorithm != EncodingAlgorithm::None) {
    pallas_warn("Encoding algorithm isn't None even though the compression algorithm is lossy.\n");
    return EncodingAlgorithm::None;
  }
  return encodingAlgorithm;
}
LoopFindingAlgorithm ParameterHandler::getLoopFindingAlgorithm() const {
  return loopFindingAlgorithm;
}

TimestampStorage ParameterHandler::getTimestampStorage() const {
  return timestampStorage;
}

std::string ParameterHandler::to_string() const {
  std::stringstream stream("");
  stream << "{\n";
  stream << '\t' << R"("compressionAlgorithm": ")" << toString(compressionAlgorithm) << "\",\n";
  stream << '\t' << R"("encodingAlgorithm": ")" << toString(encodingAlgorithm) << "\",\n";
  stream << '\t' << R"("loopFindingAlgorithm": ")" << toString(loopFindingAlgorithm) << "\",\n";
  stream << '\t' << R"("maxLoopLength": )" << maxLoopLength << ",\n";
  stream << '\t' << R"("zstdCompressionLevel": )" << zstdCompressionLevel << ",\n";
  stream << '\t' << R"("timestampStorage": ")" << toString(timestampStorage) << "\",\n";
  stream << "}";
  return stream.str();
}

}  // namespace pallas
