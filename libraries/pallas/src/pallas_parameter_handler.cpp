/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_parameter_handler.h"

#include <pallas_config.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_log.h"

namespace pallas {

std::string loadStringFromEnv(const std::string& envName) {
  const char* env_value = getenv(envName.c_str());
  if (env_value)
    return {env_value};
  return "";
}

uint64_t loadUInt64FromEnv(const std::string& envName) {
  const char* env_value = getenv(envName.c_str());
  if (env_value) {
    try {
      return std::stoull(env_value);
    } catch (std::invalid_argument& e) {
      pallas_warn("Invalid UInt64 in config file: %s\n", env_value);
    }
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

CompressionAlgorithm compressionAlgorithmFromString(const std::string& str) {
  for (auto& [en, enStr] : CompressionAlgorithmMap) {
    if (enStr == str) {
      return en;
    }
  }
  return CompressionAlgorithm::Invalid;
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

EncodingAlgorithm encodingAlgorithmFromString(const std::string& str) {
  for (auto& [en, enStr] : EncodingAlgorithmMap) {
    if (enStr == str) {
      return en;
    }
  }
  return EncodingAlgorithm::Invalid;
}

std::map<LoopFindingAlgorithm, std::string> LoopFindingAlgorithmMap = {
  {LoopFindingAlgorithm::None, "None"},
  {LoopFindingAlgorithm::Basic, "Basic"},
  {LoopFindingAlgorithm::BasicTruncated, "BasicTruncated"},
  {LoopFindingAlgorithm::Invalid, "Invalid"},
};

std::string toString(LoopFindingAlgorithm alg) {
  return LoopFindingAlgorithmMap[alg];
}

LoopFindingAlgorithm loopFindingAlgorithmFromString(const std::string& str) {
  for (auto& [en, enStr] : LoopFindingAlgorithmMap) {
    if (enStr == str) {
      return en;
    }
  }
  return LoopFindingAlgorithm::Invalid;
}

std::map<TimestampStorage, std::string> TimestampStorageMap = {{TimestampStorage::None, "None"},
                                                               {TimestampStorage::Delta, "Delta"},
                                                               {TimestampStorage::Timestamp, "Timestamp"},
                                                               {TimestampStorage::Invalid, "Invalid"}};

std::string toString(TimestampStorage alg) {
  return TimestampStorageMap[alg];
}

TimestampStorage timestampStorageFromString(const std::string& str) {
  for (auto& [en, enStr] : TimestampStorageMap) {
    if (enStr == str) {
      return en;
    }
  }
  return TimestampStorage::Invalid;
}

class ConfigFile {
  std::map<std::string, std::string> config;

 public:
  std::string loadStringFromConfig(const std::string& fieldName) {
    if (config.find(fieldName) != config.end()) {
      return config[fieldName];
    }
    pallas_warn("Invalid key in config: %s\n", fieldName.c_str());
    return "";
  }

  uint64_t loadUInt64FromConfig(const std::string& fieldName) {
    if (config.find(fieldName) != config.end()) {
      try {
        return std::stoull(config[fieldName]);
      } catch (std::invalid_argument& e) {
        pallas_warn("Invalid UInt64 in config file: %s\n", config[fieldName].c_str());
      }
    }
    pallas_warn("Invalid key in config: %s\n", fieldName.c_str());
    return UINT64_MAX;
  }

  EncodingAlgorithm loadEncodingAlgorithmConfig() {
    EncodingAlgorithm ret = EncodingAlgorithm::None;

    std::string value = loadStringFromEnv("PALLAS_ENCODING");
    if (value.empty() && !config.empty()) {
      value = loadStringFromConfig("encodingAlgorithm");
    }
    if (!value.empty())
      ret = encodingAlgorithmFromString(value);
    return ret;
  }

  CompressionAlgorithm loadCompressionAlgorithmConfig() {
    CompressionAlgorithm ret = CompressionAlgorithm::None;

    std::string value = loadStringFromEnv("PALLAS_COMPRESSION");
    if (value.empty() && !config.empty()) {
      value = loadStringFromConfig("compressionAlgorithm");
    }
    if (!value.empty())
      ret = compressionAlgorithmFromString(value);

    return ret;
  }

  LoopFindingAlgorithm loadLoopFindingAlgorithmConfig() {
    LoopFindingAlgorithm ret = LoopFindingAlgorithm::BasicTruncated;

    std::string value = loadStringFromEnv("PALLAS_LOOP_FINDING");
    if (value.empty() && !config.empty()) {
      value = loadStringFromConfig("loopFindingAlgorithm");
    }
    if (!value.empty())
      ret = loopFindingAlgorithmFromString(value);
    return ret;
  }

  uint64_t loadMaxLoopLength() {
    uint64_t value = loadUInt64FromEnv("PALLAS_LOOP_LENGTH");
    if (value == UINT64_MAX && !config.empty()) {
      value = loadUInt64FromConfig("maxLoopLength");
    }

    if (value == UINT64_MAX) {
      return 100;
    }
    return value;
  }

  uint64_t loadZSTDCompressionLevel() {
    uint64_t value = loadUInt64FromEnv("PALLAS_ZSTD_LVL");
    if (value == UINT64_MAX && !config.empty()) {
      value = loadUInt64FromConfig("zstdCompressionLevel");
    }
    if (value == UINT64_MAX) {
      return 3;
    }
    return value;
  }

  TimestampStorage loadTimestampStorageConfig() {
    TimestampStorage ret = TimestampStorage::Delta;

    std::string value = loadStringFromEnv("PALLAS_TIMESTAMP_STORAGE");
    if (value.empty() && !config.empty()) {
      value = loadStringFromConfig("timestampStorageAlgorithm");
    }
    if (!value.empty())
      ret = timestampStorageFromString(value);
    return ret;
  }

  explicit ConfigFile(const std::string& configPath) {
    std::ifstream configFile(configPath);
    if (configFile.is_open()) {
      std::string line;
      while (getline(configFile, line)) {
        auto separator = line.find('=');
        auto key = line.substr(0, separator);
        auto value = line.substr(separator + 1, line.length());
        config[key] = value;
      }
    }
  }
};

const char* defaultConfigFile = PALLAS_CONFIG_PATH;
ParameterHandler* parameterHandler = nullptr;
ParameterHandler::ParameterHandler(const std::string& stringConfig) {
  ConfigFile config(stringConfig);
  compressionAlgorithm = config.loadCompressionAlgorithmConfig();
  encodingAlgorithm = config.loadEncodingAlgorithmConfig();
  loopFindingAlgorithm = config.loadLoopFindingAlgorithmConfig();
  maxLoopLength = config.loadMaxLoopLength();
  zstdCompressionLevel = config.loadZSTDCompressionLevel();
  timestampStorage = config.loadTimestampStorageConfig();

  pallas_log(DebugLevel::Normal, "%s\n", to_string().c_str());
}

ParameterHandler::ParameterHandler() {
  std::string configPath;
  bool useDefault = false;
  if (const char* givenConfigFile = getenv("PALLAS_CONFIG_PATH"); givenConfigFile) {
    pallas_log(DebugLevel::Debug, "Loading configuration file from %s\n", givenConfigFile);

    std::ifstream configFile(givenConfigFile);
    if (!configFile.good()) {
      pallas_warn("Provided config file didn't exist, or couldn't be read: %s.\n", givenConfigFile);
      useDefault = true;
    }
    configPath = givenConfigFile;
  } else {
    useDefault = true;
  }
  if (useDefault) {
    pallas_log(DebugLevel::Debug, "No config file provided, using default: %s\n", defaultConfigFile);
    std::ifstream configFile(defaultConfigFile);
    if (!configFile.good()) {
      pallas_warn("No config file found at default install path ! Check your installation.\n");
      return;
    }
    configPath = defaultConfigFile;
  }

  ConfigFile config(configPath);
  compressionAlgorithm = config.loadCompressionAlgorithmConfig();
  encodingAlgorithm = config.loadEncodingAlgorithmConfig();
  loopFindingAlgorithm = config.loadLoopFindingAlgorithmConfig();
  maxLoopLength = config.loadMaxLoopLength();
  zstdCompressionLevel = config.loadZSTDCompressionLevel();
  timestampStorage = config.loadTimestampStorageConfig();

  pallas_log(DebugLevel::Debug, "%s\n", to_string().c_str());
}

size_t ParameterHandler::getMaxLoopLength() const {
  if (loopFindingAlgorithm == LoopFindingAlgorithm::BasicTruncated)
    return maxLoopLength;
  pallas_error("Asked for the max loop length but wasn't using a LoopFindingBasicTruncated algorithm.\n");
}
u_int8_t ParameterHandler::getZstdCompressionLevel() const {
  return zstdCompressionLevel;
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
  stream << "compressionAlgorithm=" << toString(compressionAlgorithm) << "\n";
  stream << "encodingAlgorithm=" << toString(encodingAlgorithm) << "\n";
  stream << "loopFindingAlgorithm=" << toString(loopFindingAlgorithm) << "\n";
  stream << "maxLoopLength=" << maxLoopLength << "\n";
  stream << "zstdCompressionLevel=" << zstdCompressionLevel << "\n";
  stream << "timestampStorage=" << toString(timestampStorage) << "\n";
  return stream.str();
}

}  // namespace pallas
