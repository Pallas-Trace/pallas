/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "pallas_config.h"

#include "pallas/utils/pallas_dbg.h"
#include "pallas/utils/pallas_log.h"
#include "pallas/utils/pallas_parameter_handler.h"
#include "pallas/utils/pallas_subarray.h"

namespace pallas {

ParameterHandler::~ParameterHandler() {
  loaded_durations_size = 0;
  subvector_queue.clear();
}

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

bool loadBoolFromEnv(const std::string& envName, bool default_value) {
  const char* env_value = getenv(envName.c_str());
  if (env_value == nullptr) {
    return default_value;
  }

  const std::string value(env_value);
  if (value == "1" || value == "true" || value == "TRUE" || value == "True") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" || value == "False") {
    return false;
  }

  pallas_warn("Invalid boolean in config/env: %s\n", env_value);
  return default_value;
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

std::map<StoragePolicy, std::string> StoragePolicyMap = {
    {StoragePolicy::None, "None"},
    {StoragePolicy::Delta, "Delta"},
    {StoragePolicy::Lossy, "Lossy"},
};

std::string toString(StoragePolicy policy) {
  return StoragePolicyMap[policy];
}

StoragePolicy storagePolicyFromString(const std::string& str) {
  for (auto& [en, enStr] : StoragePolicyMap) {
    if (enStr == str) {
      return en;
    }
  }
  return static_cast<StoragePolicy>(UINT8_MAX);
}

std::map<LossyPolicy, std::string> LossyPolicyMap = {
    {LossyPolicy::PLA4, "PLA4"},
    {LossyPolicy::PLA8, "PLA8"},
    {LossyPolicy::PLA16, "PLA16"},
    {LossyPolicy::PLA32, "PLA32"},
    {LossyPolicy::Spike4, "Spike4"},
    {LossyPolicy::Spike8, "Spike8"},
    {LossyPolicy::Spike16, "Spike16"},
    {LossyPolicy::Spike32, "Spike32"},
};

std::string toString(LossyPolicy policy) {
  return LossyPolicyMap[policy];
}

LossyPolicy lossyPolicyFromString(const std::string& str) {
  for (auto& [en, enStr] : LossyPolicyMap) {
    if (enStr == str) {
      return en;
    }
  }
  return static_cast<LossyPolicy>(UINT8_MAX);
}

/** Simple class to handle the parsing of the configuration file. */
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

  StoragePolicy loadStoragePolicyConfig() {
    StoragePolicy ret = StoragePolicy::None;

    std::string value = loadStringFromEnv("PALLAS_STORAGE_POLICY");
    if (value.empty() && !config.empty() && config.find("storagePolicy") != config.end()) {
      value = config["storagePolicy"];
    }
    if (!value.empty()) {
      ret = storagePolicyFromString(value);
      if (ret == static_cast<StoragePolicy>(UINT8_MAX)) {
        pallas_warn("Invalid StoragePolicy in config: %s\n", value.c_str());
        ret = StoragePolicy::None;
      }
    }
    return ret;
  }

  LossyPolicy loadTimeLossyPolicyConfig() {
    LossyPolicy ret = LossyPolicy::PLA8;

    std::string value = loadStringFromEnv("PALLAS_TIME_LOSSY_POLICY");
    if (value.empty() && !config.empty() && config.find("timeLossyPolicy") != config.end()) {
      value = config["timeLossyPolicy"];
    }
    if (!value.empty()) {
      ret = lossyPolicyFromString(value);
      if (ret == static_cast<LossyPolicy>(UINT8_MAX)) {
        pallas_warn("Invalid TimeLossyPolicy in config: %s\n", value.c_str());
        ret = LossyPolicy::PLA8;
      }
    }
    return ret;
  }

  LossyPolicy loadDurationLossyPolicyConfig() {
    LossyPolicy ret = LossyPolicy::Spike8;

    std::string value = loadStringFromEnv("PALLAS_DURATION_LOSSY_POLICY");
    if (value.empty() && !config.empty() && config.find("durationLossyPolicy") != config.end()) {
      value = config["durationLossyPolicy"];
    }
    if (!value.empty()) {
      ret = lossyPolicyFromString(value);
      if (ret == static_cast<LossyPolicy>(UINT8_MAX)) {
        pallas_warn("Invalid DurationLossyPolicy in config: %s\n", value.c_str());
        ret = LossyPolicy::Spike8;
      }
    }
    return ret;
  }

  bool loadOverrideLoopDetectionConfig() {
    bool value = loadBoolFromEnv("PALLAS_OVERRIDE_LOOP_DETECTION", false);
    if (!value && !config.empty() && config.find("overrideLoopDetection") != config.end()) {
      const auto& config_value = config["overrideLoopDetection"];
      if (config_value == "1" || config_value == "true" || config_value == "TRUE" || config_value == "True") {
        value = true;
      } else if (config_value == "0" || config_value == "false" || config_value == "FALSE" || config_value == "False") {
        value = false;
      } else {
        pallas_warn("Invalid overrideLoopDetection in config: %s\n", config_value.c_str());
      }
    }
    return value;
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

ParameterHandler::ParameterHandler(const std::string& stringConfig) {
  ConfigFile config(stringConfig);
  compressionAlgorithm = config.loadCompressionAlgorithmConfig();
  encodingAlgorithm = config.loadEncodingAlgorithmConfig();
  loopFindingAlgorithm = config.loadLoopFindingAlgorithmConfig();
  maxLoopLength = config.loadMaxLoopLength();
  zstdCompressionLevel = config.loadZSTDCompressionLevel();
  timestampStorage = config.loadTimestampStorageConfig();
  storagePolicy = config.loadStoragePolicyConfig();
  timeLossyPolicy = config.loadTimeLossyPolicyConfig();
  durationLossyPolicy = config.loadDurationLossyPolicyConfig();
  overrideLoopDetection = config.loadOverrideLoopDetectionConfig();

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
  storagePolicy = config.loadStoragePolicyConfig();
  timeLossyPolicy = config.loadTimeLossyPolicyConfig();
  durationLossyPolicy = config.loadDurationLossyPolicyConfig();
  overrideLoopDetection = config.loadOverrideLoopDetectionConfig();

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

bool ParameterHandler::shouldOverrideLoopDetection() const {
  return overrideLoopDetection;
}

StoragePolicy ParameterHandler::getStoragePolicy() const {
  return storagePolicy;
}

LossyPolicy ParameterHandler::getTimeLossyPolicy() const {
  return timeLossyPolicy;
}

LossyPolicy ParameterHandler::getDurationLossyPolicy() const {
  return durationLossyPolicy;
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
  stream << "overrideLoopDetection=" << (overrideLoopDetection ? "true" : "false") << "\n";
  stream << "zstdCompressionLevel=" << zstdCompressionLevel << "\n";
  stream << "timestampStorageAlgorithm=" << toString(timestampStorage) << "\n";
  stream << "storagePolicy=" << toString(storagePolicy) << "\n";
  stream << "timeLossyPolicy=" << toString(timeLossyPolicy) << "\n";
  stream << "durationLossyPolicy=" << toString(durationLossyPolicy) << "\n";
  return stream.str();
}

}  // namespace pallas
