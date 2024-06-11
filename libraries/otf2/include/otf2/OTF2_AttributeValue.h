/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2014,
 * Technische Universitaet Dresden, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef OTF2_ATTRIBUTE_VALUE_H
#define OTF2_ATTRIBUTE_VALUE_H

/**
 *  @file
 *  @source     template/OTF2_AttributeValue.tmpl.h
 *
 *  @brief      Declares the @eref{OTF2_AttributeValue} and provides convenience
 *              functions to convert from and to OTF2 enum values.
 *
 */

#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_attribute.h"

#include <otf2/OTF2_ErrorCodes.h>

#include <otf2/OTF2_Definitions.h>
#include <otf2/OTF2_Events.h>
#include <otf2/OTF2_GeneralDefinitions.h>

/** @brief Value container for an attributes.
 *
 * For definition references (@eref{OTF2_MappingType}) use the same data type as
 * the definition.
 */
typedef union OTF2_AttributeValue_union {
  /** @brief Arbitrary value of type uint8_t */
  uint8_t uint8;
  /** @brief Arbitrary value of type uint16_t */
  uint16_t uint16;
  /** @brief Arbitrary value of type uint32_t */
  uint32_t uint32;
  /** @brief Arbitrary value of type uint64_t */
  uint64_t uint64;
  /** @brief Arbitrary value of type int8_t */
  int8_t int8;
  /** @brief Arbitrary value of type int16_t */
  int16_t int16;
  /** @brief Arbitrary value of type int32_t */
  int32_t int32;
  /** @brief Arbitrary value of type int64_t */
  int64_t int64;
  /** @brief Arbitrary value of type float */
  float float32;
  /** @brief Arbitrary value of type double */
  double float64;
  /**
   * @brief References a @eref{String} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_STRING} is available.
   */
  OTF2_StringRef stringRef;
  /**
   * @brief References a @eref{Attribute} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_ATTRIBUTE} is available.
   */
  OTF2_AttributeRef attributeRef;
  /**
   * @brief References a @eref{Location} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_LOCATION} is available.
   */
  OTF2_LocationRef locationRef;
  /**
   * @brief References a @eref{Region} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_REGION} is available.
   */
  OTF2_RegionRef regionRef;
  /**
   * @brief References a @eref{Group} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_GROUP} is available.
   */
  OTF2_GroupRef groupRef;
  /**
   * @brief References a @eref{MetricClass}, or a @eref{MetricInstance} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_METRIC} is available.
   */
  OTF2_MetricRef metricRef;
  /**
   * @brief References a @eref{Comm}, or a @eref{InterComm} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_COMM} is available.
   */
  OTF2_CommRef commRef;
  /**
   * @brief References a @eref{Parameter} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_PARAMETER} is available.
   */
  OTF2_ParameterRef parameterRef;
  /**
   * @brief References a @eref{RmaWin} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_RMA_WIN} is available.
   */
  OTF2_RmaWinRef rmaWinRef;
  /**
   * @brief References a @eref{SourceCodeLocation} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_SOURCE_CODE_LOCATION} is available.
   */
  OTF2_SourceCodeLocationRef sourceCodeLocationRef;
  /**
   * @brief References a @eref{CallingContext} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_CALLING_CONTEXT} is available.
   */
  OTF2_CallingContextRef callingContextRef;
  /**
   * @brief References a @eref{InterruptGenerator} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_INTERRUPT_GENERATOR} is available.
   */
  OTF2_InterruptGeneratorRef interruptGeneratorRef;
  /**
   * @brief References a @eref{IoRegularFile}, or a @eref{IoDirectory} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_IO_FILE} is available.
   */
  OTF2_IoFileRef ioFileRef;
  /**
   * @brief References a @eref{IoHandle} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_IO_HANDLE} is available.
   */
  OTF2_IoHandleRef ioHandleRef;
  /**
   * @brief References a @eref{LocationGroup} definition
   * and will be mapped to the global definition if a mapping table of
   * type @eref{OTF2_MAPPING_LOCATION_GROUP} is available.
   */
  OTF2_LocationGroupRef locationGroupRef;
} OTF2_AttributeValue;


  static PALLAS(pallas_type_t) OTF2_PALLAS_TYPE(OTF2_Type type) __attribute__((unused));
  static PALLAS(pallas_type_t) OTF2_PALLAS_TYPE(OTF2_Type type) {
  switch (type) {
  case PALLAS(PALLAS_TYPE_NONE): return OTF2_TYPE_NONE;
  case PALLAS(PALLAS_TYPE_UINT8): return OTF2_TYPE_UINT8;
  case PALLAS(PALLAS_TYPE_UINT16): return OTF2_TYPE_UINT16;
  case PALLAS(PALLAS_TYPE_UINT32): return OTF2_TYPE_UINT32;
  case PALLAS(PALLAS_TYPE_UINT64): return OTF2_TYPE_UINT64;
  case PALLAS(PALLAS_TYPE_INT8): return OTF2_TYPE_INT8;
  case PALLAS(PALLAS_TYPE_INT16): return OTF2_TYPE_INT16;
  case PALLAS(PALLAS_TYPE_INT32): return OTF2_TYPE_INT32;
  case PALLAS(PALLAS_TYPE_INT64): return OTF2_TYPE_INT64;
  case PALLAS(PALLAS_TYPE_FLOAT): return OTF2_TYPE_FLOAT;
  case PALLAS(PALLAS_TYPE_DOUBLE): return OTF2_TYPE_DOUBLE;
  case PALLAS(PALLAS_TYPE_STRING): return OTF2_TYPE_STRING;
  case PALLAS(PALLAS_TYPE_ATTRIBUTE): return OTF2_TYPE_ATTRIBUTE;
  case PALLAS(PALLAS_TYPE_LOCATION): return OTF2_TYPE_LOCATION;
  case PALLAS(PALLAS_TYPE_REGION): return OTF2_TYPE_REGION;
  case PALLAS(PALLAS_TYPE_GROUP): return OTF2_TYPE_GROUP;
  case PALLAS(PALLAS_TYPE_METRIC): return OTF2_TYPE_METRIC;
  case PALLAS(PALLAS_TYPE_COMM): return OTF2_TYPE_COMM;
  case PALLAS(PALLAS_TYPE_PARAMETER): return OTF2_TYPE_PARAMETER;
  case PALLAS(PALLAS_TYPE_RMA_WIN): return OTF2_TYPE_RMA_WIN;
  case PALLAS(PALLAS_TYPE_SOURCE_CODE_LOCATION): return OTF2_TYPE_SOURCE_CODE_LOCATION;
  case PALLAS(PALLAS_TYPE_CALLING_CONTEXT): return OTF2_TYPE_CALLING_CONTEXT;
  case PALLAS(PALLAS_TYPE_INTERRUPT_GENERATOR): return OTF2_TYPE_INTERRUPT_GENERATOR;
  case PALLAS(PALLAS_TYPE_IO_FILE): return OTF2_TYPE_IO_FILE;
  case PALLAS(PALLAS_TYPE_IO_HANDLE): return OTF2_TYPE_IO_HANDLE;
  case PALLAS(PALLAS_TYPE_LOCATION_GROUP): return OTF2_TYPE_LOCATION_GROUP;
  }
  return OTF2_TYPE_NONE;
}

  static PALLAS(AttributeValue) OTF2_PALLAS_ATTRIBUTE_VALUE(OTF2_AttributeValue u, PALLAS(pallas_type_t) t) __attribute__((unused));
  static PALLAS(AttributeValue) OTF2_PALLAS_ATTRIBUTE_VALUE(OTF2_AttributeValue u, PALLAS(pallas_type_t) t) {
  PALLAS(AttributeValue) v;
  switch(t) {
  case PALLAS(PALLAS_TYPE_NONE): v.uint8 = 0; break;
  case PALLAS(PALLAS_TYPE_UINT8): v.uint8 = u.uint8; break;
  case PALLAS(PALLAS_TYPE_UINT16): v.uint16 = u.uint16; break;
  case PALLAS(PALLAS_TYPE_UINT32): v.uint32 = u.uint32; break;
  case PALLAS(PALLAS_TYPE_UINT64): v.uint64 = u.uint64; break;
  case PALLAS(PALLAS_TYPE_INT8): v.int8 = u.int8; break;
  case PALLAS(PALLAS_TYPE_INT16): v.int16 = u.int16; break;
  case PALLAS(PALLAS_TYPE_INT32): v.int32 = u.int32; break;
  case PALLAS(PALLAS_TYPE_INT64): v.int64 = u.int64; break;
  case PALLAS(PALLAS_TYPE_FLOAT): v.float32 = u.float32; break;
  case PALLAS(PALLAS_TYPE_DOUBLE): v.float64 = u.float64; break;
  case PALLAS(PALLAS_TYPE_STRING): v.string_ref = u.stringRef; break;
  case PALLAS(PALLAS_TYPE_ATTRIBUTE): v.attribute_ref = u.attributeRef; break;
  case PALLAS(PALLAS_TYPE_LOCATION): v.location_ref = u.locationRef; break;
  case PALLAS(PALLAS_TYPE_REGION): v.region_ref = u.regionRef; break;
  case PALLAS(PALLAS_TYPE_GROUP): v.group_ref = u.groupRef; break;
  case PALLAS(PALLAS_TYPE_METRIC): v.metric_ref = u.metricRef; break;
  case PALLAS(PALLAS_TYPE_COMM): v.comm_ref = u.commRef; break;
  case PALLAS(PALLAS_TYPE_PARAMETER): v.parameter_ref = u.parameterRef; break;
  case PALLAS(PALLAS_TYPE_RMA_WIN): v.rma_win_ref = u.rmaWinRef; break;
  case PALLAS(PALLAS_TYPE_SOURCE_CODE_LOCATION): v.source_code_location_ref = u.sourceCodeLocationRef; break;
  case PALLAS(PALLAS_TYPE_CALLING_CONTEXT): v.calling_context_ref = u.callingContextRef; break;
  case PALLAS(PALLAS_TYPE_INTERRUPT_GENERATOR): v.interrupt_generator_ref = u.interruptGeneratorRef; break;
  case PALLAS(PALLAS_TYPE_IO_FILE): v.io_file_ref = u.ioFileRef; break;
  case PALLAS(PALLAS_TYPE_IO_HANDLE): v.io_handle_ref = u.ioHandleRef; break;
  case PALLAS(PALLAS_TYPE_LOCATION_GROUP): v.location_group_ref = u.locationGroupRef; break;
  }
  return v;
}

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_Boolean}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetBoolean(OTF2_Boolean enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_FileType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetFileType(OTF2_FileType enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_FileSubstrate}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetFileSubstrate(OTF2_FileSubstrate enumValue,
                                                    OTF2_Type* type,
                                                    OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MappingType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMappingType(OTF2_MappingType enumValue,
                                                  OTF2_Type* type,
                                                  OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_Type}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetType(OTF2_Type enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_Paradigm}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetParadigm(OTF2_Paradigm enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_ParadigmClass}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetParadigmClass(OTF2_ParadigmClass enumValue,
                                                    OTF2_Type* type,
                                                    OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_ParadigmProperty}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetParadigmProperty(OTF2_ParadigmProperty enumValue,
                                                       OTF2_Type* type,
                                                       OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_ThumbnailType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetThumbnailType(OTF2_ThumbnailType enumValue,
                                                    OTF2_Type* type,
                                                    OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_SystemTreeDomain}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetSystemTreeDomain(OTF2_SystemTreeDomain enumValue,
                                                       OTF2_Type* type,
                                                       OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_LocationGroupType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetLocationGroupType(OTF2_LocationGroupType enumValue,
                                                        OTF2_Type* type,
                                                        OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_LocationType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetLocationType(OTF2_LocationType enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RegionRole}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRegionRole(OTF2_RegionRole enumValue,
                                                 OTF2_Type* type,
                                                 OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RegionFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRegionFlag(OTF2_RegionFlag enumValue,
                                                 OTF2_Type* type,
                                                 OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_GroupType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetGroupType(OTF2_GroupType enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_GroupFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetGroupFlag(OTF2_GroupFlag enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_Base}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetBase(OTF2_Base enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MetricOccurrence}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMetricOccurrence(OTF2_MetricOccurrence enumValue,
                                                       OTF2_Type* type,
                                                       OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MetricType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMetricType(OTF2_MetricType enumValue,
                                                 OTF2_Type* type,
                                                 OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MetricValueProperty}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMetricValueProperty(OTF2_MetricValueProperty enumValue,
                                                          OTF2_Type* type,
                                                          OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MetricTiming}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMetricTiming(OTF2_MetricTiming enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MetricMode}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMetricMode(OTF2_MetricMode enumValue,
                                                 OTF2_Type* type,
                                                 OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MetricScope}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMetricScope(OTF2_MetricScope enumValue,
                                                  OTF2_Type* type,
                                                  OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RecorderKind}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRecorderKind(OTF2_RecorderKind enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_ParameterType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetParameterType(OTF2_ParameterType enumValue,
                                                    OTF2_Type* type,
                                                    OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_CartPeriodicity}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetCartPeriodicity(OTF2_CartPeriodicity enumValue,
                                                      OTF2_Type* type,
                                                      OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_InterruptGeneratorMode}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetInterruptGeneratorMode(OTF2_InterruptGeneratorMode enumValue,
                                                             OTF2_Type* type,
                                                             OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_MeasurementMode}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetMeasurementMode(OTF2_MeasurementMode enumValue,
                                                      OTF2_Type* type,
                                                      OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_CollectiveOp}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetCollectiveOp(OTF2_CollectiveOp enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RmaSyncType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRmaSyncType(OTF2_RmaSyncType enumValue,
                                                  OTF2_Type* type,
                                                  OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RmaSyncLevel}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRmaSyncLevel(OTF2_RmaSyncLevel enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_LockType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetLockType(OTF2_LockType enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RmaAtomicType}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRmaAtomicType(OTF2_RmaAtomicType enumValue,
                                                    OTF2_Type* type,
                                                    OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoParadigmClass}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoParadigmClass(OTF2_IoParadigmClass enumValue,
                                                      OTF2_Type* type,
                                                      OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoParadigmFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoParadigmFlag(OTF2_IoParadigmFlag enumValue,
                                                     OTF2_Type* type,
                                                     OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoParadigmProperty}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoParadigmProperty(OTF2_IoParadigmProperty enumValue,
                                                         OTF2_Type* type,
                                                         OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoHandleFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoHandleFlag(OTF2_IoHandleFlag enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoAccessMode}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoAccessMode(OTF2_IoAccessMode enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoCreationFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoCreationFlag(OTF2_IoCreationFlag enumValue,
                                                     OTF2_Type* type,
                                                     OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoStatusFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoStatusFlag(OTF2_IoStatusFlag enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoSeekOption}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoSeekOption(OTF2_IoSeekOption enumValue,
                                                   OTF2_Type* type,
                                                   OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoOperationMode}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoOperationMode(OTF2_IoOperationMode enumValue,
                                                      OTF2_Type* type,
                                                      OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_IoOperationFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetIoOperationFlag(OTF2_IoOperationFlag enumValue,
                                                      OTF2_Type* type,
                                                      OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_CommFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetCommFlag(OTF2_CommFlag enumValue, OTF2_Type* type, OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_RmaWinFlag}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetRmaWinFlag(OTF2_RmaWinFlag enumValue,
                                                 OTF2_Type* type,
                                                 OTF2_AttributeValue* value);

/** @brief Set @eref{OTF2_Type} and @eref{OTF2_AttributeValue} to the appropriate
 *  values for the given enum entry.  No value range checking done.
 *
 *  @param      enumValue The enum value to be converted.
 *  @param[out] type      Matching @eref{OTF2_Type} for the enum
 *                        @eref{OTF2_CollectiveRoot}.
 *  @param[out] value     Matching @eref{OTF2_AttributeValue} for the
 *                        @a enumValue value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if either @a type or @a value are NULL pointer}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_SetCollectiveRoot(OTF2_CollectiveRoot enumValue,
                                                     OTF2_Type* type,
                                                     OTF2_AttributeValue* value);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_Boolean}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_Boolean}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetBoolean(OTF2_Type type, OTF2_AttributeValue value, OTF2_Boolean* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_FileType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_FileType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetFileType(OTF2_Type type, OTF2_AttributeValue value, OTF2_FileType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_FileSubstrate}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_FileSubstrate}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetFileSubstrate(OTF2_Type type,
                                                    OTF2_AttributeValue value,
                                                    OTF2_FileSubstrate* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MappingType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MappingType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMappingType(OTF2_Type type,
                                                  OTF2_AttributeValue value,
                                                  OTF2_MappingType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_Type}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_Type}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetType(OTF2_Type type, OTF2_AttributeValue value, OTF2_Type* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_Paradigm}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_Paradigm}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetParadigm(OTF2_Type type, OTF2_AttributeValue value, OTF2_Paradigm* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_ParadigmClass}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_ParadigmClass}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetParadigmClass(OTF2_Type type,
                                                    OTF2_AttributeValue value,
                                                    OTF2_ParadigmClass* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_ParadigmProperty}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_ParadigmProperty}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetParadigmProperty(OTF2_Type type,
                                                       OTF2_AttributeValue value,
                                                       OTF2_ParadigmProperty* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_ThumbnailType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_ThumbnailType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetThumbnailType(OTF2_Type type,
                                                    OTF2_AttributeValue value,
                                                    OTF2_ThumbnailType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_SystemTreeDomain}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_SystemTreeDomain}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetSystemTreeDomain(OTF2_Type type,
                                                       OTF2_AttributeValue value,
                                                       OTF2_SystemTreeDomain* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_LocationGroupType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_LocationGroupType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetLocationGroupType(OTF2_Type type,
                                                        OTF2_AttributeValue value,
                                                        OTF2_LocationGroupType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_LocationType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_LocationType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetLocationType(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_LocationType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RegionRole}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RegionRole}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRegionRole(OTF2_Type type, OTF2_AttributeValue value, OTF2_RegionRole* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RegionFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RegionFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRegionFlag(OTF2_Type type, OTF2_AttributeValue value, OTF2_RegionFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_GroupType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_GroupType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetGroupType(OTF2_Type type, OTF2_AttributeValue value, OTF2_GroupType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_GroupFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_GroupFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetGroupFlag(OTF2_Type type, OTF2_AttributeValue value, OTF2_GroupFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_Base}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_Base}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetBase(OTF2_Type type, OTF2_AttributeValue value, OTF2_Base* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MetricOccurrence}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MetricOccurrence}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMetricOccurrence(OTF2_Type type,
                                                       OTF2_AttributeValue value,
                                                       OTF2_MetricOccurrence* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MetricType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MetricType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMetricType(OTF2_Type type, OTF2_AttributeValue value, OTF2_MetricType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MetricValueProperty}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MetricValueProperty}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMetricValueProperty(OTF2_Type type,
                                                          OTF2_AttributeValue value,
                                                          OTF2_MetricValueProperty* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MetricTiming}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MetricTiming}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMetricTiming(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_MetricTiming* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MetricMode}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MetricMode}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMetricMode(OTF2_Type type, OTF2_AttributeValue value, OTF2_MetricMode* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MetricScope}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MetricScope}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMetricScope(OTF2_Type type,
                                                  OTF2_AttributeValue value,
                                                  OTF2_MetricScope* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RecorderKind}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RecorderKind}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRecorderKind(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_RecorderKind* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_ParameterType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_ParameterType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetParameterType(OTF2_Type type,
                                                    OTF2_AttributeValue value,
                                                    OTF2_ParameterType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_CartPeriodicity}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_CartPeriodicity}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetCartPeriodicity(OTF2_Type type,
                                                      OTF2_AttributeValue value,
                                                      OTF2_CartPeriodicity* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_InterruptGeneratorMode}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_InterruptGeneratorMode}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetInterruptGeneratorMode(OTF2_Type type,
                                                             OTF2_AttributeValue value,
                                                             OTF2_InterruptGeneratorMode* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_MeasurementMode}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_MeasurementMode}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetMeasurementMode(OTF2_Type type,
                                                      OTF2_AttributeValue value,
                                                      OTF2_MeasurementMode* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_CollectiveOp}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_CollectiveOp}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetCollectiveOp(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_CollectiveOp* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RmaSyncType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RmaSyncType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRmaSyncType(OTF2_Type type,
                                                  OTF2_AttributeValue value,
                                                  OTF2_RmaSyncType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RmaSyncLevel}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RmaSyncLevel}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRmaSyncLevel(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_RmaSyncLevel* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_LockType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_LockType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetLockType(OTF2_Type type, OTF2_AttributeValue value, OTF2_LockType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RmaAtomicType}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RmaAtomicType}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRmaAtomicType(OTF2_Type type,
                                                    OTF2_AttributeValue value,
                                                    OTF2_RmaAtomicType* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoParadigmClass}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoParadigmClass}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoParadigmClass(OTF2_Type type,
                                                      OTF2_AttributeValue value,
                                                      OTF2_IoParadigmClass* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoParadigmFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoParadigmFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoParadigmFlag(OTF2_Type type,
                                                     OTF2_AttributeValue value,
                                                     OTF2_IoParadigmFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoParadigmProperty}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoParadigmProperty}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoParadigmProperty(OTF2_Type type,
                                                         OTF2_AttributeValue value,
                                                         OTF2_IoParadigmProperty* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoHandleFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoHandleFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoHandleFlag(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_IoHandleFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoAccessMode}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoAccessMode}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoAccessMode(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_IoAccessMode* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoCreationFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoCreationFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoCreationFlag(OTF2_Type type,
                                                     OTF2_AttributeValue value,
                                                     OTF2_IoCreationFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoStatusFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoStatusFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoStatusFlag(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_IoStatusFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoSeekOption}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoSeekOption}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoSeekOption(OTF2_Type type,
                                                   OTF2_AttributeValue value,
                                                   OTF2_IoSeekOption* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoOperationMode}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoOperationMode}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoOperationMode(OTF2_Type type,
                                                      OTF2_AttributeValue value,
                                                      OTF2_IoOperationMode* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_IoOperationFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_IoOperationFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetIoOperationFlag(OTF2_Type type,
                                                      OTF2_AttributeValue value,
                                                      OTF2_IoOperationFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_CommFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_CommFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetCommFlag(OTF2_Type type, OTF2_AttributeValue value, OTF2_CommFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_RmaWinFlag}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_RmaWinFlag}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetRmaWinFlag(OTF2_Type type, OTF2_AttributeValue value, OTF2_RmaWinFlag* enumValue);

/** @brief Converts a @eref{OTF2_Type} and @eref{OTF2_AttributeValue} pair to
 *  the appropriate value for the enum @eref{OTF2_CollectiveRoot}.  No value range
 *  checking done.
 *
 *  @param      type      Given type.
 *  @param      value     Given value.
 *  @param[out] enumValue Converted enum value.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             if @a enumValue a NULL pointer}
 *    @retcode{OTF2_ERROR_INVALID_ATTRIBUTE_TYPE,
 *             if @a type does not match the base type of the enum @eref{OTF2_CollectiveRoot}}
 *  @retend
 */
OTF2_ErrorCode OTF2_AttributeValue_GetCollectiveRoot(OTF2_Type type,
                                                     OTF2_AttributeValue value,
                                                     OTF2_CollectiveRoot* enumValue);

#endif /* !OTF2_ATTRIBUTE_VALUE_H */
