/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2013,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2013,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2014, 2021,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2013,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2013,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2013,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef OTF2_GENERAL_DEFINITIONS_H
#define OTF2_GENERAL_DEFINITIONS_H

/**
 *  @file
 *  @source     templates/OTF2_GeneralDefinitions.tmpl.h
 *
 *  @brief      This header file provides general definitions which should be
 *              accessible in all internal and external modules.
 */

#include <stdint.h>
#include <limits.h>

#include "otf2_compiler.h"

#include <otf2/OTF2_ErrorCodes.h>

#include "pallas/pallas.h"

/* ___ OTF2 library version information _____________________________________ */

/** @name OTF2 library version.
 *  @{ */
/** @brief Major version number of this OTF2 version. */
#define OTF2_VERSION_MAJOR 3
/** @brief Minor version number of this OTF2 version. */
#define OTF2_VERSION_MINOR 0
/** @brief Bugfix version number of this OTF2 version. */
#define OTF2_VERSION_BUGFIX 0
/** @brief Any string suffix of this OTF2 version. */
#define OTF2_VERSION_SUFFIX ""
/** @brief The OTF2 version as string. */
#define OTF2_VERSION "3.0"
/** @} */

/* ___ Symbols for undefined values _________________________________________ */

/** @name Standard undefined values for basic data types.
 *  @{ */
/** @brief Undefined value for type @a uint8_t. */
#define OTF2_UNDEFINED_UINT8 ((uint8_t)(~((uint8_t)0u)))
/** @brief Undefined value for type @a int8_t. */
#define OTF2_UNDEFINED_INT8 ((int8_t)(~(OTF2_UNDEFINED_UINT8 >> 1)))
/** @brief Undefined value for type @a uint16_t. */
#define OTF2_UNDEFINED_UINT16 ((uint16_t)(~((uint16_t)0u)))
/** @brief Undefined value for type @a int16_t. */
#define OTF2_UNDEFINED_INT16 ((int16_t)(~(OTF2_UNDEFINED_UINT16 >> 1)))
/** @brief Undefined value for type @a uint32_t. */
#define OTF2_UNDEFINED_UINT32 ((uint32_t)(~((uint32_t)0u)))
/** @brief Undefined value for type @a int32_t. */
#define OTF2_UNDEFINED_INT32 ((int32_t)(~(OTF2_UNDEFINED_UINT32 >> 1)))
/** @brief Undefined value for type @a uint64_t. */
#define OTF2_UNDEFINED_UINT64 ((uint64_t)(~((uint64_t)0u)))
/** @brief Undefined value for type @a int64_t. */
#define OTF2_UNDEFINED_INT64 ((int64_t)(~(OTF2_UNDEFINED_UINT64 >> 1)))
/** @brief Undefined value for type @a int. */
#define OTF2_UNDEFINED_INT ((int)(INT_MAX))
/** @} */

/** @brief Undefined value for enums */
#define OTF2_UNDEFINED_TYPE OTF2_UNDEFINED_UINT8

/** @brief OTF2 time stamp. */
typedef pallas_timestamp_t OTF2_TimeStamp;
/** @brief Undefined value for @eref{OTF2_TimeStamp}. */
#define OTF2_UNDEFINED_TIMESTAMP PALLAS_TIMESTAMP_INVALID

/** @brief Type used to indicate a reference to a @eref{IoParadigm} definition */
typedef uint8_t OTF2_IoParadigmRef;
/** @brief The invalid value for a reference to a @eref{IoParadigm} definition */
#define OTF2_UNDEFINED_IO_PARADIGM ((OTF2_IoParadigmRef)OTF2_UNDEFINED_UINT8)

/** @brief Type used to indicate a reference to a @eref{String} definition */
typedef PALLAS(StringRef) OTF2_StringRef;
/** @brief The invalid value for a reference to a @eref{String} definition */
#define OTF2_UNDEFINED_STRING ((OTF2_StringRef)PALLAS_STRING_REF_INVALID)

/** @brief Type used to indicate a reference to a @eref{Attribute} definition */
typedef uint32_t OTF2_AttributeRef;
/** @brief The invalid value for a reference to a @eref{Attribute} definition */
#define OTF2_UNDEFINED_ATTRIBUTE ((OTF2_AttributeRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{SystemTreeNode} definition */
typedef uint32_t OTF2_SystemTreeNodeRef;
/** @brief The invalid value for a reference to a @eref{SystemTreeNode} definition */
#define OTF2_UNDEFINED_SYSTEM_TREE_NODE ((OTF2_SystemTreeNodeRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{LocationGroup} definition */
typedef uint32_t OTF2_LocationGroupRef;
/** @brief The invalid value for a reference to a @eref{LocationGroup} definition */
#define OTF2_UNDEFINED_LOCATION_GROUP ((OTF2_LocationGroupRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{Location} definition */
typedef uint64_t OTF2_LocationRef;
/** @brief The invalid value for a reference to a @eref{Location} definition */
#define OTF2_UNDEFINED_LOCATION ((OTF2_LocationRef)OTF2_UNDEFINED_UINT64)

/** @brief Type used to indicate a reference to a @eref{Region} definition */
typedef PALLAS(RegionRef) OTF2_RegionRef;
/** @brief The invalid value for a reference to a @eref{Region} definition */
#define OTF2_UNDEFINED_REGION ((OTF2_RegionRef)PALLAS_REGION_REF_INVALID)

/** @brief Type used to indicate a reference to a @eref{Callsite} definition */
typedef uint32_t OTF2_CallsiteRef;
/** @brief The invalid value for a reference to a @eref{Callsite} definition */
#define OTF2_UNDEFINED_CALLSITE ((OTF2_CallsiteRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{Callpath} definition */
typedef uint32_t OTF2_CallpathRef;
/** @brief The invalid value for a reference to a @eref{Callpath} definition */
#define OTF2_UNDEFINED_CALLPATH ((OTF2_CallpathRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{Group} definition */
typedef uint32_t OTF2_GroupRef;
/** @brief The invalid value for a reference to a @eref{Group} definition */
#define OTF2_UNDEFINED_GROUP ((OTF2_GroupRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{MetricMember} definition */
typedef uint32_t OTF2_MetricMemberRef;
/** @brief The invalid value for a reference to a @eref{MetricMember} definition */
#define OTF2_UNDEFINED_METRIC_MEMBER ((OTF2_MetricMemberRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{MetricClass}, or a @eref{MetricInstance} definition */
typedef uint32_t OTF2_MetricRef;
/** @brief The invalid value for a reference to a @eref{MetricClass}, or a @eref{MetricInstance} definition */
#define OTF2_UNDEFINED_METRIC ((OTF2_MetricRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{Comm}, or a @eref{InterComm} definition */
typedef uint32_t OTF2_CommRef;
/** @brief The invalid value for a reference to a @eref{Comm}, or a @eref{InterComm} definition */
#define OTF2_UNDEFINED_COMM ((OTF2_CommRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{Parameter} definition */
typedef uint32_t OTF2_ParameterRef;
/** @brief The invalid value for a reference to a @eref{Parameter} definition */
#define OTF2_UNDEFINED_PARAMETER ((OTF2_ParameterRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{RmaWin} definition */
typedef uint32_t OTF2_RmaWinRef;
/** @brief The invalid value for a reference to a @eref{RmaWin} definition */
#define OTF2_UNDEFINED_RMA_WIN ((OTF2_RmaWinRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{CartDimension} definition */
typedef uint32_t OTF2_CartDimensionRef;
/** @brief The invalid value for a reference to a @eref{CartDimension} definition */
#define OTF2_UNDEFINED_CART_DIMENSION ((OTF2_CartDimensionRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{CartTopology} definition */
typedef uint32_t OTF2_CartTopologyRef;
/** @brief The invalid value for a reference to a @eref{CartTopology} definition */
#define OTF2_UNDEFINED_CART_TOPOLOGY ((OTF2_CartTopologyRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{SourceCodeLocation} definition */
typedef uint32_t OTF2_SourceCodeLocationRef;
/** @brief The invalid value for a reference to a @eref{SourceCodeLocation} definition */
#define OTF2_UNDEFINED_SOURCE_CODE_LOCATION ((OTF2_SourceCodeLocationRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{CallingContext} definition */
typedef uint32_t OTF2_CallingContextRef;
/** @brief The invalid value for a reference to a @eref{CallingContext} definition */
#define OTF2_UNDEFINED_CALLING_CONTEXT ((OTF2_CallingContextRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{InterruptGenerator} definition */
typedef uint32_t OTF2_InterruptGeneratorRef;
/** @brief The invalid value for a reference to a @eref{InterruptGenerator} definition */
#define OTF2_UNDEFINED_INTERRUPT_GENERATOR ((OTF2_InterruptGeneratorRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{IoRegularFile}, or a @eref{IoDirectory} definition */
typedef uint32_t OTF2_IoFileRef;
/** @brief The invalid value for a reference to a @eref{IoRegularFile}, or a @eref{IoDirectory} definition */
#define OTF2_UNDEFINED_IO_FILE ((OTF2_IoFileRef)OTF2_UNDEFINED_UINT32)

/** @brief Type used to indicate a reference to a @eref{IoHandle} definition */
typedef uint32_t OTF2_IoHandleRef;
/** @brief The invalid value for a reference to a @eref{IoHandle} definition */
#define OTF2_UNDEFINED_IO_HANDLE ((OTF2_IoHandleRef)OTF2_UNDEFINED_UINT32)

/* ___ Declarations of opaque OTF2 objects __________________________________ */

/** @brief OTF2 local event reader handle. */
typedef struct OTF2_EvtReader_struct OTF2_EvtReader;

/** @brief OTF2 global event reader handle. */
typedef struct OTF2_GlobalEvtReader_struct OTF2_GlobalEvtReader;

/** @brief OTF2 local definition reader handle. */
typedef struct OTF2_DefReader_struct OTF2_DefReader;

/** @brief OTF2 global definition reader handle. */
typedef struct OTF2_GlobalDefReader_struct OTF2_GlobalDefReader;

/** @brief OTF2 local snap reader handle. */
typedef struct OTF2_SnapReader_struct OTF2_SnapReader;

/** @brief OTF2 global snap reader handle. */
typedef struct OTF2_GlobalSnapReader_struct OTF2_GlobalSnapReader;

/** @brief OTF2 marker reader handle. */
typedef struct OTF2_MarkerReader_struct OTF2_MarkerReader;

/* ___ Definitions related to the file interaction __________________________ */

/** @brief Defines the minimum size of a chunk.*/
#define OTF2_CHUNK_SIZE_MIN (uint64_t)(256 * 1024)

/** @brief Defines the maximum size of a chunk. */
#define OTF2_CHUNK_SIZE_MAX (uint64_t)(1024 * 1024 * 16)

/** @brief Defines how to interact with files.
 *  Please see @eref{OTF2_FileMode_enum} for a detailed description. */
typedef uint8_t OTF2_FileMode;

/** @brief Defines how to interact with files. */
enum OTF2_FileMode_enum {
  /** Open a file in write-only mode. */
  OTF2_FILEMODE_WRITE = 0,
  /** Open a file in read-only mode. */
  OTF2_FILEMODE_READ = 1,
  /** Open a file in write-read mode. */
  OTF2_FILEMODE_MODIFY = 2
};

/** @brief Defines which compression is used.
 *  Please see @eref{OTF2_Compression_enum} for a detailed description. */
typedef uint8_t OTF2_Compression;

/** @brief Defines which compression is used. */
enum OTF2_Compression_enum {
  /** @brief Undefined. */
  OTF2_COMPRESSION_UNDEFINED = 0,
  /** @brief No compression is used. */
  OTF2_COMPRESSION_NONE = 1,
  /** @brief Use zlib compression.
   *  @deprecated In version 3.0
   */
  OTF2_COMPRESSION_ZLIB __otf2_deprecated__ = 2
};

/** @brief Return value to indicate that the record reading should be
 *         interrupted.
 *
 * Returning @eref{OTF2_CALLBACK_INTERRUPT} will stop reading more events, if functions
 * like:
 *
 * - @eref{OTF2_Reader_ReadLocalEvents}
 * - @eref{OTF2_Reader_ReadAllLocalEvents}
 * - @eref{OTF2_Reader_ReadLocalEventsBackward}
 * - @eref{OTF2_Reader_ReadGlobalEvents}
 * - @eref{OTF2_Reader_ReadAllGlobalEvents}
 * - @eref{OTF2_Reader_ReadLocalDefinitions}
 * - @eref{OTF2_Reader_ReadAllLocalDefinitions}
 * - @eref{OTF2_Reader_ReadGlobalDefinitions}
 * - @eref{OTF2_Reader_ReadAllGlobalDefinitions}
 *
 * where called. The return value for these functions is
 * @eref{OTF2_ERROR_INTERRUPTED_BY_CALLBACK} in this case. It is valid
 * to call any reader functions in such a condition again.
 *
 * This type is also used as return type in the collective and locking callbacks
 * (see @ref callbacks_collective and @ref callbacks_locking). Any value
 * different than @eref{OTF2_CALLBACK_SUCCESS} is treated as an error and the
 * calling function will return @eref{OTF2_ERROR_COLLECTIVE_CALLBACK} or
 * @eref{OTF2_ERROR_LOCKING_CALLBACK} to its caller, respectively. As the name
 * @eref{OTF2_CALLBACK_INTERRUPT} does not really fit in this context, the alias
 * @eref{OTF2_CALLBACK_ERROR} is provided for these callbacks.
 */
typedef enum OTF2_CallbackCode_enum {
  /** @brief Record reading can continue. */
  OTF2_CALLBACK_SUCCESS = 0,

  /** @brief Interrupt record reading.
   *
   *  Control returns to the caller of the read function
   *  with error OTF2_ERROR_INTERRUPTED_BY_CALLBACK to signal this.
   *  The actual value can be any except OTF2_CALLBACK_SUCCESS.
   */
  OTF2_CALLBACK_INTERRUPT = !OTF2_CALLBACK_SUCCESS,

  /** @brief Signaling an error in the callback. */
  OTF2_CALLBACK_ERROR = !OTF2_CALLBACK_SUCCESS
} OTF2_CallbackCode;

/* ___ Definitions related to the memory buffer ______________________________ */

/** @brief Defines whether the recorded data is flushed to a file or not. Please
    see OTF2_FlushType_enum for a detailed description. */
typedef uint8_t OTF2_FlushType;

/** @brief Defines whether the recorded data is flushed to a file or not. */
enum OTF2_FlushType_enum {
  /** @brief Flushing will be suppressed when running out of memory. */
  OTF2_NO_FLUSH = 0,
  /** @brief Recorded data is flushed when running out of memory. */
  OTF2_FLUSH = 1
};

/** @brief Wrapper for enum @eref{OTF2_Hint_enum}. */
typedef uint8_t OTF2_Hint;

/**
 * @brief List of possible hints.
 *
 * @since Version 1.5
 */
enum OTF2_Hint_enum {
  /** Hint the reader that the user will use the global reader to read
   *  per-location data (e.g., event and snapshot data).
   *
   *  In case of the SIONlib substrate that means the SION handles of the
   *  per-location local reader are not duplicated and thus not thread safe.
   *
   *  Datatype @eref{OTF2_Boolean} with default value @eref{OTF2_FALSE}.
   *
   *  This is for an @eref{OTF2_Archive} only valid if the file mode equals to
   *  @eref{OTF2_FILEMODE_READ}.
   *
   *  The hint will be locked when opening any of the per-location data files.
   */
  OTF2_HINT_GLOBAL_READER = 0
};

/** @brief Wrapper for enum @eref{OTF2_Boolean_enum}. */
typedef uint8_t OTF2_Boolean;

/**
 * @brief A boolean.
 *
 * @since Version 1.5
 */
enum OTF2_Boolean_enum {
  /** @brief False.
   */
  OTF2_FALSE = 0,

  /** @brief True.
   */
  OTF2_TRUE = !OTF2_FALSE
};

/** @brief Wrapper for enum @eref{OTF2_FileType_enum}. */
typedef uint8_t OTF2_FileType;

/**
 * @brief Defines which file type is used.
 *
 * @since Version 1.0
 */
enum OTF2_FileType_enum {
  /** @brief Represents the type for the anchor file (.otf2).
   */
  OTF2_FILETYPE_ANCHOR = 0,

  /** @brief Represents the type for the global definition file (.def).
   */
  OTF2_FILETYPE_GLOBAL_DEFS = 1,

  /** @brief Represents the type for a local definition file (.def).
   */
  OTF2_FILETYPE_LOCAL_DEFS = 2,

  /** @brief Represents the type for a event file (.evt).
   */
  OTF2_FILETYPE_EVENTS = 3,

  /** @brief Represents the type for a snapshot file (.snap).
   */
  OTF2_FILETYPE_SNAPSHOTS = 4,

  /** @brief Represents the type for a thumb file (.thumb).
   */
  OTF2_FILETYPE_THUMBNAIL = 5,

  /** @brief Represents the type for a marker file (.marker).
   */
  OTF2_FILETYPE_MARKER = 6,

  /** @brief Internal file which holds the SION rank map (.srm).
   */
  OTF2_FILETYPE_SIONRANKMAP = 7
};

/** @brief Wrapper for enum @eref{OTF2_FileSubstrate_enum}. */
typedef uint8_t OTF2_FileSubstrate;

/**
 * @brief Defines which file substrate is used.
 *
 * @since Version 1.0
 */
enum OTF2_FileSubstrate_enum {
  /** @brief Undefined.
   */
  OTF2_SUBSTRATE_UNDEFINED = 0,

  /** @brief Use standard posix file interface.
   */
  OTF2_SUBSTRATE_POSIX = 1,

  /** @brief Use the interface of the SIONlib to write many logical files into few
   *  physical files.
   */
  OTF2_SUBSTRATE_SION = 2,

  /** @brief Do not use any file interface. No data is written to a file.
   */
  OTF2_SUBSTRATE_NONE = 3
};

/** @brief Wrapper for enum @eref{OTF2_MappingType_enum}. */
typedef uint8_t OTF2_MappingType;

/**
 * @brief Possible mappings from local to global identifiers.
 *
 * @since Version 1.0
 */
enum OTF2_MappingType_enum {
  /** @brief Mapping of @eref{String} identifiers.
   */
  OTF2_MAPPING_STRING = 0,

  /** @brief Mapping of @eref{Attribute} identifiers.
   */
  OTF2_MAPPING_ATTRIBUTE = 1,

  /** @brief Mapping of @eref{Location} identifiers.
   */
  OTF2_MAPPING_LOCATION = 2,

  /** @brief Mapping of @eref{Region} identifiers.
   */
  OTF2_MAPPING_REGION = 3,

  /** @brief Mapping of @eref{Group} identifiers.
   */
  OTF2_MAPPING_GROUP = 4,

  /** @brief Mapping of @eref{Metric} identifiers.
   */
  OTF2_MAPPING_METRIC = 5,

  /** @brief Mapping of @eref{Comm} identifiers.
   */
  OTF2_MAPPING_COMM = 6,

  /** @brief Mapping of @eref{Parameter} identifiers.
   */
  OTF2_MAPPING_PARAMETER = 7,

  /** @brief Mapping of @eref{RmaWin} identifiers.
   *
   *  @since Version 1.2.
   */
  OTF2_MAPPING_RMA_WIN = 8,

  /** @brief Mapping of @eref{SourceCodeLocation} identifiers.
   *
   *  @since Version 1.5.
   */
  OTF2_MAPPING_SOURCE_CODE_LOCATION = 9,

  /** @brief Mapping of @eref{CallingContext} identifiers.
   *
   *  @since Version 1.5.
   */
  OTF2_MAPPING_CALLING_CONTEXT = 10,

  /** @brief Mapping of @eref{InterruptGenerator} identifiers.
   *
   *  @since Version 1.5.
   */
  OTF2_MAPPING_INTERRUPT_GENERATOR = 11,

  /** @brief Mapping of @eref{IoFile} identifiers.
   *
   *  @since Version 2.1.
   */
  OTF2_MAPPING_IO_FILE = 12,

  /** @brief Mapping of @eref{IoHandle} identifiers.
   *
   *  @since Version 2.1.
   */
  OTF2_MAPPING_IO_HANDLE = 13,

  /** @brief Mapping of @eref{LocationGroup} identifiers.
   *
   *  @since Version 3.0.
   */
  OTF2_MAPPING_LOCATION_GROUP = 14,

  /** @brief Max entry.
   */
  OTF2_MAPPING_MAX = 15
};

/** @brief Wrapper for enum @eref{OTF2_Type_enum}. */
typedef uint8_t OTF2_Type;

/**
 * @brief OTF2 basic data types.
 *
 * @since Version 1.0
 */
enum OTF2_Type_enum {
  /** @brief Undefined type.
   *
   *  Type category: None
   */
  OTF2_TYPE_NONE = 0,

  /** @brief Unsigned 8-bit integer.
   *
   *  Type category: Integer
   */
  OTF2_TYPE_UINT8 = 1,

  /** @brief Unsigned 16-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_UINT16 = 2,

  /** @brief Unsigned 32-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_UINT32 = 3,

  /** @brief Unsigned 64-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_UINT64 = 4,

  /** @brief Signed 8-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_INT8 = 5,

  /** @brief Signed 16-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_INT16 = 6,

  /** @brief Signed 32-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_INT32 = 7,

  /** @brief Signed 64-bit integer
   *
   *  Type category: Integer
   */
  OTF2_TYPE_INT64 = 8,

  /** @brief 32-bit floating point value
   *
   *  Type category: Floating point
   */
  OTF2_TYPE_FLOAT = 9,

  /** @brief 64-bit floating point value
   *
   *  Type category: Floating point
   */
  OTF2_TYPE_DOUBLE = 10,

  /** @brief Mapping of @eref{String} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_STRING = 11,

  /** @brief Mapping of @eref{Attribute} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_ATTRIBUTE = 12,

  /** @brief Mapping of @eref{Location} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_LOCATION = 13,

  /** @brief Mapping of @eref{Region} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_REGION = 14,

  /** @brief Mapping of @eref{Group} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_GROUP = 15,

  /** @brief Mapping of @eref{Metric} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_METRIC = 16,

  /** @brief Mapping of @eref{Comm} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_COMM = 17,

  /** @brief Mapping of @eref{Parameter} identifiers.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_PARAMETER = 18,

  /** @brief Mapping of @eref{RmaWin} identifiers.
   *
   *  @since Version 1.2.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_RMA_WIN = 19,

  /** @brief Mapping of @eref{SourceCodeLocation} identifiers.
   *
   *  @since Version 1.5.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_SOURCE_CODE_LOCATION = 20,

  /** @brief Mapping of @eref{CallingContext} identifiers.
   *
   *  @since Version 1.5.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_CALLING_CONTEXT = 21,

  /** @brief Mapping of @eref{InterruptGenerator} identifiers.
   *
   *  @since Version 1.5.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_INTERRUPT_GENERATOR = 22,

  /** @brief Mapping of @eref{IoFile} identifiers.
   *
   *  @since Version 2.1.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_IO_FILE = 23,

  /** @brief Mapping of @eref{IoHandle} identifiers.
   *
   *  @since Version 2.1.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_IO_HANDLE = 24,

  /** @brief Mapping of @eref{LocationGroup} identifiers.
   *
   *  @since Version 3.0.
   *
   *  Type category: Definition reference
   */
  OTF2_TYPE_LOCATION_GROUP = 25
};

/** @brief Wrapper for enum @eref{OTF2_Paradigm_enum}. */
typedef uint8_t OTF2_Paradigm;

/**
 * @brief List of known paradigms.  Parallel paradigms have their expected
 * paradigm class and known paradigm properties attached.
 *
 * @since Version 1.1
 */
enum OTF2_Paradigm_enum {
  /** @brief An unknown paradigm.
   */
  OTF2_PARADIGM_UNKNOWN = 0,

  /** @brief User instrumentation.
   */
  OTF2_PARADIGM_USER = 1,

  /** @brief Compiler instrumentation.
   */
  OTF2_PARADIGM_COMPILER = 2,

  /** @brief OpenMP.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_FORK_JOIN}
   */
  OTF2_PARADIGM_OPENMP = 3,

  /** @brief MPI.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_PROCESS}
   */
  OTF2_PARADIGM_MPI = 4,

  /** @brief CUDA.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_ACCELERATOR}
   *  @pproperty{OTF2_PARADIGM_PROPERTY_RMA_ONLY,@eref{OTF2_TRUE}}
   */
  OTF2_PARADIGM_CUDA = 5,

  /** @brief The measurement software.
   *
   *  @since Version 1.2.
   */
  OTF2_PARADIGM_MEASUREMENT_SYSTEM = 6,

  /** @brief POSIX threads.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_CREATE_WAIT}
   *
   *  @since Version 1.3.
   */
  OTF2_PARADIGM_PTHREAD = 7,

  /** @brief HMPP.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_ACCELERATOR}
   *
   *  @since Version 1.3.
   */
  OTF2_PARADIGM_HMPP = 8,

  /** @brief OmpSs.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_FORK_JOIN}
   *
   *  @since Version 1.3.
   */
  OTF2_PARADIGM_OMPSS = 9,

  /** @brief Hardware.
   *
   *  @since Version 1.3.
   */
  OTF2_PARADIGM_HARDWARE = 10,

  /** @brief GASPI.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_PROCESS}
   *
   *  @since Version 1.4.
   */
  OTF2_PARADIGM_GASPI = 11,

  /** @brief Unified Parallel C (UPC).
   *
   *  @pclass{OTF2_PARADIGM_CLASS_PROCESS}
   *
   *  @since Version 1.4.
   */
  OTF2_PARADIGM_UPC = 12,

  /** @brief SGI SHMEM, Cray SHMEM, OpenSHMEM.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_PROCESS}
   *  @pproperty{OTF2_PARADIGM_PROPERTY_RMA_ONLY,@eref{OTF2_TRUE}}
   *
   *  @since Version 1.4.
   */
  OTF2_PARADIGM_SHMEM = 13,

  /** @brief Windows threads.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_CREATE_WAIT}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_WINTHREAD = 14,

  /** @brief Qt threads.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_CREATE_WAIT}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_QTTHREAD = 15,

  /** @brief ACE threads.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_CREATE_WAIT}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_ACETHREAD = 16,

  /** @brief TBB threads.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_FORK_JOIN}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_TBBTHREAD = 17,

  /** @brief OpenACC directives.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_ACCELERATOR}
   *  @pproperty{OTF2_PARADIGM_PROPERTY_RMA_ONLY,@eref{OTF2_TRUE}}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_OPENACC = 18,

  /** @brief OpenCL API functions and kernels.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_ACCELERATOR}
   *  @pproperty{OTF2_PARADIGM_PROPERTY_RMA_ONLY,@eref{OTF2_TRUE}}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_OPENCL = 19,

  /** @brief Multicore Task API functions.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_THREAD_FORK_JOIN}
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_MTAPI = 20,

  /** @brief Functions recorded by sampling, not by any means of instrumentation.
   *
   *  Regions belonging to this paradigm are not eligible for
   *  @eref{CallingContext} definitions if these
   *  @eref{CallingContext} definitions will be referenced by
   *  @eref{CallingContextEnter} or @eref{CallingContextLeave}
   *  events.
   *
   *  @since Version 1.5.
   */
  OTF2_PARADIGM_SAMPLING = 21,

  /** @brief Entity does not belong to any specific paradigm.
   *
   *  @since Version 2.0.
   */
  OTF2_PARADIGM_NONE = 22,

  /** @brief HIP API functions and kernels.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_ACCELERATOR}
   *  @pproperty{OTF2_PARADIGM_PROPERTY_RMA_ONLY,@eref{OTF2_TRUE}}
   *
   *  @since Version 2.3.
   */
  OTF2_PARADIGM_HIP = 23,

  /** @brief Kokkos API functions and kernels.
   *
   *  @pclass{OTF2_PARADIGM_CLASS_ACCELERATOR}
   *  @pproperty{OTF2_PARADIGM_PROPERTY_RMA_ONLY,@eref{OTF2_TRUE}}
   *
   *  @since Version 2.3.
   */
  OTF2_PARADIGM_KOKKOS = 24
};

/** @brief Wrapper for enum @eref{OTF2_ParadigmClass_enum}. */
typedef uint8_t OTF2_ParadigmClass;

/**
 * @brief List of paradigm classes.
 *
 * @since Version 1.5
 */
enum OTF2_ParadigmClass_enum {
  /** @brief A communication paradigm across multiple processes.
   */
  OTF2_PARADIGM_CLASS_PROCESS = 0,

  /** @brief A threading paradigm which uses the fork/join model.
   */
  OTF2_PARADIGM_CLASS_THREAD_FORK_JOIN = 1,

  /** @brief A threading paradigm which uses the create/wait model.
   */
  OTF2_PARADIGM_CLASS_THREAD_CREATE_WAIT = 2,

  /** @brief A paradigm which uses external accelerators to offload computation.
   */
  OTF2_PARADIGM_CLASS_ACCELERATOR = 3
};

/** @brief Wrapper for enum @eref{OTF2_ParadigmProperty_enum}. */
typedef uint8_t OTF2_ParadigmProperty;

/**
 * @brief List of paradigm properties.
 *
 * @since Version 1.5
 */
enum OTF2_ParadigmProperty_enum {
  /** @brief Template for unnamed @eref{Comm} definitions. A unique name can be
   *  derived by replacing '${id}' with a unique ID.
   *
   *  Type: @eref{String}
   */
  OTF2_PARADIGM_PROPERTY_COMM_NAME_TEMPLATE = 0,

  /** @brief Template for unnamed @eref{RmaWin} definitions. A unique name can be
   *  derived by replacing '${id}' with a unique ID.
   *
   *  Type: @eref{String}
   */
  OTF2_PARADIGM_PROPERTY_RMA_WIN_NAME_TEMPLATE = 1,

  /** @brief Attests that this parallel paradigm only uses @eref{RmaWin}
   *  definitions. The @eref{Comm} definitions exists only for
   *  compliance and won't be referenced in event records.
   *
   *  Type: @eref{OTF2_Boolean}
   */
  OTF2_PARADIGM_PROPERTY_RMA_ONLY = 2
};

/** @brief Wrapper for enum @eref{OTF2_ThumbnailType_enum}. */
typedef uint8_t OTF2_ThumbnailType;

/**
 * @brief Type of definitions used as metric in a thumbnail.
 *
 * @since Version 1.2
 */
enum OTF2_ThumbnailType_enum {
  /** @brief The referenced definitions are of type @eref{Region}.
   */
  OTF2_THUMBNAIL_TYPE_REGION = 0,

  /** @brief The referenced definitions are of type @eref{MetricMember}.
   */
  OTF2_THUMBNAIL_TYPE_METRIC = 1,

  /** @brief The referenced definitions are of type @eref{Attribute}.
   */
  OTF2_THUMBNAIL_TYPE_ATTRIBUTES = 2
};

#endif /* !OTF2_GENERAL_DEFINITIONS_H */
