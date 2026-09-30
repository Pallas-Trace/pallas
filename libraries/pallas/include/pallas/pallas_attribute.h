/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

/** @file
 * This file contains the definitions and needed functions for Attributes.
 */

#pragma once
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include "pallas.h"

#ifdef __cplusplus
namespace pallas {
#endif
/** Value container for an attributes.
 */
typedef union AttributeValue {
  /** Arbitrary value of type uint8_t */
  uint8_t uint8;
  /** Arbitrary value of type uint16_t */
  uint16_t uint16;
  /** Arbitrary value of type uint32_t */
  uint32_t uint32;
  /** Arbitrary value of type uint64_t */
  uint64_t uint64;
  /** Arbitrary value of type int8_t */
  int8_t int8;
  /** Arbitrary value of type int16_t */
  int16_t int16;
  /** Arbitrary value of type int32_t */
  int32_t int32;
  /** Arbitrary value of type int64_t */
  int64_t int64;
  /** Arbitrary value of type float */
  float float32;
  /** Arbitrary value of type double */
  double float64;
  /** References a String definition */
  StringRef string_ref;
  /** References a Attribute definition */
  AttributeRef attribute_ref;
  /** References a Location definition */
  Ref location_ref;
  /** References a Region definition */
  RegionRef region_ref;
  /** References a Group definition */
  Ref group_ref;
  /** References a MetricClass, or a MetricInstance definition */
  Ref metric_ref;
  /** References a Comm, or a InterComm definition */
  Ref comm_ref;
  /** References a Parameter definition */
  Ref parameter_ref;
  /** References a RmaWin definition */
  Ref rma_win_ref;
  /** References a SourceCodeLocation definition */
  Ref source_code_location_ref;
  /** References a CallingContext definition */
  Ref calling_context_ref;
  /** References a InterruptGenerator definition */
  Ref interrupt_generator_ref;
  /** References a IoRegularFile, or a IoDirectory definition */
  Ref io_file_ref;
  /** References a IoHandle definition */
  Ref io_handle_ref;
  /** References a LocationGroup definition */
  Ref location_group_ref;
} AttributeValue;

typedef Type AttributeType;

#define NB_ATTRIBUTE_MAX 128
#define ATTRIBUTE_MAX_BUFFER_SIZE (sizeof(PALLAS(AttributeList)))
#define ATTRIBUTE_HEADER_SIZE (sizeof(uint16_t) + sizeof(PALLAS(AttributeRef)))
/** Data container for an attribute. */
typedef struct AttributeData {
  uint16_t struct_size;
  AttributeRef ref;
  union AttributeValue value;
} __attribute__((packed)) AttributeData;

#define ATTRIBUTE_LIST_HEADER_SIZE (sizeof(int) + sizeof(uint16_t) + sizeof(uint8_t))
/** Array of attributes. */
typedef struct AttributeList {
  int index;
  uint16_t struct_size;
  uint8_t nb_values;
  AttributeData attributes[NB_ATTRIBUTE_MAX];
} __attribute__((packed)) AttributeList;

static inline size_t get_value_size(pallas_type_t t) {
  union AttributeValue u;
  switch (t) {
  case PALLAS_TYPE_NONE:
    return 0;
    break;
  case PALLAS_TYPE_UINT8:
    return sizeof(u.uint8);
    break;
  case PALLAS_TYPE_UINT16:
    return sizeof(u.uint16);
    break;
  case PALLAS_TYPE_UINT32:
    return sizeof(u.uint32);
    break;
  case PALLAS_TYPE_UINT64:
    return sizeof(u.uint64);
    break;
  case PALLAS_TYPE_INT8:
    return sizeof(u.int8);
    break;
  case PALLAS_TYPE_INT16:
    return sizeof(u.int16);
    break;
  case PALLAS_TYPE_INT32:
    return sizeof(u.int32);
    break;
  case PALLAS_TYPE_INT64:
    return sizeof(u.int64);
    break;
  case PALLAS_TYPE_FLOAT:
    return sizeof(u.float32);
    break;
  case PALLAS_TYPE_DOUBLE:
    return sizeof(u.float64);
    break;
  case PALLAS_TYPE_STRING:
    return sizeof(u.string_ref);
    break;
  case PALLAS_TYPE_ATTRIBUTE:
    return sizeof(u.attribute_ref);
    break;
  case PALLAS_TYPE_LOCATION:
    return sizeof(u.location_ref);
    break;
  case PALLAS_TYPE_REGION:
    return sizeof(u.region_ref);
    break;
  case PALLAS_TYPE_GROUP:
    return sizeof(u.group_ref);
    break;
  case PALLAS_TYPE_METRIC:
    return sizeof(u.metric_ref);
    break;
  case PALLAS_TYPE_COMM:
    return sizeof(u.comm_ref);
    break;
  case PALLAS_TYPE_PARAMETER:
    return sizeof(u.parameter_ref);
    break;
  case PALLAS_TYPE_RMA_WIN:
    return sizeof(u.rma_win_ref);
    break;
  case PALLAS_TYPE_SOURCE_CODE_LOCATION:
    return sizeof(u.source_code_location_ref);
    break;
  case PALLAS_TYPE_CALLING_CONTEXT:
    return sizeof(u.calling_context_ref);
    break;
  case PALLAS_TYPE_INTERRUPT_GENERATOR:
    return sizeof(u.interrupt_generator_ref);
    break;
  case PALLAS_TYPE_IO_FILE:
    return sizeof(u.io_file_ref);
    break;
  case PALLAS_TYPE_IO_HANDLE:
    return sizeof(u.io_handle_ref);
    break;
  case PALLAS_TYPE_LOCATION_GROUP:
    return sizeof(u.location_group_ref);
    break;
  }
  return 0;
}
#ifdef __cplusplus
}; /* namespace pallas */
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern void pallas_attribute_list_push_data(PALLAS(AttributeList) * l, PALLAS(AttributeData) * data);

extern void pallas_attribute_list_pop_data(const PALLAS(AttributeList) * l,
                                           PALLAS(AttributeData) * data,
                                           uint16_t* current_offset);

extern void pallas_attribute_list_init(PALLAS(AttributeList) * l);

extern void pallas_attribute_list_finalize(PALLAS(AttributeList) * l __attribute__((unused)));

extern int pallas_attribute_list_add_attribute(PALLAS(AttributeList) * list,
                                               PALLAS(AttributeRef) attribute,
                                               size_t data_size,
                                               PALLAS(AttributeValue) value);

extern void pallas_print_attribute_value(PALLAS(Thread) * thread,
                                         PALLAS(AttributeData) * attr,
                                         PALLAS(pallas_type_t) type);

// extern void pallas_print_event_attributes(PALLAS(Thread) * thread, struct PALLAS(EventOccurrence) * e);

extern void pallas_print_attribute_list(PALLAS(Thread) * thread, PALLAS(AttributeList) * l);

#ifdef __cplusplus
};
#endif
