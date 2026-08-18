/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cinttypes>
#include <iostream>

#include "pallas/pallas.h"
#include "pallas/pallas_attribute.h"
#include "pallas/pallas_archive.h"
#include "pallas/pallas_read.h"

#include "pallas/utils/pallas_log.h"


namespace pallas {
void Thread::printAttribute(AttributeRef ref) const {
  const Attribute* attr = archive->getAttribute(ref);
  if (attr) {
    const String* attr_string = archive->getString(attr->name);
    if (attr_string) {
      printf("\"%s\" <%d>", attr_string->str, ref);
      return;
    }
  }

  printf("INVALID <%d>", ref);
}

static enum AttributeType _guess_attribute_size(const AttributeData* attr) {
  uint16_t data_size = attr->struct_size - ATTRIBUTE_HEADER_SIZE;
  switch (data_size) {
  case 1:
    return PALLAS_TYPE_UINT8;
  case 2:
    return PALLAS_TYPE_UINT16;
  case 4:
    return PALLAS_TYPE_UINT32;
  case 8:
    return PALLAS_TYPE_UINT64;
  default:
    return PALLAS_TYPE_NONE;
  }
}

std::string Thread::stringRefToString(StringRef string_ref) const {
  auto* str = archive->getString(string_ref);
  if (str)
    return std::string(std::format("%s <%d>", str->str, string_ref));
  else
    return std::string(std::format("INVALID_STRING <%d>", string_ref));
}

void Thread::printString(StringRef string_ref) const {
  std::cout << stringRefToString(string_ref);
}

std::string Thread::attributeRefToString(AttributeRef attribute_ref) const {
  auto* attr = archive->getAttribute(attribute_ref);
  if (attr)
    return std::string(std::format("attribute <%d>", attribute_ref));
  else
    return std::string(std::format("INVALID_ATTRIBUTE <%d>", attribute_ref));
}
void Thread::printAttributeRef(AttributeRef attribute_ref) const {
  std::cout << attributeRefToString(attribute_ref);
}

std::string Thread::commRefToString(CommRef comm_ref) const {
  auto* comm = archive->getComm(comm_ref);
  if (comm) {
      auto* name = archive->getString(comm->name);
      return std::string(std::format("Comm %s %d <%d>", name->str, comm->group, comm_ref));
  } else {
      return std::string(std::format("INVALID_COMM <%d>", comm_ref));
  }
}
void Thread::printCommRef(CommRef comm_ref) const {
  std::cout << commRefToString(comm_ref);
}

std::string Thread::groupRefToString(GroupRef group_ref) const {
  auto* group = archive->getGroup(group_ref);
  if (group) {
      auto* name = archive->getString(group->name);
      return std::string(std::format("Group %s <%d>", name->str, group_ref));
  } else {
      return std::string(std::format("INVALID_GROUP <%d>", group_ref));
  }
}
void Thread::printGroupRef(GroupRef group_ref) const {
  std::cout << groupRefToString(group_ref);
}

std::string Thread::locationRefToString(Ref location_ref) const {
  auto* attr = archive->getLocation(location_ref);
  if (attr)
    return std::string(std::format("location <%d>", location_ref));
  else
    return std::string(std::format("INVALID_LOCATION <%d>", location_ref));
}

void Thread::printLocation(Ref location_ref) const {
  std::cout << locationRefToString(location_ref);
}

std::string Thread::regionRefToString(RegionRef region_ref) const {
  auto* attr = archive->getRegion(region_ref);
  if (attr)
    return std::string(std::format("region <%d>", region_ref));
  else
    return std::string(std::format("INVALID_REGION <%d>", region_ref));
}

void Thread::printRegion(RegionRef region_ref) const {
  std::cout << regionRefToString(region_ref);
}

static std::string _group_ref_to_string(Ref group_ref) {
  return std::string(std::format("group <%d>", group_ref));
}
static void _pallas_print_group(Ref group_ref) {
  std::cout << _group_ref_to_string(group_ref);
}

static std::string _metric_ref_to_string(Ref metric_ref) {
  return std::string(std::format("metric <%d>", metric_ref));
} 
static void _pallas_print_metric(Ref metric_ref) {
  std::cout << _metric_ref_to_string(metric_ref);
}

static std::string _comm_ref_to_string(Ref comm_ref) {
  return std::string(std::format("comm <%d>", comm_ref));
}
static void _pallas_print_comm(Ref comm_ref) {
  std::cout << _comm_ref_to_string(comm_ref);
}

static std::string _parameter_ref_to_string(Ref parameter_ref) {
  return std::string(std::format("parameter <%d>", parameter_ref));
}
static void _pallas_print_parameter(Ref parameter_ref) {
  std::cout << _parameter_ref_to_string(parameter_ref);
}

static std::string _rma_win_ref_to_string(Ref rma_win_ref) {
  return std::string(std::format("rma_win <%d>", rma_win_ref));
}
static void _pallas_print_rma_win(Ref rma_win_ref) {
  std::cout << _rma_win_ref_to_string(rma_win_ref);
}

static std::string _source_code_location_ref_to_string(Ref source_code_location_ref) {
  return std::string(std::format("source_code_location <%d>", source_code_location_ref));
}
static void _pallas_print_source_code_location(Ref source_code_location_ref) {
  std::cout << _source_code_location_ref_to_string(source_code_location_ref);
}

static std::string _calling_context_ref_to_string(Ref calling_context_ref) {
  return std::string(std::format("calling_context <%d>", calling_context_ref));
}
static void _pallas_print_calling_context(Ref calling_context_ref) {
  std::cout << _calling_context_ref_to_string(calling_context_ref);
}

static std::string _interrupt_generator_ref_to_string(Ref interrupt_generator_ref) {
  return std::string(std::format("interrupt_generator <%d>", interrupt_generator_ref));
}
static void _pallas_print_interrupt_generator(Ref interrupt_generator_ref) {
  std::cout << _interrupt_generator_ref_to_string(interrupt_generator_ref);
}

static std::string _io_file_ref_to_string(Ref io_file_ref) {
  return std::string(std::format("io_file <%d>", io_file_ref));
}
static void _pallas_print_io_file(Ref io_file_ref) {
  std::cout << _io_file_ref_to_string(io_file_ref);
}

static std::string _io_handle_ref_to_string(Ref io_handle_ref) {
  return std::string(std::format("io_handle <%d>", io_handle_ref));
}
static void _pallas_print_io_handle(Ref io_handle_ref) {
  std::cout << _io_handle_ref_to_string(io_handle_ref);
}

static std::string _location_group_ref_to_string(Ref location_group_ref) {
  return std::string(std::format("location_group <%d>", location_group_ref));
}
static void _pallas_print_location_group(Ref location_group_ref) {
  std::cout << _location_group_ref_to_string(location_group_ref);
}

std::string Thread::attributeValueToString(const struct AttributeData* attr, pallas_type_t type) const {
  switch (type) {
  case PALLAS_TYPE_NONE:
    return std::string("NONE");
    break;
  case PALLAS_TYPE_UINT8:
    return std::to_string(attr->value.uint8);
    break;
  case PALLAS_TYPE_UINT16:
    return std::to_string(attr->value.uint16);
    break;
  case PALLAS_TYPE_UINT32:
    return std::to_string(attr->value.uint32);
    break;
  case PALLAS_TYPE_UINT64:
    return std::to_string(attr->value.uint64);
    break;
  case PALLAS_TYPE_INT8:
    return std::to_string(attr->value.int8);
    break;
  case PALLAS_TYPE_INT16:
    return std::to_string(attr->value.int16);
    break;
  case PALLAS_TYPE_INT32:
    return std::to_string(attr->value.int32);
    break;
  case PALLAS_TYPE_INT64:
    return std::to_string(attr->value.int64);
    break;
  case PALLAS_TYPE_FLOAT:
    return std::to_string(attr->value.float32);
    break;
  case PALLAS_TYPE_DOUBLE:
    return std::to_string(attr->value.float64);
    break;
  case PALLAS_TYPE_STRING:
    return stringRefToString(attr->value.string_ref);
    break;
  case PALLAS_TYPE_ATTRIBUTE:
    return attributeRefToString(attr->value.attribute_ref);
    break;
  case PALLAS_TYPE_LOCATION:
    return locationRefToString(attr->value.location_ref);
    break;
  case PALLAS_TYPE_REGION:
    return regionRefToString(attr->value.region_ref);
    break;
  case PALLAS_TYPE_GROUP:
    return groupRefToString(attr->value.group_ref);
    break;
  case PALLAS_TYPE_METRIC:
    return _metric_ref_to_string(attr->value.metric_ref);
    break;
  case PALLAS_TYPE_COMM:
    return _comm_ref_to_string(attr->value.comm_ref);
    break;
  case PALLAS_TYPE_PARAMETER:
    return _parameter_ref_to_string(attr->value.parameter_ref);
    break;
  case PALLAS_TYPE_RMA_WIN:
    return _rma_win_ref_to_string(attr->value.rma_win_ref);
    break;
  case PALLAS_TYPE_SOURCE_CODE_LOCATION:
    return _source_code_location_ref_to_string(attr->value.source_code_location_ref);
    break;
  case PALLAS_TYPE_CALLING_CONTEXT:
    return _calling_context_ref_to_string(attr->value.calling_context_ref);
    break;
  case PALLAS_TYPE_INTERRUPT_GENERATOR:
    return _interrupt_generator_ref_to_string(attr->value.interrupt_generator_ref);
    break;
  case PALLAS_TYPE_IO_FILE:
    return _io_file_ref_to_string(attr->value.io_file_ref);
    break;
  case PALLAS_TYPE_IO_HANDLE:
    return _io_handle_ref_to_string(attr->value.io_handle_ref);
    break;
  case PALLAS_TYPE_LOCATION_GROUP:
    return _location_group_ref_to_string(attr->value.location_group_ref);
    break;
  }
  return std::string("Invalid");
}

void Thread::printAttributeValue(const struct AttributeData* attr, pallas_type_t type) const {
  std::cout<< attributeValueToString(attr, type);
}

std::string Thread::attributeToString(const struct AttributeData* attr) const {
  const char* attr_string = "INVALID";
  enum AttributeType type = _guess_attribute_size(attr);

  auto* a = archive->getAttribute(attr->ref);
  if (a) {
    auto* str = archive->getString(a->name);
    if (str) {
      attr_string = str->str;
    }

    type = static_cast<AttributeType>(a->type);
  }

  std::string s1= std::string(std::format("%s <%d>: ", attr_string, attr->ref));
  return s1 + attributeValueToString(attr, type);
}

void Thread::printAttribute(const struct AttributeData* attr) const {
  std::cout << attributeToString(attr) << std::endl;
}

std::vector<AttributeData> Thread::getAttributes(const AttributeList* attribute_list) const {
  std::vector<AttributeData> attributes;
  if (attribute_list == nullptr)
    return attributes;

  uint16_t pos = 0;
  for (int i = 0; i < attribute_list->nb_values; i++) {
    AttributeData attr;
    pallas_attribute_list_pop_data(attribute_list, &attr, &pos);
    pallas_assert(ATTRIBUTE_LIST_HEADER_SIZE + pos <= attribute_list->struct_size);
    attributes.push_back(attr);
  }
  return attributes;
}

void Thread::printAttributeList(const AttributeList* attribute_list) const {
  if (attribute_list == nullptr)
    return;
  auto attributes = getAttributes(attribute_list);
  printf(" { ");

  bool first = true;
  for (const auto& attr : attributes) {
    if(!first) {
      printf(", ");
    }
    first = false;
    printAttribute(&attr);
  }
  printf(" }");
}

std::vector<AttributeData> Thread::getAttributes(const struct EventOccurrence *es) const {
  if(!es) {
    return std::vector<AttributeData>();
  }
  return getAttributes(es->attributes);
}

void Thread::printEventAttribute(const struct EventOccurrence* e) const {
  printAttributeList(e->attributes);
}

} // namespace pallas

void pallas_attribute_list_push_data(pallas::AttributeList * l, pallas::AttributeData * data) {
  uintptr_t offset = l->struct_size;
  pallas_assert(offset + data->struct_size <= ATTRIBUTE_MAX_BUFFER_SIZE);
  uintptr_t addr = ((uintptr_t)l) + offset;
  memcpy((void*)addr, data, data->struct_size);
  l->struct_size += data->struct_size;
  l->nb_values++;
}

void pallas_attribute_list_pop_data(const pallas::AttributeList * l,
                                    pallas::AttributeData * data,
                                    uint16_t* current_offset) {
  uintptr_t addr = ((uintptr_t)&l->attributes[0]) + (*current_offset);
  pallas::AttributeData* attr_data = (pallas::AttributeData*)addr;
  uint16_t struct_size = attr_data->struct_size;

  pallas_assert(struct_size + (*current_offset) <= ATTRIBUTE_MAX_BUFFER_SIZE);
  pallas_assert(struct_size + (*current_offset) <= l->struct_size);

  memcpy(data, attr_data, struct_size);
  data->struct_size = struct_size;
  *current_offset += struct_size;
}

void pallas_attribute_list_init(pallas::AttributeList * l) {
  l->index = -1;
  l->nb_values = 0;
  l->struct_size = ATTRIBUTE_LIST_HEADER_SIZE;
}

void pallas_attribute_list_finalize(pallas::AttributeList * l __attribute__((unused))) {}

int pallas_attribute_list_add_attribute(pallas::AttributeList * list,
                                        pallas::AttributeRef attribute,
                                        size_t data_size,
                                        pallas::AttributeValue value) {
  if (list->nb_values + 1 >= NB_ATTRIBUTE_MAX) {
    pallas_warn("[PALLAS] too many attributes\n");
    return -1;
  }
  pallas::AttributeData d;
  d.ref = attribute;
  d.value = value;
  d.struct_size = ATTRIBUTE_HEADER_SIZE + data_size;

  pallas_attribute_list_push_data(list, &d);
  return 0;
}

void pallas_print_attribute_value(pallas::Thread* thread, pallas::AttributeData* attr, pallas::pallas_type_t type) {
  thread->printAttributeValue(attr, type);
};

void pallas_print_event_attributes(pallas::Thread* thread, pallas::EventOccurrence* e) {
  thread->printEventAttribute(e);
};

void pallas_print_attribute_list(pallas::Thread* thread, pallas::AttributeList* l) {
  thread->printAttributeList(l);
};


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
