/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <cinttypes>

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

void Thread::printString(StringRef string_ref) const {
  auto* str = archive->getString(string_ref);
  if (str)
    printf("%s <%d>", str->str, string_ref);
  else
    printf("INVALID_STRING <%d>", string_ref);
}

void Thread::printAttributeRef(AttributeRef attribute_ref) const {
  auto* attr = archive->getAttribute(attribute_ref);
  if (attr)
    printf("attribute <%d>", attribute_ref);
  else
    printf("INVALID_ATTRIBUTE <%d>", attribute_ref);
}

void Thread::printCommRef(CommRef comm_ref) const {
    auto* comm = archive->getComm(comm_ref);
    if (comm) {
        auto* name = archive->getString(comm->name);
        printf("Comm %s %d <%d>", name->str, comm->group, comm_ref);
    } else
        printf("INVALID_COMM <%d>", comm_ref);
}

void Thread::printGroupRef(GroupRef group_ref) const {
    auto* group = archive->getGroup(group_ref);
    if (group) {
        auto* name = archive->getString(group->name);
        printf("Group %s <%d>", name->str, group_ref);
    } else
        printf("INVALID_GROUP <%d>", group_ref);
}

void Thread::printLocation(Ref location_ref) const {
  auto* attr = archive->getLocation(location_ref);
  if (attr)
    printf("location <%d>", location_ref);
  else
    printf("INVALID_LOCATION <%d>", location_ref);
}

void Thread::printRegion(Ref region_ref) const {
  auto* attr = archive->getRegion(region_ref);
  if (attr)
    printf("region <%d>", region_ref);
  else
    printf("INVALID_REGION <%d>", region_ref);
}

static void _pallas_print_group(Ref group_ref) {
  printf("group <%d>", group_ref);
}

static void _pallas_print_metric(Ref metric_ref) {
  printf("metric <%d>", metric_ref);
}

static void _pallas_print_comm(Ref comm_ref) {
  printf("comm <%d>", comm_ref);
}

static void _pallas_print_parameter(Ref parameter_ref) {
  printf("parameter <%d>", parameter_ref);
}

static void _pallas_print_rma_win(Ref rma_win_ref) {
  printf("rma_win <%d>", rma_win_ref);
}

static void _pallas_print_source_code_location(Ref source_code_location_ref) {
  printf("source_code_location <%d>", source_code_location_ref);
}

static void _pallas_print_calling_context(Ref calling_context_ref) {
  printf("calling_context <%d>", calling_context_ref);
}

static void _pallas_print_interrupt_generator(Ref interrupt_generator_ref) {
  printf("interrupt_generator <%d>", interrupt_generator_ref);
}

static void _pallas_print_io_file(Ref io_file_ref) {
  printf("io_file <%d>", io_file_ref);
}

static void _pallas_print_io_handle(Ref io_handle_ref) {
  printf("io_handle <%d>", io_handle_ref);
}

static void _pallas_print_location_group(Ref location_group_ref) {
  printf("location_group <%d>", location_group_ref);
}

void Thread::printAttributeValue(const struct AttributeData* attr, pallas_type_t type) const {
  switch (type) {
  case PALLAS_TYPE_NONE:
    printf("NONE");
    break;
  case PALLAS_TYPE_UINT8:
    printf("%u", attr->value.uint8);
    break;
  case PALLAS_TYPE_UINT16:
    printf("%u", attr->value.uint16);
    break;
  case PALLAS_TYPE_UINT32:
    printf("%u", attr->value.uint32);
    break;
  case PALLAS_TYPE_UINT64:
    printf("%" PRIu64, attr->value.uint64);
    break;
  case PALLAS_TYPE_INT8:
    printf("%d", attr->value.int8);
    break;
  case PALLAS_TYPE_INT16:
    printf("%d", attr->value.int16);
    break;
  case PALLAS_TYPE_INT32:
    printf("%d", attr->value.int32);
    break;
  case PALLAS_TYPE_INT64:
    printf("%" PRId64, attr->value.int64);
    break;
  case PALLAS_TYPE_FLOAT:
    printf("%f", attr->value.float32);
    break;
  case PALLAS_TYPE_DOUBLE:
    printf("%lf", attr->value.float64);
    break;
  case PALLAS_TYPE_STRING:
    printString(attr->value.string_ref);
    break;
  case PALLAS_TYPE_ATTRIBUTE:
    printAttributeRef(attr->value.attribute_ref);
    break;
  case PALLAS_TYPE_LOCATION:
    printLocation(attr->value.location_ref);
    break;
  case PALLAS_TYPE_REGION:
    printRegion(attr->value.region_ref);
    break;
  case PALLAS_TYPE_GROUP:
    _pallas_print_group(attr->value.group_ref);
    break;
  case PALLAS_TYPE_METRIC:
    _pallas_print_metric(attr->value.metric_ref);
    break;
  case PALLAS_TYPE_COMM:
    _pallas_print_comm(attr->value.comm_ref);
    break;
  case PALLAS_TYPE_PARAMETER:
    _pallas_print_parameter(attr->value.parameter_ref);
    break;
  case PALLAS_TYPE_RMA_WIN:
    _pallas_print_rma_win(attr->value.rma_win_ref);
    break;
  case PALLAS_TYPE_SOURCE_CODE_LOCATION:
    _pallas_print_source_code_location(attr->value.source_code_location_ref);
    break;
  case PALLAS_TYPE_CALLING_CONTEXT:
    _pallas_print_calling_context(attr->value.calling_context_ref);
    break;
  case PALLAS_TYPE_INTERRUPT_GENERATOR:
    _pallas_print_interrupt_generator(attr->value.interrupt_generator_ref);
    break;
  case PALLAS_TYPE_IO_FILE:
    _pallas_print_io_file(attr->value.io_file_ref);
    break;
  case PALLAS_TYPE_IO_HANDLE:
    _pallas_print_io_handle(attr->value.io_handle_ref);
    break;
  case PALLAS_TYPE_LOCATION_GROUP:
    _pallas_print_location_group(attr->value.location_group_ref);
    break;
  }
}

void Thread::printAttribute(const struct AttributeData* attr) const {
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

  printf("%s <%d>: ", attr_string, attr->ref);
  printAttributeValue(attr, type);
}

void Thread::printAttributeList(const AttributeList* attribute_list) const {
  if (attribute_list == nullptr)
    return;
  printf(" { ");
  uint16_t pos = 0;
  for (int i = 0; i < attribute_list->nb_values; i++) {
    AttributeData attr;
    pallas_attribute_list_pop_data(attribute_list, &attr, &pos);
    pallas_assert(ATTRIBUTE_LIST_HEADER_SIZE + pos <= attribute_list->struct_size);

    if (i > 0)
      printf(", ");
    printAttribute(&attr);
  }
  printf(" }");
}

void Thread::printEventAttribute(const struct EventOccurence* e) const {
  printAttributeList(e->attributes);
}
}  // namespace pallas

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

void pallas_print_event_attributes(pallas::Thread* thread, pallas::EventOccurence* e) {
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
