/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#pragma once
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include "pallas.h"
#include "pallas_attribute.h"
#include "pallas_read.h"
#include "pallas_write.h"
/* Event Records */
#ifdef __cplusplus
namespace pallas {
extern "C" {
#endif
extern void pallas_event_pop_data(const PALLAS(EventData) * e, void* data, size_t data_size, const byte** cursor);
extern void pallas_record_enter(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, RegionRef region_ref);
extern void pallas_read_enter(const EventData* data,AttributeList** attribute_list, RegionRef* region_ref);

extern void pallas_record_leave(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, RegionRef region_ref);
extern void pallas_read_leave(const EventData* data,AttributeList** attribute_list, RegionRef* region_ref);

extern void pallas_record_thread_begin(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_begin(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_end(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_end(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_team_begin(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_team_begin(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_team_end(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_team_end(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_fork(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint32_t numberOfRequestedThreads);
extern void pallas_read_thread_fork(const EventData* data,AttributeList** attribute_list, uint32_t* numberOfRequestedThreads);

extern void pallas_record_thread_join(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_join(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_acquire_lock(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint32_t lockID, uint32_t acquisitionOrder);
extern void pallas_read_thread_acquire_lock(const EventData* data,AttributeList** attribute_list, uint32_t* lockID, uint32_t* acquisitionOrder);

extern void pallas_record_thread_release_lock(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint32_t lockID, uint32_t acquisitionOrder);
extern void pallas_read_thread_release_lock(const EventData* data,AttributeList** attribute_list, uint32_t* lockID, uint32_t* acquisitionOrder);

extern void pallas_record_mpi_send(ThreadWriter* thread_writer,
                                   AttributeList* attribute_list,
                                   pallas_timestamp_t time,
                                   uint32_t receiver,
                                   uint32_t communicator,
                                   uint32_t msgTag,
                                   uint64_t msgLength);
extern void pallas_read_mpi_send(const EventData* data,
                                 AttributeList** attribute_list,
                                 uint32_t* receiver,
                                 uint32_t* communicator,
                                 uint32_t* msgTag,
                                 uint64_t* msgLength);

extern void pallas_record_mpi_isend(ThreadWriter* thread_writer,
                                    AttributeList* attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t receiver,
                                    uint32_t communicator,
                                    uint32_t msgTag,
                                    uint64_t msgLength,
                                    uint64_t requestID);
extern void pallas_read_mpi_isend(const EventData* data,
                                  AttributeList** attribute_list,
                                  uint32_t* receiver,
                                  uint32_t* communicator,
                                  uint32_t* msgTag,
                                  uint64_t* msgLength,
                                  uint64_t* requestID);

extern void pallas_record_mpi_isend_complete(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint64_t requestID);
extern void pallas_read_mpi_isend_complete(const EventData* data,AttributeList** attribute_list, uint64_t* requestID);

extern void pallas_record_mpi_irecv_request(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint64_t requestID);
extern void pallas_read_mpi_irecv_request(const EventData* data,AttributeList** attribute_list, uint64_t* requestID);

extern void pallas_record_mpi_recv(ThreadWriter* thread_writer,
                                   AttributeList* attribute_list,
                                   pallas_timestamp_t time,
                                   uint32_t sender,
                                   uint32_t communicator,
                                   uint32_t msgTag,
                                   uint64_t msgLength);
extern void pallas_read_mpi_recv(const EventData* data,
                                 AttributeList** attribute_list,
                                 uint32_t* sender,
                                 uint32_t* communicator,
                                 uint32_t* msgTag,
                                 uint64_t* msgLength);

extern void pallas_record_mpi_irecv(ThreadWriter* thread_writer,
                                    AttributeList* attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t sender,
                                    uint32_t communicator,
                                    uint32_t msgTag,
                                    uint64_t msgLength,
                                    uint64_t requestID);
extern void pallas_read_mpi_irecv(const EventData* data,
                                  AttributeList** attribute_list,
                                  uint32_t* sender,
                                  uint32_t* communicator,
                                  uint32_t* msgTag,
                                  uint64_t* msgLength,
                                  uint64_t* requestID);

extern void pallas_record_mpi_collective_begin(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_mpi_collective_begin(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_mpi_collective_end(ThreadWriter* thread_writer,
                                             AttributeList* attribute_list,
                                             pallas_timestamp_t time,
                                             uint32_t collectiveOp,
                                             uint32_t communicator,
                                             uint32_t root,
                                             uint64_t sizeSent,
                                             uint64_t sizeReceived);
extern void pallas_read_mpi_collective_end(const EventData* data,
                                           AttributeList** attribute_list,
                                           uint32_t* collectiveOp,
                                           uint32_t* communicator,
                                           uint32_t* root,
                                           uint64_t* sizeSent,
                                           uint64_t* sizeReceived);

extern void pallas_record_generic(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, StringRef event_name);

extern void pallas_record_singleton(ThreadWriter* thread_writer,
                                    AttributeList* attribute_list,
                                    enum Record record,
                                    pallas_timestamp_t time,
                                    uint32_t args_n_bytes,
                                    byte arg_array[]);
extern void pallas_read_singleton(const EventData* data,
                                  AttributeList** attribute_list,
                                  enum Record* record,
                                  uint32_t* args_n_bytes,
                                  byte* arg_array[]);

extern void pallas_record_omp_fork(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint32_t numberOfRequestedThreads);
extern void pallas_read_omp_fork(const EventData* data,AttributeList** attribute_list, uint32_t* numberOfRequestedThreads);

extern void pallas_record_omp_join(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_omp_join(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_omp_acquire_lock(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint32_t lockID, uint32_t acquisitionOrder);
extern void pallas_read_omp_acquire_lock(const EventData* data,AttributeList** attribute_list, uint32_t* lockID, uint32_t* acquisitionOrder);

extern void pallas_record_omp_release_lock(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint32_t lockID, uint32_t acquisitionOrder);
extern void pallas_read_omp_release_lock(const EventData* data,AttributeList** attribute_list, uint32_t* lockID, uint32_t* acquisitionOrder);

extern void pallas_record_omp_task_create(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint64_t taskID);
extern void pallas_read_omp_task_create(const EventData* data,AttributeList** attribute_list, uint64_t* taskID);

extern void pallas_record_omp_task_switch(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint64_t taskID);
extern void pallas_read_omp_task_switch(const EventData* data,AttributeList** attribute_list, uint64_t* taskID);

extern void pallas_record_omp_task_complete(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time, uint64_t taskID);
extern void pallas_read_omp_task_complete(const EventData* data,AttributeList** attribute_list, uint64_t* taskID);

extern void pallas_record_thread_task_create(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_task_create(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_task_switch(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_task_switch(const EventData* data,AttributeList** attribute_list);

extern void pallas_record_thread_task_complete(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time);
extern void pallas_read_thread_task_complete(const EventData* data,AttributeList** attribute_list);

extern void pallas_read_generic(const EventData* data,struct AttributeList** attribute_list, StringRef* event_name_ref);

#ifdef __cplusplus
};
}
#endif
extern C_CXX(_Thread_local, thread_local) int pallas_recursion_shield;

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
