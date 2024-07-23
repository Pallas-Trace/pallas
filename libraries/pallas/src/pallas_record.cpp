//
// Created by khatharsis on 08/07/24.
//

#include "pallas/pallas_record.h"
#include <stdarg.h>

namespace pallas {
static inline void init_event(Event* e, enum Record record) {
  e->event_size = offsetof(Event, event_data);
  e->record = record;
  memset(&e->event_data[0], 0, sizeof(e->event_data));
}

static inline void push_data(Event* e, void* data, size_t data_size) {
  size_t o = e->event_size - offsetof(Event, event_data);
  pallas_assert(o < 256);
  pallas_assert(o + data_size < 256);
  memcpy(&e->event_data[o], data, data_size);
  e->event_size += data_size;
}

void pallas_record_generic(ThreadWriter* thread_writer,
                           struct AttributeList* attribute_list,
                           pallas_timestamp_t time,
                           StringRef event_name) {
  pallas_record_singleton(thread_writer, attribute_list, PALLAS_EVENT_GENERIC, time, sizeof(event_name),
                          reinterpret_cast<byte*>(&event_name));
}

void pallas_record_singleton(ThreadWriter* thread_writer,
                             struct AttributeList* attribute_list,
                             Record record,
                             pallas_timestamp_t time,
                             uint32_t args_n_bytes,
                             byte arg_array[]) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, record);
  push_data(&e, arg_array, args_n_bytes);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}

void pallas_record_enter(ThreadWriter* thread_writer,
                         struct AttributeList* attribute_list __attribute__((unused)),
                         pallas_timestamp_t time,
                         RegionRef region_ref) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_ENTER);

  push_data(&e, &region_ref, sizeof(region_ref));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);

  thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_leave(ThreadWriter* thread_writer,
                         struct AttributeList* attribute_list __attribute__((unused)),
                         pallas_timestamp_t time,
                         RegionRef region_ref) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_LEAVE);

  push_data(&e, &region_ref, sizeof(region_ref));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_begin(ThreadWriter* thread_writer,
                                struct AttributeList* attribute_list __attribute__((unused)),
                                pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_BEGIN);

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_end(ThreadWriter* thread_writer,
                              struct AttributeList* attribute_list __attribute__((unused)),
                              pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_END);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_team_begin(ThreadWriter* thread_writer,
                                     struct AttributeList* attribute_list __attribute__((unused)),
                                     pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_TEAM_BEGIN);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_team_end(ThreadWriter* thread_writer,
                                   struct AttributeList* attribute_list __attribute__((unused)),
                                   pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_TEAM_END);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_thread_fork(ThreadWriter* thread_writer,
                               AttributeList* attributeList __attribute__((unused)),
                               pallas_timestamp_t time,
                               uint32_t numberOfRequestedThreads) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_FORK);
  push_data(&e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attributeList);

  pallas_recursion_shield--;
}

void pallas_record_thread_join(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time) {
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_JOIN);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);
  pallas_recursion_shield--;
}

void pallas_record_mpi_send(ThreadWriter* thread_writer,
                            struct AttributeList* attribute_list __attribute__((unused)),
                            pallas_timestamp_t time,
                            uint32_t receiver,
                            uint32_t communicator,
                            uint32_t msgTag,
                            uint64_t msgLength) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_SEND);

  push_data(&e, &receiver, sizeof(receiver));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_isend(ThreadWriter* thread_writer,
                             struct AttributeList* attribute_list __attribute__((unused)),
                             pallas_timestamp_t time,
                             uint32_t receiver,
                             uint32_t communicator,
                             uint32_t msgTag,
                             uint64_t msgLength,
                             uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_ISEND);

  push_data(&e, &receiver, sizeof(receiver));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));
  push_data(&e, &requestID, sizeof(requestID));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_isend_complete(ThreadWriter* thread_writer,
                                      struct AttributeList* attribute_list __attribute__((unused)),
                                      pallas_timestamp_t time,
                                      uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_ISEND_COMPLETE);

  push_data(&e, &requestID, sizeof(requestID));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_irecv_request(ThreadWriter* thread_writer,
                                     struct AttributeList* attribute_list __attribute__((unused)),
                                     pallas_timestamp_t time,
                                     uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_IRECV_REQUEST);

  push_data(&e, &requestID, sizeof(requestID));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_recv(ThreadWriter* thread_writer,
                            struct AttributeList* attribute_list __attribute__((unused)),
                            pallas_timestamp_t time,
                            uint32_t sender,
                            uint32_t communicator,
                            uint32_t msgTag,
                            uint64_t msgLength) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_RECV);

  push_data(&e, &sender, sizeof(sender));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_irecv(ThreadWriter* thread_writer,
                             struct AttributeList* attribute_list __attribute__((unused)),
                             pallas_timestamp_t time,
                             uint32_t sender,
                             uint32_t communicator,
                             uint32_t msgTag,
                             uint64_t msgLength,
                             uint64_t requestID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_IRECV);

  push_data(&e, &sender, sizeof(sender));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &msgTag, sizeof(msgTag));
  push_data(&e, &msgLength, sizeof(msgLength));
  push_data(&e, &requestID, sizeof(requestID));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_collective_begin(ThreadWriter* thread_writer,
                                        struct AttributeList* attribute_list __attribute__((unused)),
                                        pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_COLLECTIVE_BEGIN);

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}

void pallas_record_mpi_collective_end(ThreadWriter* thread_writer,
                                      struct AttributeList* attribute_list __attribute__((unused)),
                                      pallas_timestamp_t time,
                                      uint32_t collectiveOp,
                                      uint32_t communicator,
                                      uint32_t root,
                                      uint64_t sizeSent,
                                      uint64_t sizeReceived) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;

  Event e;
  init_event(&e, PALLAS_EVENT_MPI_COLLECTIVE_END);

  push_data(&e, &collectiveOp, sizeof(collectiveOp));
  push_data(&e, &communicator, sizeof(communicator));
  push_data(&e, &root, sizeof(root));
  push_data(&e, &sizeSent, sizeof(sizeSent));
  push_data(&e, &sizeReceived, sizeof(sizeReceived));

  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

  pallas_recursion_shield--;
}


void pallas_record_omp_fork(ThreadWriter* thread_writer,
                            AttributeList* attribute_list,
                            pallas_timestamp_t time,
                            uint32_t numberOfRequestedThreads) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_FORK);
  push_data(&e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

  pallas_recursion_shield--;
}
void pallas_record_omp_join(ThreadWriter* thread_writer, AttributeList* attribute_list, pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_JOIN);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_omp_acquire_lock(ThreadWriter* thread_writer,
                                    AttributeList* attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t lockID,
                                    uint32_t acquisitionOrder) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_ACQUIRE_LOCK);
  push_data(&e, &lockID, sizeof(lockID));
  //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}


void pallas_record_thread_acquire_lock(ThreadWriter* thread_writer,
                                    AttributeList* attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t lockID,
                                    uint32_t acquisitionOrder) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_ACQUIRE_LOCK);
  push_data(&e, &lockID, sizeof(lockID));
  //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}

void pallas_record_thread_release_lock(ThreadWriter* thread_writer,
                                       AttributeList* attribute_list,
                                       pallas_timestamp_t time,
                                       uint32_t lockID,
                                       uint32_t acquisitionOrder) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_RELEASE_LOCK);
  push_data(&e, &lockID, sizeof(lockID));
  //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_omp_release_lock(ThreadWriter* thread_writer,
                                    AttributeList* attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t lockID,
                                    uint32_t acquisitionOrder) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_RELEASE_LOCK);
  push_data(&e, &lockID, sizeof(lockID));
  //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_omp_task_create(ThreadWriter* thread_writer,
                                   AttributeList* attribute_list,
                                   pallas_timestamp_t time,
                                   uint64_t taskID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_TASK_CREATE);
  push_data(&e, &taskID, sizeof(taskID));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_omp_task_switch(ThreadWriter* thread_writer,
                                   AttributeList* attribute_list,
                                   pallas_timestamp_t time,
                                   uint64_t taskID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_TASK_SWITCH);
  push_data(&e, &taskID, sizeof(taskID));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_omp_task_complete(ThreadWriter* thread_writer,
                                     AttributeList* attribute_list,
                                     pallas_timestamp_t time,
                                     uint64_t taskID) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_OMP_TASK_COMPLETE);
  push_data(&e, &taskID, sizeof(taskID));
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_thread_task_create(ThreadWriter* thread_writer,
                                   AttributeList* attribute_list,
                                   pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_TASK_CREATE);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_thread_task_switch(ThreadWriter* thread_writer,
                                   AttributeList* attribute_list,
                                   pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_TASK_SWITCH);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}
void pallas_record_thread_task_complete(ThreadWriter* thread_writer,
                                     AttributeList* attribute_list,
                                     pallas_timestamp_t time) {
  if (pallas_recursion_shield)
    return;
  pallas_recursion_shield++;
  Event e;
  init_event(&e, PALLAS_EVENT_THREAD_TASK_COMPLETE);
  TokenId e_id = thread_writer->thread_trace.getEventId(&e);
  thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
  pallas_recursion_shield--;
}


}  // namespace pallas