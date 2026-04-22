/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <stdarg.h>

#include "pallas/pallas_record.h"

#include "pallas/utils/pallas_log.h"

namespace pallas {
static inline void init_event(EventData *e, enum Record record) {
    e->event_size = offsetof(EventData, event_data);
    e->record = record;
    memset(&e->event_data[0], 0, sizeof(e->event_data));
}

static inline void push_data(EventData *e, void *data, size_t data_size) {
    size_t o = e->event_size - offsetof(EventData, event_data);
    pallas_assert(o < PALLAS_EVENT_DATA_MAX_SIZE);
    pallas_assert(o + data_size < PALLAS_EVENT_DATA_MAX_SIZE);
    memcpy(&e->event_data[o], data, data_size);
    e->event_size += data_size;
}

void pallas_event_pop_data(const EventData *e, void *data, size_t data_size, const byte **cursor) {
    if (*cursor == nullptr) {
        /* initialize the cursor to the beginning of event data */
        *cursor = &e->event_data[0];
    }

    memcpy(data, *cursor, data_size);
    *cursor += data_size;
}

void pallas_record_generic(ThreadWriter *thread_writer,
                           struct AttributeList *attribute_list,
                           pallas_timestamp_t time,
                           StringRef event_name) {
    pallas_record_singleton(thread_writer, attribute_list, PALLAS_EVENT_GENERIC, time, sizeof(event_name),
                            reinterpret_cast<byte *>(&event_name));
}

#define PALLAS_READ_PROLOG(expected_event_type)			\
  const byte* cursor = nullptr;						    \
  pallas::Record event_type = data->record;				\
  pallas_assert(event_type == (expected_event_type));

void pallas_read_generic(const EventData *data,
                         struct AttributeList **attribute_list,
                         StringRef *event_name_ref) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_GENERIC);
    if (attribute_list) *attribute_list = NULL; // TODO : add support for attribute_lists
    if (event_name_ref) pallas_event_pop_data(data, event_name_ref, sizeof(*event_name_ref), &cursor);
}

void pallas_record_singleton(ThreadWriter *thread_writer,
                             struct AttributeList *attribute_list,
                             Record record,
                             pallas_timestamp_t time,
                             uint32_t args_n_bytes,
                             byte arg_array[]) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, record);
    push_data(&e, arg_array, args_n_bytes);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_record_enter(ThreadWriter *thread_writer,
                         struct AttributeList *attribute_list __attribute__((unused)),
                         pallas_timestamp_t time,
                         RegionRef region_ref) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_ENTER);

    push_data(&e, &region_ref, sizeof(region_ref));

    TokenId e_id = thread_writer->getEventId(&e);

    thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_enter(const EventData *data, struct AttributeList **attribute_list,
                       RegionRef *region_ref) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_ENTER);
    if (attribute_list) *attribute_list = nullptr; // TODO : add support for attribute_lists
    if (region_ref) pallas_event_pop_data(data, region_ref, sizeof(*region_ref), &cursor);
}

void pallas_record_leave(ThreadWriter *thread_writer,
                         struct AttributeList *attribute_list __attribute__((unused)),
                         pallas_timestamp_t time,
                         RegionRef region_ref) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_LEAVE);

    push_data(&e, &region_ref, sizeof(region_ref));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_leave(const EventData *data,
                       struct AttributeList **attribute_list,
                       RegionRef *region_ref) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_LEAVE);
    if (attribute_list) *attribute_list = NULL; // TODO : add support for attribute_lists
    if (region_ref) pallas_event_pop_data(data, region_ref, sizeof(*region_ref), &cursor);
}

void pallas_record_thread_begin(ThreadWriter *thread_writer,
                                struct AttributeList *attribute_list __attribute__((unused)),
                                pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_BEGIN);

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_thread_begin(const EventData *data,
                              struct AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_BEGIN);
    if (attribute_list) *attribute_list = NULL; // TODO : add support for attribute_lists
}

void pallas_record_thread_end(ThreadWriter *thread_writer,
                              struct AttributeList *attribute_list __attribute__((unused)),
                              pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_END);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

extern void pallas_read_thread_end(const EventData *data,
                                   AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_END);
    if (attribute_list) *attribute_list = NULL; // todo: implement
}

void pallas_record_thread_team_begin(ThreadWriter *thread_writer,
                                     struct AttributeList *attribute_list __attribute__((unused)),
                                     pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_TEAM_BEGIN);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_thread_team_begin(const EventData *data,
                                   AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_TEAM_BEGIN);
    if (attribute_list) *attribute_list = NULL; //TODO
}

void pallas_record_thread_team_end(ThreadWriter *thread_writer,
                                   struct AttributeList *attribute_list __attribute__((unused)),
                                   pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_TEAM_END);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_thread_team_end(const EventData *data,
                                 AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_TEAM_END);
    if (attribute_list) *attribute_list = NULL;
}

void pallas_record_thread_fork(ThreadWriter *thread_writer,
                               AttributeList *attribute_list __attribute__((unused)),
                               pallas_timestamp_t time,
                               uint32_t numberOfRequestedThreads) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_FORK);
    push_data(&e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_thread_fork(const EventData *data,
                             AttributeList **attribute_list,
                             uint32_t *numberOfRequestedThreads) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_FORK);
    if (attribute_list) *attribute_list = NULL;
    if (numberOfRequestedThreads)
        pallas_event_pop_data(data, numberOfRequestedThreads,
                              sizeof(*numberOfRequestedThreads), &cursor);
}

void pallas_record_thread_join(ThreadWriter *thread_writer,
                               AttributeList *attribute_list,
                               pallas_timestamp_t time) {
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_JOIN);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_thread_join(const EventData *data,
                             AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_JOIN);
    if (attribute_list) *attribute_list = NULL;
}

void pallas_record_mpi_send(ThreadWriter *thread_writer,
                            struct AttributeList *attribute_list __attribute__((unused)),
                            pallas_timestamp_t time,
                            uint32_t receiver,
                            uint32_t communicator,
                            uint32_t msgTag,
                            uint64_t msgLength) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_SEND);

    push_data(&e, &receiver, sizeof(receiver));
    push_data(&e, &communicator, sizeof(communicator));
    push_data(&e, &msgTag, sizeof(msgTag));
    push_data(&e, &msgLength, sizeof(msgLength));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_send(const EventData *data,
                          AttributeList **attribute_list,
                          uint32_t *receiver,
                          uint32_t *communicator,
                          uint32_t *msgTag,
                          uint64_t *msgLength) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_SEND);

    if (attribute_list) *attribute_list = NULL;
    if (receiver) pallas_event_pop_data(data, receiver, sizeof(*receiver), &cursor);
    if (communicator) pallas_event_pop_data(data, communicator, sizeof(*communicator), &cursor);
    if (msgTag) pallas_event_pop_data(data, msgTag, sizeof(*msgTag), &cursor);
    if (msgLength) pallas_event_pop_data(data, msgLength, sizeof(*msgLength), &cursor);
}

void pallas_record_mpi_isend(ThreadWriter *thread_writer,
                             struct AttributeList *attribute_list __attribute__((unused)),
                             pallas_timestamp_t time,
                             uint32_t receiver,
                             uint32_t communicator,
                             uint32_t msgTag,
                             uint64_t msgLength,
                             uint64_t requestID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_ISEND);

    push_data(&e, &receiver, sizeof(receiver));
    push_data(&e, &communicator, sizeof(communicator));
    push_data(&e, &msgTag, sizeof(msgTag));
    push_data(&e, &msgLength, sizeof(msgLength));
    push_data(&e, &requestID, sizeof(requestID));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_isend(const EventData *data,
                           AttributeList **attribute_list,
                           uint32_t *receiver,
                           uint32_t *communicator,
                           uint32_t *msgTag,
                           uint64_t *msgLength,
                           uint64_t *requestID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_ISEND);
    if (attribute_list) *attribute_list = NULL;


    if (receiver) pallas_event_pop_data(data, receiver, sizeof(*receiver), &cursor);
    if (communicator) pallas_event_pop_data(data, communicator, sizeof(*communicator), &cursor);
    if (msgTag) pallas_event_pop_data(data, msgTag, sizeof(*msgTag), &cursor);
    if (msgLength) pallas_event_pop_data(data, msgLength, sizeof(*msgLength), &cursor);
    if (requestID) pallas_event_pop_data(data, requestID, sizeof(*requestID), &cursor);
}


void pallas_record_mpi_isend_complete(ThreadWriter *thread_writer,
                                      struct AttributeList *attribute_list __attribute__((unused)),
                                      pallas_timestamp_t time,
                                      uint64_t requestID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_ISEND_COMPLETE);

    push_data(&e, &requestID, sizeof(requestID));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_isend_complete(const EventData *data,
                                    AttributeList **attribute_list,
                                    uint64_t *requestID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_ISEND_COMPLETE);
    if (attribute_list) *attribute_list = NULL;

    if (requestID) pallas_event_pop_data(data, requestID, sizeof(*requestID), &cursor);
}


void pallas_record_mpi_irecv_request(ThreadWriter *thread_writer,
                                     struct AttributeList *attribute_list __attribute__((unused)),
                                     pallas_timestamp_t time,
                                     uint64_t requestID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_IRECV_REQUEST);

    push_data(&e, &requestID, sizeof(requestID));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_irecv_request(const EventData *data,
                                   AttributeList **attribute_list,
                                   uint64_t *requestID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_IRECV_REQUEST);
    if (attribute_list) *attribute_list = NULL;

    if (requestID) pallas_event_pop_data(data, requestID, sizeof(*requestID), &cursor);
}


void pallas_record_mpi_recv(ThreadWriter *thread_writer,
                            struct AttributeList *attribute_list __attribute__((unused)),
                            pallas_timestamp_t time,
                            uint32_t sender,
                            uint32_t communicator,
                            uint32_t msgTag,
                            uint64_t msgLength) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_RECV);

    push_data(&e, &sender, sizeof(sender));
    push_data(&e, &communicator, sizeof(communicator));
    push_data(&e, &msgTag, sizeof(msgTag));
    push_data(&e, &msgLength, sizeof(msgLength));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_recv(const EventData *data,
                          AttributeList **attribute_list,
                          uint32_t *sender,
                          uint32_t *communicator,
                          uint32_t *msgTag,
                          uint64_t *msgLength) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_RECV);
    if (attribute_list) *attribute_list = NULL;

    if (sender) pallas_event_pop_data(data, sender, sizeof(*sender), &cursor);
    if (communicator) pallas_event_pop_data(data, communicator, sizeof(*communicator), &cursor);
    if (msgTag) pallas_event_pop_data(data, msgTag, sizeof(*msgTag), &cursor);
    if (msgLength) pallas_event_pop_data(data, msgLength, sizeof(*msgLength), &cursor);
}

void pallas_record_mpi_irecv(ThreadWriter *thread_writer,
                             struct AttributeList *attribute_list __attribute__((unused)),
                             pallas_timestamp_t time,
                             uint32_t sender,
                             uint32_t communicator,
                             uint32_t msgTag,
                             uint64_t msgLength,
                             uint64_t requestID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_IRECV);

    push_data(&e, &sender, sizeof(sender));
    push_data(&e, &communicator, sizeof(communicator));
    push_data(&e, &msgTag, sizeof(msgTag));
    push_data(&e, &msgLength, sizeof(msgLength));
    push_data(&e, &requestID, sizeof(requestID));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_irecv(const EventData *data,
                           AttributeList **attribute_list,
                           uint32_t *sender,
                           uint32_t *communicator,
                           uint32_t *msgTag,
                           uint64_t *msgLength,
                           uint64_t *requestID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_IRECV);
    if (attribute_list) *attribute_list = NULL;

    if (sender) pallas_event_pop_data(data, sender, sizeof(*sender), &cursor);
    if (communicator) pallas_event_pop_data(data, communicator, sizeof(*communicator), &cursor);
    if (msgTag) pallas_event_pop_data(data, msgTag, sizeof(*msgTag), &cursor);
    if (msgLength) pallas_event_pop_data(data, msgLength, sizeof(*msgLength), &cursor);
    if (requestID) pallas_event_pop_data(data, requestID, sizeof(*requestID), &cursor);
}


void pallas_record_mpi_collective_begin(ThreadWriter *thread_writer,
                                        struct AttributeList *attribute_list __attribute__((unused)),
                                        pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_COLLECTIVE_BEGIN);

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_collective_begin(const EventData *data,
                                      AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_COLLECTIVE_BEGIN);
    if (attribute_list) *attribute_list = NULL;
}


void pallas_record_mpi_collective_end(ThreadWriter *thread_writer,
                                      struct AttributeList *attribute_list __attribute__((unused)),
                                      pallas_timestamp_t time,
                                      uint32_t collectiveOp,
                                      uint32_t communicator,
                                      uint32_t root,
                                      uint64_t sizeSent,
                                      uint64_t sizeReceived) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;

    EventData e;
    init_event(&e, PALLAS_EVENT_MPI_COLLECTIVE_END);

    push_data(&e, &collectiveOp, sizeof(collectiveOp));
    push_data(&e, &communicator, sizeof(communicator));
    push_data(&e, &root, sizeof(root));
    push_data(&e, &sizeSent, sizeof(sizeSent));
    push_data(&e, &sizeReceived, sizeof(sizeReceived));

    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_mpi_collective_end(const EventData *data,
                                    AttributeList **attribute_list,
                                    uint32_t *collectiveOp,
                                    uint32_t *communicator,
                                    uint32_t *root,
                                    uint64_t *sizeSent,
                                    uint64_t *sizeReceived) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_MPI_COLLECTIVE_END);
    if (attribute_list) *attribute_list = NULL;


    if (collectiveOp) pallas_event_pop_data(data, collectiveOp, sizeof(*collectiveOp), &cursor);
    if (communicator) pallas_event_pop_data(data, communicator, sizeof(*communicator), &cursor);
    if (root) pallas_event_pop_data(data, root, sizeof(*root), &cursor);
    if (sizeSent) pallas_event_pop_data(data, sizeSent, sizeof(*sizeSent), &cursor);
    if (sizeReceived) pallas_event_pop_data(data, sizeReceived, sizeof(*sizeReceived), &cursor);
}

void pallas_record_omp_fork(ThreadWriter *thread_writer,
                            AttributeList *attribute_list,
                            pallas_timestamp_t time,
                            uint32_t numberOfRequestedThreads) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_FORK);
    push_data(&e, &numberOfRequestedThreads, sizeof(numberOfRequestedThreads));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_START, e_id, time, attribute_list);

    pallas_recursion_shield--;
}

void pallas_read_omp_fork(const EventData *data,
                          AttributeList **attribute_list,
                          uint32_t *numberOfRequestedThreads) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_FORK);
    if (attribute_list) *attribute_list = NULL;

    if (numberOfRequestedThreads)
        pallas_event_pop_data(data, numberOfRequestedThreads,
                              sizeof(*numberOfRequestedThreads), &cursor);
}


void pallas_record_omp_join(ThreadWriter *thread_writer,
                            AttributeList *attribute_list,
                            pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_JOIN);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_BLOCK_END, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_omp_join(const EventData *data,
                          AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_JOIN);
    if (attribute_list) *attribute_list = NULL;
}

void pallas_record_omp_acquire_lock(ThreadWriter *thread_writer,
                                    AttributeList *attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t lockID,
                                    uint32_t acquisitionOrder) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_ACQUIRE_LOCK);
    push_data(&e, &lockID, sizeof(lockID));
    //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_omp_acquire_lock(const EventData *data,
                                  AttributeList **attribute_list,
                                  uint32_t *lockID,
                                  uint32_t *acquisitionOrder) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_ACQUIRE_LOCK);
    if (attribute_list) *attribute_list = NULL;

    if (lockID) pallas_event_pop_data(data, lockID, sizeof(*lockID), &cursor);
}


void pallas_record_thread_acquire_lock(ThreadWriter *thread_writer,
                                       AttributeList *attribute_list,
                                       pallas_timestamp_t time,
                                       uint32_t lockID,
                                       uint32_t acquisitionOrder) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_ACQUIRE_LOCK);
    push_data(&e, &lockID, sizeof(lockID));
    //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
    TokenId e_id = thread_writer->getEventId(&e);
    // TODO We are currently discarding the "OTF2 PARADIGM" argument from OTF2
    //      And I'm not sure this is a good idea.
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_thread_acquire_lock(const EventData *data,
                                     AttributeList **attribute_list,
                                     uint32_t *lockID,
                                     uint32_t *acquisitionOrder) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_ACQUIRE_LOCK);
    if (attribute_list) *attribute_list = NULL;

    if (lockID) pallas_event_pop_data(data, lockID, sizeof(*lockID), &cursor);
    if (acquisitionOrder) *acquisitionOrder = 0;
}


void pallas_record_thread_release_lock(ThreadWriter *thread_writer,
                                       AttributeList *attribute_list,
                                       pallas_timestamp_t time,
                                       uint32_t lockID,
                                       uint32_t acquisitionOrder) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_RELEASE_LOCK);
    push_data(&e, &lockID, sizeof(lockID));
    //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_thread_release_lock(const EventData *data,
                                     AttributeList **attribute_list,
                                     uint32_t *lockID,
                                     uint32_t *acquisitionOrder) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_RELEASE_LOCK);
    if (attribute_list) *attribute_list = NULL;

    if (lockID) pallas_event_pop_data(data, lockID, sizeof(*lockID), &cursor);
    if (acquisitionOrder) *acquisitionOrder = 0;
}


void pallas_record_omp_release_lock(ThreadWriter *thread_writer,
                                    AttributeList *attribute_list,
                                    pallas_timestamp_t time,
                                    uint32_t lockID,
                                    uint32_t acquisitionOrder) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_RELEASE_LOCK);
    push_data(&e, &lockID, sizeof(lockID));
    //push_data(&e, &acquisitionOrder, sizeof(acquisitionOrder));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_omp_release_lock(const EventData *data,
                                  AttributeList **attribute_list,
                                  uint32_t *lockID,
                                  uint32_t *acquisitionOrder) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_RELEASE_LOCK);
    if (attribute_list) *attribute_list = NULL;

    if (lockID) pallas_event_pop_data(data, lockID, sizeof(*lockID), &cursor);
    if (acquisitionOrder) *acquisitionOrder = 0;
}


void pallas_record_omp_task_create(ThreadWriter *thread_writer,
                                   AttributeList *attribute_list,
                                   pallas_timestamp_t time,
                                   uint64_t taskID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_TASK_CREATE);
    push_data(&e, &taskID, sizeof(taskID));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_omp_task_create(const EventData *data,
                                 AttributeList **attribute_list,
                                 uint64_t *taskID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_TASK_CREATE);
    if (attribute_list) *attribute_list = NULL;

    if (taskID) pallas_event_pop_data(data, taskID, sizeof(*taskID), &cursor);
}

void pallas_record_omp_task_switch(ThreadWriter *thread_writer,
                                   AttributeList *attribute_list,
                                   pallas_timestamp_t time,
                                   uint64_t taskID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_TASK_SWITCH);
    push_data(&e, &taskID, sizeof(taskID));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_omp_task_switch(const EventData *data,
                                 AttributeList **attribute_list,
                                 uint64_t *taskID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_TASK_SWITCH);
    if (attribute_list) *attribute_list = NULL;

    if (taskID) pallas_event_pop_data(data, taskID, sizeof(*taskID), &cursor);
}

void pallas_record_omp_task_complete(ThreadWriter *thread_writer,
                                     AttributeList *attribute_list,
                                     pallas_timestamp_t time,
                                     uint64_t taskID) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_OMP_TASK_COMPLETE);
    push_data(&e, &taskID, sizeof(taskID));
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_omp_task_complete(const EventData *data,
                                   AttributeList **attribute_list,
                                   uint64_t *taskID) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_OMP_TASK_COMPLETE);
    if (attribute_list) *attribute_list = NULL;

    if (taskID) pallas_event_pop_data(data, taskID, sizeof(*taskID), &cursor);
}

void pallas_record_thread_task_create(ThreadWriter *thread_writer,
                                      AttributeList *attribute_list,
                                      pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_TASK_CREATE);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_thread_task_create(const EventData *data,
                                    AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_TASK_CREATE);
    if (attribute_list) *attribute_list = NULL;
}

void pallas_record_thread_task_switch(ThreadWriter *thread_writer,
                                      AttributeList *attribute_list,
                                      pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_TASK_SWITCH);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_thread_task_switch(const EventData *data,
                                    AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_TASK_SWITCH);
    if (attribute_list) *attribute_list = NULL;
}

void pallas_record_thread_task_complete(ThreadWriter *thread_writer,
                                        AttributeList *attribute_list,
                                        pallas_timestamp_t time) {
    if (pallas_recursion_shield)
        return;
    pallas_recursion_shield++;
    EventData e;
    init_event(&e, PALLAS_EVENT_THREAD_TASK_COMPLETE);
    TokenId e_id = thread_writer->getEventId(&e);
    thread_writer->storeEvent(PALLAS_SINGLETON, e_id, time, attribute_list);
    pallas_recursion_shield--;
}

void pallas_read_thread_task_complete(const EventData *data,
                                      AttributeList **attribute_list) {
    PALLAS_READ_PROLOG(PALLAS_EVENT_THREAD_TASK_COMPLETE);
    if (attribute_list) *attribute_list = NULL;
}
} // namespace pallas


/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
