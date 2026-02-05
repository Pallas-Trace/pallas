#include "pallas_python.h"
#include "python_tokens.h"

std::string Token_toString(pallas::Token t) {
    std::string out;
    switch (t.type) {
    case pallas::TypeInvalid:
        out += "U";
        break;
    case pallas::TypeEvent:
        out += "E";
        break;
    case pallas::TypeSequence:
        out += "S";
        break;
    case pallas::TypeLoop:
        out += "L";
        break;
    }
    out += std::to_string(t.id);
    return out;
}

py::dict& EventData_get_data(pallas::EventData* data) {
    auto& dict = *new py::dict();
    byte* cursor = nullptr;

    switch (data->record) {
    case pallas::PALLAS_EVENT_ENTER:
    case pallas::PALLAS_EVENT_LEAVE: {
        READ(data, cursor, pallas::RegionRef, region_ref);
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_FORK:
    case pallas::PALLAS_EVENT_OMP_FORK: {
        READ(data, cursor, uint32_t, numberOfRequestedThreads);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_SEND:
    case pallas::PALLAS_EVENT_MPI_ISEND: {
        READ(data, cursor, uint32_t, receiver);
        READ(data, cursor, uint32_t, communicator);
        READ(data, cursor, uint32_t, msgTag);
        READ(data, cursor, uint64_t, msgLength);
        if (data->record == pallas::PALLAS_EVENT_MPI_ISEND)
            READ(data, cursor, uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_RECV:
    case pallas::PALLAS_EVENT_MPI_IRECV: {
        READ(data, cursor, uint32_t, sender);
        READ(data, cursor, uint32_t, communicator);
        READ(data, cursor, uint32_t, msgTag);
        READ(data, cursor, uint64_t, msgLength);
        if (data->record == pallas::PALLAS_EVENT_MPI_IRECV)
            READ(data, cursor, uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE:
    case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
        READ(data, cursor, uint64_t, requestID);
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK:
    case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK:
    case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK:
    case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK: {
        READ(data, cursor, uint32_t, lockID);
        READ(data, cursor, uint32_t, acquisitionOrder);
        break;
    }
    case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END: {
        READ(data, cursor, uint32_t, collectiveOp);
        READ(data, cursor, uint32_t, communicator);
        READ(data, cursor, uint32_t, root);
        READ(data, cursor, uint64_t, sizeSent);
        READ(data, cursor, uint64_t, sizeReceived);
        break;
    }
    case pallas::PALLAS_EVENT_OMP_TASK_CREATE:
    case pallas::PALLAS_EVENT_OMP_TASK_SWITCH:
    case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE: {
        READ(data, cursor, uint64_t, taskID);
        break;
    }
    case pallas::PALLAS_EVENT_GENERIC: {
        pallas::StringRef event_name;
        pallas_event_pop_data(data, &event_name, sizeof(event_name), &cursor);
        dict["event_name"] = event_name;
        break;
    }
    case pallas::PALLAS_EVENT_THREAD_BEGIN:
    case pallas::PALLAS_EVENT_THREAD_END:
    case pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN:
    case pallas::PALLAS_EVENT_THREAD_TEAM_END:
    case pallas::PALLAS_EVENT_THREAD_JOIN:
    case pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
    case pallas::PALLAS_EVENT_THREAD_TASK_CREATE:
    case pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE:
    case pallas::PALLAS_EVENT_THREAD_TASK_SWITCH:
        // No additional data for these events
    default:
        break;
    }

    return dict;
}
