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

py::dict &EventData_get_data(pallas::EventData *data) {
    auto &dict = *new py::dict();
#define ADD_TO_DICT(name) dict[#name] = name
    byte *cursor = nullptr;

    switch (data->record) {
        case pallas::PALLAS_EVENT_ENTER: {
            pallas::RegionRef region_ref;
            pallas::pallas_read_enter(data, nullptr, &region_ref);
            ADD_TO_DICT(region_ref);
            break;
        }
        case pallas::PALLAS_EVENT_LEAVE: {
            pallas::RegionRef region_ref;
            pallas::pallas_read_leave(data, nullptr, &region_ref);
            ADD_TO_DICT(region_ref);
            break;
        }
        case pallas::PALLAS_EVENT_THREAD_FORK: {
            uint32_t numberOfRequestThreads;
            pallas::pallas_read_thread_fork(data, nullptr, &numberOfRequestThreads);
            ADD_TO_DICT(numberOfRequestThreads);
            break;
        }
        case pallas::PALLAS_EVENT_OMP_FORK: {
            uint32_t numberOfRequestThreads;
            pallas::pallas_read_omp_fork(data, nullptr, &numberOfRequestThreads);
            ADD_TO_DICT(numberOfRequestThreads);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_SEND: {
            uint32_t receiver, communicator, msgTag;
            uint64_t msgLength;
            pallas::pallas_read_mpi_send(data, nullptr,& receiver,& communicator, &msgTag, &msgLength);
            ADD_TO_DICT(receiver);
            ADD_TO_DICT(communicator);
            ADD_TO_DICT(msgTag);
            ADD_TO_DICT(msgLength);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_ISEND: {
            uint32_t receiver, communicator, msgTag;
            uint64_t msgLength, requestID;
            pallas::pallas_read_mpi_isend(data, nullptr,& receiver,& communicator, &msgTag, &msgLength, &requestID);
            ADD_TO_DICT(receiver);
            ADD_TO_DICT(communicator);
            ADD_TO_DICT(msgTag);
            ADD_TO_DICT(msgLength);
            ADD_TO_DICT(requestID);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_RECV: {
            uint32_t sender, communicator, msgTag;
            uint64_t msgLength;
            pallas::pallas_read_mpi_recv(data, nullptr, &sender, &communicator, &msgTag, &msgLength);
            ADD_TO_DICT(sender);
            ADD_TO_DICT(communicator);
            ADD_TO_DICT(msgTag);
            ADD_TO_DICT(msgLength);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_IRECV: {
            uint32_t sender, communicator, msgTag;
            uint64_t msgLength, requestID;
            pallas::pallas_read_mpi_irecv(data, nullptr, &sender, &communicator, &msgTag, &msgLength, &requestID);
            ADD_TO_DICT(sender);
            ADD_TO_DICT(communicator);
            ADD_TO_DICT(msgTag);
            ADD_TO_DICT(msgLength);
            ADD_TO_DICT(requestID);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE: {
            uint64_t requestID;
            pallas::pallas_read_mpi_isend_complete(data, nullptr, &requestID);
            ADD_TO_DICT(requestID);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
            uint64_t requestID;
            pallas::pallas_read_mpi_irecv_request(data, nullptr, &requestID);
            ADD_TO_DICT(requestID);
            break;
        }
        case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK: {
            uint32_t lockID, acquisitionOrder;
            pallas::pallas_read_thread_acquire_lock(data, nullptr, &lockID, &acquisitionOrder);
            ADD_TO_DICT(lockID);
            ADD_TO_DICT(acquisitionOrder);
            break;
        }
        case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK: {
            uint32_t lockID, acquisitionOrder;
            pallas::pallas_read_thread_release_lock(data, nullptr, &lockID, &acquisitionOrder);
            ADD_TO_DICT(lockID);
            ADD_TO_DICT(acquisitionOrder);
            break;
        }
        case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK: {
            uint32_t lockID, acquisitionOrder;
            pallas::pallas_read_omp_acquire_lock(data, nullptr, &lockID, &acquisitionOrder);
            ADD_TO_DICT(lockID);
            ADD_TO_DICT(acquisitionOrder);
            break;
        }
        case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK: {
            uint32_t lockID, acquisitionOrder;
            pallas::pallas_read_omp_release_lock(data, nullptr, &lockID, &acquisitionOrder);
            ADD_TO_DICT(lockID);
            ADD_TO_DICT(acquisitionOrder);
            break;
        }
        case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END: {
            uint32_t collectiveOp, communicator, root;
            uint64_t sizeSent, sizeReceived;
            pallas::pallas_read_mpi_collective_end(data, nullptr, &collectiveOp, &communicator, &root, &sizeSent, &sizeReceived);
            ADD_TO_DICT(collectiveOp);
            ADD_TO_DICT(communicator);
            ADD_TO_DICT(root);
            ADD_TO_DICT(sizeSent);
            ADD_TO_DICT(sizeReceived);
            break;
        }
        case pallas::PALLAS_EVENT_OMP_TASK_CREATE: {
            uint64_t taskID;
            pallas::pallas_read_omp_task_create(data, nullptr, &taskID);
            ADD_TO_DICT(taskID);
            break;
        }
        case pallas::PALLAS_EVENT_OMP_TASK_SWITCH: {
            uint64_t taskID;
            pallas::pallas_read_omp_task_switch(data, nullptr, &taskID);
            ADD_TO_DICT(taskID);
            break;
        }
        case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE: {
            uint64_t taskID;
            pallas::pallas_read_omp_task_complete(data, nullptr, &taskID);
            ADD_TO_DICT(taskID);
            break;
        }
        case pallas::PALLAS_EVENT_GENERIC: {
            pallas::StringRef event_name;
            pallas::pallas_read_generic(data, nullptr, &event_name);
            ADD_TO_DICT(event_name);
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
