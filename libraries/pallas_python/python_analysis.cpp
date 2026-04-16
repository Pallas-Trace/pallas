#include "python_analysis.h"

#include <iostream>
#include <bitset>
#include "pallas_python.h"
#include <regex>
extern py::module pandas;

py::array_t<uint64_t> get_communication_matrix(pallas::GlobalArchive &trace) {
    size_t size = trace.nb_archives * trace.nb_archives;
    size_t datasize = sizeof(uint64_t);
    auto *matrix = new uint64_t[size]();
    py::capsule free_when_done(matrix, [](void *f) {
        auto *matrix = reinterpret_cast<uint64_t *>(f);
        delete[] matrix;
    });

    for (auto &thread: trace.getThreadList()) {
        auto &receiver = thread->archive->id;
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto &event = thread->events[i];
            if (!IS_MPI_RECV(event)) {
                continue;
            }
            uint32_t sender = *(uint32_t *) &event.data.event_data[0];
            uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
            matrix[sender * trace.nb_archives + receiver] += msgLength * event.nb_occurrences;
        }
    }
    return py::array_t<uint64_t>(
        {trace.nb_archives, trace.nb_archives},
        {trace.nb_archives * datasize, datasize},
        matrix,
        free_when_done);
}

py::array_t<uint64_t> get_communication_matrix_timed(pallas::GlobalArchive &trace, pallas_timestamp_t start,
                                                     pallas_timestamp_t end) {
    size_t size = trace.nb_archives * trace.nb_archives;
    size_t datasize = sizeof(uint64_t);
    auto *matrix = new uint64_t[size]();
    py::capsule free_when_done(matrix, [](void *f) {
        auto *matrix = reinterpret_cast<uint64_t *>(f);
        delete[] matrix;
    });

    for (auto &thread: trace.getThreadList()) {
        auto &pid = thread->archive->id;
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto &event = thread->events[i];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            if (event.timestamps->back() < start || event.timestamps->front() > end)
                continue;
            size_t count = 0;
            for (size_t j = event.timestamps->getFirstOccurrenceBefore(start); j < event.nb_occurrences; j++) {
                auto ts = event.timestamps->at(j);
                if (ts < start)
                    continue;
                if (end < ts)
                    break;
                count++;
            }
            if (IS_MPI_RECV(event)) {
                uint32_t sender = *(uint32_t *) &event.data.event_data[0];
                uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
                matrix[sender * trace.nb_archives + pid] += msgLength * count;
            } else {
                uint32_t receiver = *(uint32_t *) &event.data.event_data[0];
                uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
                matrix[pid * trace.nb_archives + receiver] += msgLength * count;
            }
        }
    }
    return py::array_t<uint64_t>(
        {trace.nb_archives, trace.nb_archives},
        {trace.nb_archives * datasize, datasize},
        matrix,
        free_when_done);
}

std::map<uint64_t, uint64_t> get_message_size_histogram(pallas::GlobalArchive &trace, bool count_data_amount) {
    std::map<uint64_t, uint64_t> output;
    for (auto &thread: trace.getThreadList()) {
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto &event = thread->events[i];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
            if (!output.contains(msgLength)) {
                output[msgLength] = 0;
            }
            output[msgLength] += count_data_amount ? msgLength : 1;
        }
    }
    return output;
}

std::map<uint64_t, uint64_t> get_message_size_histogram_local(pallas::Archive &archive, bool count_data_amount) {
    std::map<uint64_t, uint64_t> output;
    for (auto &loc: archive.locations) {
        auto *thread = archive.getThread(loc.id);
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto &event = thread->events[i];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            // TODO Mayyyyybe we should allow for segregation between Recv, IRecv, Send, and ISend
            uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
            if (!output.contains(msgLength)) {
                output[msgLength] = 0;
            }
            output[msgLength] += count_data_amount ? msgLength : 1;
        }
    }
    return output;
}

py::array_t<uint64_t> get_communication_over_time(pallas::GlobalArchive &trace, py::array_t<uint64_t> timestamps,
                                                  bool count_messages) {
    // Warning: timestamps are bins, meaning the return is one size smaller than the actually value
    size_t n_bins = timestamps.size() - 1;
    auto output_numpy = py::array_t<uint64_t>(n_bins);
    uint64_t *output = (uint64_t *) output_numpy.request().ptr;
    std::memset(output, 0, sizeof(uint64_t) * n_bins);
    for (auto &thread: trace.getThreadList()) {
        for (size_t eid = 0; eid < thread->nb_events; eid++) {
            auto &event = thread->events[eid];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
            size_t last_occurrence = event.timestamps->getFirstOccurrenceBefore(timestamps.at(0));
            for (size_t i = 0; i < n_bins; i++) {
                pallas_timestamp_t start = timestamps.at(i);
                pallas_timestamp_t end = timestamps.at(i + 1);
                if (event.timestamps->back() < start)
                    break;
                if (event.timestamps->front() > end)
                    continue;
                size_t count = 0;
                // TODO This might be optimized if end-start is "long enough" ( ie more than one subvector length )
                //      This is optimized enough tho
                for (; last_occurrence < event.nb_occurrences; last_occurrence++) {
                    auto ts = event.timestamps->at(last_occurrence);
                    if (ts < start)
                        continue;
                    if (end < ts)
                        break;
                    count++;
                }
                output[i] += count * (count_messages ? 1 : msgLength);
            }
            // event.timestamps->free_data();
        }
    }
    return output_numpy;
}


py::array_t<uint64_t> get_communication_over_time_archive(pallas::Archive &archive, py::array_t<uint64_t> timestamps,
                                                          bool count_messages) {
    // Warning: timestamps are bins, meaning the return is one size smaller than the actually value
    size_t n_bins = timestamps.size() - 1;
    auto output_numpy = py::array_t<uint64_t>(n_bins);
    uint64_t *output = (uint64_t *) output_numpy.request().ptr;
    std::memset(output, 0, sizeof(uint64_t) * n_bins);
    for (auto &loc: archive.locations) {
        auto *thread = archive.getThread(loc.id);
        for (size_t eid = 0; eid < thread->nb_events; eid++) {
            auto &event = thread->events[eid];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            uint64_t msgLength = *(uint64_t *) &event.data.event_data[sizeof(uint32_t) * 3];
            size_t last_occurrence = event.timestamps->getFirstOccurrenceBefore(timestamps.at(0));
            for (size_t i = 0; i < n_bins; i++) {
                pallas_timestamp_t start = timestamps.at(i);
                pallas_timestamp_t end = timestamps.at(i + 1);
                if (event.timestamps->back() < start)
                    break;
                if (event.timestamps->front() > end)
                    continue;
                size_t count = 0;
                // TODO This might be optimized if end-start is "long enough" ( ie more than one subvector length )
                //      This is optimized enough tho
                for (; last_occurrence < event.nb_occurrences; last_occurrence++) {
                    auto ts = event.timestamps->at(last_occurrence);
                    if (ts < start)
                        continue;
                    if (end < ts)
                        break;
                    count++;
                }
                output[i] += count * (count_messages ? 1 : msgLength);
            }
            event.timestamps->free_data();
        }
    }
    return output_numpy;
}


py::object get_sequences_statistics(pallas::Thread &thread) {
    // 2. Créer un dictionnaire Python
    size_t nb_columns = 6;
    size_t nb_lines = thread.nb_sequences;

    py::array_t<SequenceStatisticsLine> test_numpy_array(nb_lines);
    py::list name_list(nb_lines);

    for (size_t i = 0; i < nb_lines; i++) {
        auto &s = thread.sequences[i];
        auto &line = test_numpy_array.mutable_at(i);
        line.sequence_id = s.id.id;
        name_list[i] = py::str(s.guessName(&thread));
        line.min = s.durations->min;
        line.mean = s.durations->mean;
        line.max = s.durations->max;
        line.nb_occurrences = s.durations->size;
    }


    // 3. Importer pandas et créer le DataFrame
    py::object df = pandas.attr("DataFrame")(test_numpy_array);
    df["name"] = name_list;

    return df;
}

enum mpi_message_status {
    status_none = 0,
    status_isend_occured = 1 << 0,
    status_swait_started = 1 << 1,
    status_swait_ended = 1 << 2,
    status_irecv_occured = 1 << 3,
    status_rwait_started = 1 << 4,
    status_rwait_ended = 1 << 5,
    status_invalid
};

int status_complete
        = (status_isend_occured | status_swait_started | status_swait_ended
           | status_irecv_occured | status_rwait_started | status_rwait_ended);

struct MPIMessage : MPIMessageLine {
    /** MPI_Request passed to MPI_ISend. */
    uint64_t isend_ptr = 0;
    /** MPI_Request passed to MPI_IRecv. */
    uint64_t irecv_ptr = 0;
    /** Status of that message.*/
    uint8_t status = status_none;

    MPIMessage(uint32_t sender, uint32_t receiver, uint32_t tag, uint64_t len, int status)
        : status(status) {
        static std::atomic<uint32_t> next_id = 1;
        this->id = next_id++;
        this->sender = sender;
        this->receiver = receiver;
        this->tag = tag;
        this->msg_length = len;
    }
};

std::ostream &operator <<(std::ostream &os, const MPIMessage &msg) {
    std::bitset<8> bitset(msg.status);
    return (os << "{" << msg.id <<
            "," << msg.sender <<
            "," << msg.receiver <<
            "," << msg.tag <<
            "," << msg.msg_length <<
            "," << bitset << "}");
}

enum send_receive { send, recv };

struct MPIMatchedMessage {
    MPIMessage *message;
    enum send_receive sr; // Is it a send, or a receive ?
    MPIMatchedMessage(MPIMessage *message, enum send_receive sr)
        : message(message), sr(sr) {
    }
};

/** MPI Request that was initiated but not yet completed by MPI_Wait. */
struct MpiRequest {
    /** MPI_Request. May be NULL. */
    uint64_t ptr = 0;
    /** Timestamp of call to MPI_IRecv. */
    pallas_timestamp_t ts = 0;
    /** Message corresponding to that request. */
    MPIMessage *message = nullptr;

    ~MpiRequest() {
        delete message;
    }
    MpiRequest(uint64_t ptr, pallas_timestamp_t ts, MPIMessage* msg): ptr(ptr), ts(ts), message(msg) {};
};

struct MPIProcessData {
    std::list<MPIMessage *> pending_smessages;
    std::list<MPIMessage *> pending_rmessages;
    std::list<MpiRequest *> pending_requests;
    std::list<MPIMatchedMessage> matched_messages;
};

#define READ(event_occurrence, cursor, type, name)                              \
    type name;                                                                  \
    pallas_event_pop_data(event_occurrence.event, &name, sizeof(type), &cursor);

static int local_rank_to_global(pallas::GlobalArchive &trace, uint32_t communicator, uint32_t local_rank) {
    return local_rank;
}

static std::map<pallas::LocationGroupId, MPIProcessData> processes;


static void update_message_timestamps(MPIProcessData &p, MPIMessage *m, int status,
                                      pallas_timestamp_t ts, std::vector<MPIMessageLine> *completed_messages) {
    if (status & status_isend_occured) {
        m->isend_ts = ts;
        m->status |= status_isend_occured;
    }
    if (status & status_swait_started) {
        m->start_swait_ts = ts;
        m->status |= status_swait_started;
        p.matched_messages.emplace_back(m, send);
        auto &lst = processes[m->sender].pending_smessages;
        assert(std::ranges::find(lst, m) != lst.end());
    }
    if (status & status_swait_ended) {
        m->end_swait_ts = ts;
        m->status |= status_swait_ended;
    }

    if (status & status_irecv_occured) {
        m->irecv_ts = ts;
        m->status |= status_irecv_occured;
    }
    if (status & status_rwait_started) {
        m->start_rwait_ts = ts;
        m->status |= status_rwait_started;

        p.matched_messages.emplace_back(m, recv);
        auto &lst = processes[m->receiver].pending_rmessages;
        assert(std::ranges::find(lst, m) != lst.end());
    }
    if (status & status_rwait_ended) {
        m->end_rwait_ts = ts;
        m->status |= status_rwait_ended;
    }

    if (m->status == status_complete) {
        /* Remove the message from the pending message list*/
        size_t count = processes[m->sender].pending_smessages.remove(m);
        count += processes[m->receiver].pending_rmessages.remove(m);
        /* Add the message to the completed message list */
        if (count > 0) {
            completed_messages->emplace_back(*m);
        }
    }
}

static void process_leave_mpi_wait(MPIProcessData &p, pallas_timestamp_t ts,
                                   std::vector<MPIMessageLine> *completed_messages) {
    for (auto &mm: p.matched_messages) {
        if (mm.sr == recv) {
            update_message_timestamps(p, mm.message, status_rwait_ended, ts, completed_messages);
        } else {
            update_message_timestamps(p, mm.message, status_swait_ended, ts, completed_messages);
        }
    }
    std::ranges::remove_if(p.matched_messages, [](const MPIMatchedMessage &mm) {
        return mm.message->status == status_complete;
    });
}

static MPIMessage *match_isend(uint32_t sender,
                               uint32_t receiver,
                               uint32_t msg_tag,
                               uint64_t msg_length,
                               pallas_timestamp_t isend_ts,
                               uint64_t isend_req,
                               int status,
                               std::vector<MPIMessageLine> *completed_messages
) {
    MPIProcessData &p_sender = processes[sender];
    MPIProcessData &p_receiver = processes[receiver];
    for (auto m: p_receiver.pending_rmessages) {
        if ((m->sender == sender || m->sender == UINT32_MAX) &&
            (m->tag == msg_tag || msg_tag == UINT32_MAX) &&
            !(m->status & status)) {
            m->status = m->status | status;
            m->sender = sender;
            m->tag = msg_tag;
            m->isend_ptr = isend_req;
            update_message_timestamps(p_sender, m, status, isend_ts, completed_messages);
            return m;
        }
    }
    MPIMessage *m = new MPIMessage(sender, receiver, msg_tag, msg_length, status);
    p_sender.pending_smessages.push_back(m);
    update_message_timestamps(p_sender, m, status, isend_ts, completed_messages);
    return m;
}

static MPIMessage *match_irecv(uint32_t sender,
                               uint32_t receiver,
                               uint32_t msg_tag,
                               uint64_t msg_length,
                               pallas_timestamp_t irecv_ts,
                               uint64_t irecv_req,
                               int status,
                               std::vector<MPIMessageLine> *completed_messages
) {
    MPIProcessData &p_sender = processes[sender];
    MPIProcessData &p_receiver = processes[receiver];
    for (auto m: p_sender.pending_smessages) {
        if ((m->receiver == receiver || m->receiver == UINT32_MAX) &&
            (m->tag == msg_tag || msg_tag == UINT32_MAX) &&
            !(m->status & status)) {
            m->status = m->status | status;
            m->receiver = receiver;
            m->tag = msg_tag;
            m->irecv_ptr = irecv_req;
            update_message_timestamps(p_receiver, m, status, irecv_ts, completed_messages);
            return m;
        }
    }
    MPIMessage *m = new MPIMessage(sender, receiver, msg_tag, msg_length, status);
    p_receiver.pending_rmessages.push_back(m);
    update_message_timestamps(p_receiver, m, status, irecv_ts, completed_messages);
    return m;
}

py::object get_mpi_message_list(pallas::GlobalArchive &trace) {
    // Il faut construire un dataframe de liste de message
    // Avec ou sans matching, de la forme suivante:
    // id,sender,receiver,tag,msg_length,isend_ts,start_swait_ts,end_swait_ts,irecv_ts,start_rwait_ts,end_rwait_ts
    auto *completed_messages = new std::vector<MPIMessageLine>();
    for (auto lgid: trace.location_groups) {
        processes[lgid.id] = MPIProcessData();
    }
    pallas_timestamp_t first_timestamp = trace.get_starting_timestamp();
    pallas_timestamp_t last_timestamp = trace.get_ending_timestamp();
    pallas_duration_t duration = last_timestamp - first_timestamp;

    pallas::MultiThreadReader reader = pallas::MultiThreadReader(trace);
    for (auto t = reader.getNextToken(); t != pallas::INVALID_TOKEN; t = reader.getNextToken()) {
        if (t.type != pallas::TypeEvent) {
            continue;
        }
        if (PyErr_CheckSignals() != 0) {
            processes.clear();
            delete completed_messages;
            throw py::error_already_set();
        }
        auto *cur_reader = reader.current_thread_reader;
        auto lgid = reader.current_thread_reader->archive->id;
        auto data = cur_reader->getEventOccurence(t, cur_reader->getCurrentTokenCount(t));
        // static uint progress_counter = 0;
        // if (progress_counter == 0) {
        //     float percent = (static_cast<float>(data.timestamp - first_timestamp) * 100) / (float) duration;
        //     std::cout << std::fixed << std::setprecision(2) << percent << "%\r";
        //     progress_counter = 1000;
        // }
        // progress_counter --;
        MPIProcessData &p = processes[lgid];
        byte *cursor = nullptr;
        switch (data.event->record) {
            case pallas::PALLAS_EVENT_MPI_SEND: {
                // An MpiSend record indicates that an MPI send operation was initiated
                // (MPI_SEND). It keeps the necessary information for this event: receiver of
                // the message, communicator, and the message tag. You can optionally add
                // further information like the message length (size of the send buffer).
                READ(data, cursor, uint32_t, receiver);
                READ(data, cursor, uint32_t, communicator);
                READ(data, cursor, uint32_t, msgTag);
                READ(data, cursor, uint64_t, msgLength);
                receiver = local_rank_to_global(trace, communicator, receiver);
                match_isend(lgid, receiver,
                            msgTag, msgLength, data.timestamp, 0,
                            status_isend_occured | status_swait_started, completed_messages);
                break;
            }
            case pallas::PALLAS_EVENT_MPI_RECV: {
                READ(data, cursor, uint32_t, sender);
                READ(data, cursor, uint32_t, communicator);
                READ(data, cursor, uint32_t, msgTag);
                READ(data, cursor, uint64_t, msgLength);
                sender = local_rank_to_global(trace, communicator, sender);
                match_irecv(sender, lgid,
                            msgTag, msgLength, data.timestamp, 0,
                            status_irecv_occured | status_rwait_started, completed_messages);
                break;
            }
            case pallas::PALLAS_EVENT_MPI_ISEND: {
                READ(data, cursor, uint32_t, receiver);
                READ(data, cursor, uint32_t, communicator);
                READ(data, cursor, uint32_t, msgTag);
                READ(data, cursor, uint64_t, msgLength);
                READ(data, cursor, uint64_t, requestID);
                receiver = local_rank_to_global(trace, communicator, receiver);
                auto m = match_isend(lgid, receiver,
                                     msgTag, msgLength, data.timestamp, requestID,
                                     status_isend_occured, completed_messages);
                auto *r = new MpiRequest(requestID, data.timestamp, m);
                p.pending_requests.push_back(r);
                break;
            }
            case pallas::PALLAS_EVENT_MPI_IRECV: {
                READ(data, cursor, uint32_t, sender);
                READ(data, cursor, uint32_t, communicator);
                READ(data, cursor, uint32_t, msgTag);
                READ(data, cursor, uint64_t, msgLength);
                READ(data, cursor, uint64_t, requestID);
                sender = local_rank_to_global(trace, communicator, sender);
                auto &lst = p.pending_requests;
                auto r = std::ranges::find_if(lst, [requestID](const MpiRequest *it) {
                    return it->ptr == requestID;
                });
                if (r != lst.end()) {
                    auto m = match_irecv(sender, lgid,
                                         msgTag, msgLength, data.timestamp, requestID,
                                         status_irecv_occured, completed_messages);
                    if ((*r)->message != nullptr) {
                        pallas_assert_equals((*r)->message, m);
                    }
                    (*r)->message = m;
                    update_message_timestamps(p, m, status_rwait_started,
                                              cur_reader->currentState.currentFrame[-1].current_timestamp
                                              // Enter timestamp
                                              , completed_messages);
                    update_message_timestamps(p, m, status_rwait_ended, data.timestamp, completed_messages);
                    size_t c = p.pending_requests.remove(*r);
                    pallas_assert_equals(c, 1);
                } else {
                    pallas_error("This should not have happened\n");
                }
                break;
            }
            case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE: {
                READ(data, cursor, uint64_t, requestID);
                auto &lst = p.pending_requests;
                auto r = std::ranges::find_if(lst, [requestID](const MpiRequest *it) {
                    return it->ptr == requestID;
                });
                if (r != lst.end() && (*r)->message != nullptr) {
                    update_message_timestamps(
                        p,
                        (*r)->message,
                        status_swait_started,
                        cur_reader->currentState.currentFrame[-1].current_timestamp // Enter timestamp
                        , completed_messages);
                    update_message_timestamps(
                        p,
                        (*r)->message,
                        status_swait_ended,
                        data.timestamp
                        , completed_messages);
                    size_t c = p.pending_requests.remove(*r);
                    pallas_assert_equals(c, 1);
                } else {
                    pallas_error("This should not have happened\n");
                }
                break;
            }

            case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
                READ(data, cursor, uint64_t, requestID);
                auto *r = new MpiRequest(requestID, data.timestamp, nullptr);
                p.pending_requests.push_back(r);
                break;
            }
            case pallas::PALLAS_EVENT_LEAVE: {
                std::string f(cur_reader->thread_trace->getRegionStringFromEvent(data.event));
                static std::set<std::string> implicit_mpi_wait{
                    "MPI_Send", "mpi_send_",
                    "MPI_Recv", "mpi_recv_",
                    "MPI_Bsend", "mpi_bsend_",
                    "MPI_Ssend", "mpi_ssend_",
                    "MPI_Rsend", "mpi_rsend_",
                    "MPI_Sendrecv", "mpi_sendrecv_",
                    "MPI_Sendrecv_replace", "mpi_sendrecv_replace_",
                    "MPI_Test", "mpi_test_",
                    "MPI_Wait", "mpi_wait_",
                    "MPI_Waitany", "mpi_waitany_",
                    "MPI_Testany", "mpi_testany_",
                    "MPI_Waitsome", "mpi_waitsome_",
                    "MPI_Testsome", "mpi_testsome_",
                    "MPI_Probe", "mpi_probe_",
                };
                if (implicit_mpi_wait.contains(f)) {
                    process_leave_mpi_wait(p, data.timestamp, completed_messages);
                }
                break;
            }
            default:
                break;
        }
    }

    py::capsule free_when_done(completed_messages, [](void *f) {
        auto foo = reinterpret_cast<std::vector<MPIMessageLine> *>(f);
        delete foo;
    });
    py::array_t<MPIMessageLine> numpy_array(
        {completed_messages->size()},
        {sizeof(MPIMessageLine)},
        completed_messages->data(),
        free_when_done
    );
    py::object df = pandas.attr("DataFrame")(numpy_array);
    processes.clear();
    return df;
}
