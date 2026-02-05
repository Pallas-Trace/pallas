#include "python_analysis.h"
#include "pallas_python.h"

extern py::module pandas;

py::array_t<uint64_t> get_communication_matrix(pallas::GlobalArchive& trace) {
    size_t size = trace.nb_archives * trace.nb_archives;
    size_t datasize = sizeof(uint64_t);
    auto* matrix = new uint64_t[size]();
    py::capsule free_when_done(matrix, [](void* f) {
        auto* matrix = reinterpret_cast<uint64_t*>(f);
        delete[] matrix;
    });

    for (auto& thread : trace.getThreadList()) {
        auto& receiver = thread->archive->id;
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto& event = thread->events[i];
            if (!IS_MPI_RECV(event)) {
                continue;
            }
            uint32_t sender = *(uint32_t*)&event.data.event_data[0];
            uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
            matrix[sender * trace.nb_archives + receiver] += msgLength * event.nb_occurrences;
        }
    }
    return py::array_t<uint64_t>(
            {trace.nb_archives, trace.nb_archives},
            {trace.nb_archives * datasize, datasize},
            matrix,
            free_when_done);
}

py::array_t<uint64_t> get_communication_matrix_timed(pallas::GlobalArchive& trace, pallas_timestamp_t start, pallas_timestamp_t end) {
    size_t size = trace.nb_archives * trace.nb_archives;
    size_t datasize = sizeof(uint64_t);
    auto* matrix = new uint64_t[size]();
    py::capsule free_when_done(matrix, [](void* f) {
        auto* matrix = reinterpret_cast<uint64_t*>(f);
        delete[] matrix;
    });

    for (auto& thread : trace.getThreadList()) {
        auto& pid = thread->archive->id;
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto& event = thread->events[i];
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
                uint32_t sender = *(uint32_t*)&event.data.event_data[0];
                uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
                matrix[sender * trace.nb_archives + pid] += msgLength * count;
            } else {
                uint32_t receiver = *(uint32_t*)&event.data.event_data[0];
                uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
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

std::map<uint64_t, uint64_t> get_message_size_histogram(pallas::GlobalArchive& trace, bool count_data_amount) {
    std::map<uint64_t, uint64_t> output;
    for (auto& thread : trace.getThreadList()) {
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto& event = thread->events[i];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
            if (!output.contains(msgLength)) {
                output[msgLength] = 0;
            }
            output[msgLength] += count_data_amount ? msgLength : 1;
        }
    }
    return output;
}

std::map<uint64_t, uint64_t> get_message_size_histogram_local(pallas::Archive& archive, bool count_data_amount) {
    std::map<uint64_t, uint64_t> output;
    for (auto& loc : archive.locations) {
        auto* thread = archive.getThread(loc.id);
        for (size_t i = 0; i < thread->nb_events; i++) {
            auto& event = thread->events[i];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            // TODO Mayyyyybe we should allow for segregation between Recv, IRecv, Send, and ISend
            uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
            if (!output.contains(msgLength)) {
                output[msgLength] = 0;
            }
            output[msgLength] += count_data_amount ? msgLength : 1;
        }
    }
    return output;
}

py::array_t<uint64_t> get_communication_over_time(pallas::GlobalArchive& trace, py::array_t<uint64_t> timestamps, bool count_messages) {
    // Warning: timestamps are bins, meaning the return is one size smaller than the actually value
    size_t n_bins = timestamps.size() - 1;
    auto output_numpy = py::array_t<uint64_t>(n_bins);
    uint64_t* output = (uint64_t*)output_numpy.request().ptr;
    std::memset(output, 0, sizeof(uint64_t) * n_bins);
    for (auto& thread : trace.getThreadList()) {
        for (size_t eid = 0; eid < thread->nb_events; eid++) {
            auto& event = thread->events[eid];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
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


py::array_t<uint64_t> get_communication_over_time_archive(pallas::Archive& archive, py::array_t<uint64_t> timestamps, bool count_messages) {
    // Warning: timestamps are bins, meaning the return is one size smaller than the actually value
    size_t n_bins = timestamps.size() - 1;
    auto output_numpy = py::array_t<uint64_t>(n_bins);
    uint64_t* output = (uint64_t*)output_numpy.request().ptr;
    std::memset(output, 0, sizeof(uint64_t) * n_bins);
    for (auto& loc : archive.locations) {
        auto* thread = archive.getThread(loc.id);
        for (size_t eid = 0; eid < thread->nb_events; eid++) {
            auto& event = thread->events[eid];
            if (!IS_MPI_COMM(event)) {
                continue;
            }
            uint64_t msgLength = *(uint64_t*)&event.data.event_data[sizeof(uint32_t) * 3];
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


py::object get_sequences_statistics(pallas::Thread& thread) {
    // 2. Créer un dictionnaire Python
    size_t nb_columns = 6;
    size_t nb_lines = thread.nb_sequences;

    py::array_t<SequenceStatisticsLine> test_numpy_array(nb_lines);
    py::list name_list(nb_lines);

    for (size_t i = 0; i < nb_lines; i ++) {
        auto& s = thread.sequences[i];
        auto& line = test_numpy_array.mutable_at(i);
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
