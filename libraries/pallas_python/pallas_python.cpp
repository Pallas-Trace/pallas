/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
#include "pallas_python.h"
#include "python_tokens.h"
#include "python_read.h"
#include "python_analysis.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

py::module pandas;

void setupEnums(const py::module_& m) {
    py::enum_<pallas::TokenType>(m, "TokenType")
            .value("INVALID", pallas::TypeInvalid)
            .value("EVENT", pallas::TypeEvent)
            .value("SEQUENCE", pallas::TypeSequence)
            .value("LOOP", pallas::TypeLoop)
            .export_values();

    py::enum_<pallas::Record>(m, "Record")
            .value("BUFFER_FLUSH", pallas::PALLAS_EVENT_BUFFER_FLUSH)
            .value("MEASUREMENT_ON_OFF", pallas::PALLAS_EVENT_MEASUREMENT_ON_OFF)
            .value("ENTER", pallas::PALLAS_EVENT_ENTER)
            .value("LEAVE", pallas::PALLAS_EVENT_LEAVE)
            .value("MPI_SEND", pallas::PALLAS_EVENT_MPI_SEND)
            .value("MPI_ISEND", pallas::PALLAS_EVENT_MPI_ISEND)
            .value("MPI_ISEND_COMPLETE", pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE)
            .value("MPI_IRECV_REQUEST", pallas::PALLAS_EVENT_MPI_IRECV_REQUEST)
            .value("MPI_RECV", pallas::PALLAS_EVENT_MPI_RECV)
            .value("MPI_IRECV", pallas::PALLAS_EVENT_MPI_IRECV)
            .value("MPI_REQUEST_TEST", pallas::PALLAS_EVENT_MPI_REQUEST_TEST)
            .value("MPI_REQUEST_CANCELLED", pallas::PALLAS_EVENT_MPI_REQUEST_CANCELLED)
            .value("MPI_COLLECTIVE_BEGIN", pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN)
            .value("MPI_COLLECTIVE_END", pallas::PALLAS_EVENT_MPI_COLLECTIVE_END)
            .value("OMP_FORK", pallas::PALLAS_EVENT_OMP_FORK)
            .value("OMP_JOIN", pallas::PALLAS_EVENT_OMP_JOIN)
            .value("OMP_ACQUIRE_LOCK", pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK)
            .value("OMP_RELEASE_LOCK", pallas::PALLAS_EVENT_OMP_RELEASE_LOCK)
            .value("OMP_TASK_CREATE", pallas::PALLAS_EVENT_OMP_TASK_CREATE)
            .value("OMP_TASK_SWITCH", pallas::PALLAS_EVENT_OMP_TASK_SWITCH)
            .value("OMP_TASK_COMPLETE", pallas::PALLAS_EVENT_OMP_TASK_COMPLETE)
            .value("METRIC", pallas::PALLAS_EVENT_METRIC)
            .value("PARAMETER_STRING", pallas::PALLAS_EVENT_PARAMETER_STRING)
            .value("PARAMETER_INT", pallas::PALLAS_EVENT_PARAMETER_INT)
            .value("PARAMETER_UNSIGNED_INT", pallas::PALLAS_EVENT_PARAMETER_UNSIGNED_INT)
            .value("THREAD_FORK", pallas::PALLAS_EVENT_THREAD_FORK)
            .value("THREAD_JOIN", pallas::PALLAS_EVENT_THREAD_JOIN)
            .value("THREAD_TEAM_BEGIN", pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN)
            .value("THREAD_TEAM_END", pallas::PALLAS_EVENT_THREAD_TEAM_END)
            .value("THREAD_ACQUIRE_LOCK", pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK)
            .value("THREAD_RELEASE_LOCK", pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK)
            .value("THREAD_TASK_CREATE", pallas::PALLAS_EVENT_THREAD_TASK_CREATE)
            .value("THREAD_TASK_SWITCH", pallas::PALLAS_EVENT_THREAD_TASK_SWITCH)
            .value("THREAD_TASK_COMPLETE", pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE)
            .value("THREAD_CREATE", pallas::PALLAS_EVENT_THREAD_CREATE)
            .value("THREAD_BEGIN", pallas::PALLAS_EVENT_THREAD_BEGIN)
            .value("THREAD_WAIT", pallas::PALLAS_EVENT_THREAD_WAIT)
            .value("THREAD_END", pallas::PALLAS_EVENT_THREAD_END)
            .value("IO_CREATE_HANDLE", pallas::PALLAS_EVENT_IO_CREATE_HANDLE)
            .value("IO_DESTROY_HANDLE", pallas::PALLAS_EVENT_IO_DESTROY_HANDLE)
            .value("IO_DUPLICATE_HANDLE", pallas::PALLAS_EVENT_IO_DUPLICATE_HANDLE)
            .value("IO_SEEK", pallas::PALLAS_EVENT_IO_SEEK)
            .value("IO_CHANGE_STATUS_FLAGS", pallas::PALLAS_EVENT_IO_CHANGE_STATUS_FLAGS)
            .value("IO_DELETE_FILE", pallas::PALLAS_EVENT_IO_DELETE_FILE)
            .value("IO_OPERATION_BEGIN", pallas::PALLAS_EVENT_IO_OPERATION_BEGIN)
            .value("IO_OPERATION_TEST", pallas::PALLAS_EVENT_IO_OPERATION_TEST)
            .value("IO_OPERATION_ISSUED", pallas::PALLAS_EVENT_IO_OPERATION_ISSUED)
            .value("IO_OPERATION_COMPLETE", pallas::PALLAS_EVENT_IO_OPERATION_COMPLETE)
            .value("IO_OPERATION_CANCELLED", pallas::PALLAS_EVENT_IO_OPERATION_CANCELLED)
            .value("IO_ACQUIRE_LOCK", pallas::PALLAS_EVENT_IO_ACQUIRE_LOCK)
            .value("IO_RELEASE_LOCK", pallas::PALLAS_EVENT_IO_RELEASE_LOCK)
            .value("IO_TRY_LOCK", pallas::PALLAS_EVENT_IO_TRY_LOCK)
            .value("PROGRAM_BEGIN", pallas::PALLAS_EVENT_PROGRAM_BEGIN)
            .value("PROGRAM_END", pallas::PALLAS_EVENT_PROGRAM_END)
            .value("NON_BLOCKING_COLLECTIVE_REQUEST", pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_REQUEST)
            .value("NON_BLOCKING_COLLECTIVE_COMPLETE", pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_COMPLETE)
            .value("COMM_CREATE", pallas::PALLAS_EVENT_COMM_CREATE)
            .value("COMM_DESTROY", pallas::PALLAS_EVENT_COMM_DESTROY)
            .value("GENERIC", pallas::PALLAS_EVENT_GENERIC)
            .export_values();
}

PYBIND11_MODULE(_core, m) {
    PYBIND11_NUMPY_DTYPE(SequenceStatisticsLine, sequence_id, min, mean, max, nb_occurrences);
    pandas = py::module::import("pandas");
    m.doc() = "Python API for the Pallas library";

    setupEnums(m);

    py::class_<pallas::Token>(m, "Token", "A Pallas token")
            .def_property_readonly("id", [](pallas::Token t) { return t.id; })
            .def_property_readonly("type", [](pallas::Token t) { return t.type; })
            .def("__repr__", &Token_toString)
            .def("__hash__", [](pallas::Token& self) { return *reinterpret_cast<uint32_t*>(&self); })
            .def("__eq__", [](pallas::Token& self, pallas::Token& other) { return self.type == other.type && self.id == other.id; });

    py::class_<PyLinkedVector>(m, "Vector", "A Pallas custom vector")
            .def_property_readonly("size", [](PyLinkedVector self) { return self.linked_vector ? self.linked_vector->size : self.linked_duration_vector->size; })
            .def("__getitem__", [](PyLinkedVector self, int i) { return self.linked_vector ? self.linked_vector->at(i) : self.linked_duration_vector->at(i); })
            .def("__iter__", [](const PyLinkedVector self) { return PyLinkedVectorIterator{self.linked_vector, self.linked_duration_vector, 0}; })
            .def("as_numpy_array", &linked_vector_to_numpy);

    py::class_<PyLinkedVectorIterator>(m, "Vector_Iterator", "An iterator over a Pallas custom vector")
        .def("__next__", [](PyLinkedVectorIterator& self) {
            if (self.linked_vector) {
                if (self.index < self.linked_vector->size) {
                    return self.linked_vector->at(self.index++);
                }
            } else {
                if (self.index < self.linked_duration_vector->size) {
                    return self.linked_duration_vector->at(self.index++);
                }
            }
            throw py::stop_iteration();
        });

    py::class_<PySequence>(m, "Sequence", "A Pallas Sequence, ie a group of tokens.")
            .def_property_readonly("id", [](const PySequence& self) { return self.self->id; })
            .def_property_readonly("tokens", [](const PySequence& self) { return self.self->tokens; })
            .def_property_readonly("content", [](const PySequence& self) { return sequenceGetContent(self); })
            .def_property_readonly("n_iterations", [](const PySequence& self) { return self.self->durations->size; })
            .def_property_readonly("timestamps", [](const PySequence& self) { return PyLinkedVector{self.self->timestamps, nullptr}; })
            .def_property_readonly("durations", [](const PySequence& self) { return PyLinkedVector{nullptr, self.self->durations}; })
            .def_property_readonly("exclusive_durations", [](const PySequence& self) { return PyLinkedVector{nullptr, self.self->exclusive_durations}; })
            .def_property_readonly("max_duration", [](const PySequence& self) { return self.self->durations->max; })
            .def_property_readonly("min_duration", [](const PySequence& self) { return self.self->durations->min; })
            .def_property_readonly("mean_duration", [](const PySequence& self) { return self.self->durations->mean; })
            .def("contains", [](const PySequence& self, const PySequence& other) { return doesSequenceContains(self, other.self->id); })
            .def("contains", [](const PySequence& self, const PyLoop& other) { return doesSequenceContains(self, other.self->self_id); })
            .def("contains", [](const PySequence& self, const PyEvent& other) { return doesSequenceContains(self, {pallas::TokenType::TypeEvent, other.self->id}); })
            .def("contains", [](const PySequence& self, const pallas::Token& other) { return doesSequenceContains(self, other); })
            .def("guessName", [](const PySequence& self) { return self.self->guessName(self.thread); })
            .def("__repr__", [](const PySequence& self) { return "<pallas_python.Sequence " + std::to_string(self.self->id.id) + ">"; });

    py::class_<PyLoop>(m, "Loop", "A Pallas Loop, ie a repetition of a Sequence token.")
            .def_property_readonly("id", [](const PyLoop& self) { return self.self->self_id; })
            .def_property_readonly("sequence", [](const PyLoop& self) { return PySequence{self.thread->getSequence(self.self->repeated_token), self.thread}; })
            .def_property_readonly("nb_iterations", [](const PyLoop& self) { return self.self->nb_iterations; })
            .def("__repr__", [](const PyLoop& self) { return "<pallas_python.Loop " + std::to_string(self.self->self_id.id) + ">"; });

    py::class_<PyEvent>(m, "Event", "A Pallas Event.")
            .def_property_readonly("id", [](const PyEvent& self) { return pallas::Token(pallas::TypeEvent, self.self->id); })
            .def_property_readonly("record", [](const PyEvent& self) { return self.self->data.record; })
            .def_property_readonly("data", [](const PyEvent& self) { return EventData_get_data(&self.self->data); })
            .def_property_readonly("nb_occurrences", [](const PyEvent& self) { return self.self->nb_occurrences; })
            .def_property_readonly("timestamps", [](const PyEvent& self) { return PyLinkedVector{self.self->timestamps, nullptr}; })
            .def("guessName", [](const PyEvent& self) { return self.thread->getEventString(&self.self->data); })
            .def("__repr__", [](const PyEvent& self) { return "<pallas_python.Event " + std::to_string(self.self->id) + ">"; });

    py::class_<pallas::Thread>(m, "Thread", "A Pallas thread.")
            .def_readonly("id", &pallas::Thread::id)
            .def_property_readonly("starting_timestamp", [](const pallas::Thread& self) { return self.first_timestamp; })
            .def_property_readonly("finish_timestamp", [](const pallas::Thread& self) { return (self.first_timestamp + self.sequences[0].durations->at(0)); })
            .def_property_readonly("events", [](pallas::Thread& self) { return threadGetEvents(self); })
            .def_property_readonly("sequences", [](pallas::Thread& self) { return threadGetSequences(self); })
            .def_property_readonly("loops", [](pallas::Thread& self) { return threadGetLoops(self); })
            .def("get_events_from_record", threadGetEventsMatching)
            .def("get_events_from_record", threadGetEventsMatchingList)
            .def("__repr__", [](const pallas::Thread& self) { return "<pallas_python.Thread " + std::to_string(self.id) + ">"; })
            .def("getSnapshotView", &pallas::Thread::getSnapshotView)
            .def("getSnapshotViewFast", &pallas::Thread::getSnapshotViewFast)
            .def("__iter__", [](const pallas::Thread& self) {
                auto inner = pallas::ThreadReader(self.archive, self.id, PALLAS_READ_FLAG_UNROLL_ALL);
                return PyThreadIterator{inner};
            })
            .def("reader", [](const pallas::Thread& self) {
                return new pallas::ThreadReader(self.archive, self.id, PALLAS_READ_FLAG_UNROLL_ALL);
            })
    ;

    py::class_<PyThreadIterator>(m, "Thread_Iterator", "An iterator over the thread.")
            .def("__next__", [](PyThreadIterator& self) {
                if (self.inner.moveToNextToken()) {
                    auto t = self.inner.pollCurToken();
                    if (!t.isValid()) {
                        throw py::stop_iteration();
                    }
                    return makePyObjectFromToken(t, self.inner);
                }
                throw py::stop_iteration();
            });

    py::class_<pallas::ThreadReader>(m, "ThreadReader", "A helper structure to read a thread")
            .def_property_readonly("callstack", &thread_reader_get_callstack)
            .def("moveToNextToken", [](pallas::ThreadReader& self, bool enter_sequence = true, bool enter_loop = true) {
                int flags = get_read_flags_from_bools(enter_sequence, enter_loop);
                self.moveToNextToken(flags);
            })
            .def("pollCurToken", [](pallas::ThreadReader& self) {
                return makePyObjectFromToken(self.pollCurToken(), self);
            })
            .def("enterIfStartOfBlock", [](pallas::ThreadReader& self, bool enter_sequence = true, bool enter_loop = true) {
                int flags = get_read_flags_from_bools(enter_sequence, enter_loop);
                return self.enterIfStartOfBlock(flags);
            })
            .def("exitIfEndOfBlock", [](pallas::ThreadReader& self, bool exit_sequence = true, bool exit_loop = true) {
                int flags = get_read_flags_from_bools(exit_sequence, exit_loop);
                return self.exitIfEndOfBlock(flags);
            })
            .def("isEndOfCurrentBlock", &pallas::ThreadReader::isEndOfCurrentBlock)
            .def("isEndOfTrace", &pallas::ThreadReader::isEndOfTrace);

    py::class_<PyLocationGroup>(m, "LocationGroup", "A group of Pallas locations. Usually means a process.")
            .def_readonly("id", &PyLocationGroup::id)
            .def_readonly("name", &PyLocationGroup::name)
            .def_readonly("parent", &PyLocationGroup::parent)
            .def("__repr__", [](const PyLocationGroup& self) { return "<pallas_python.LocationGroup " + std::to_string(self.id) + ": '" + self.name + "'>"; });

    py::class_<PyLocation>(m, "Location", "A Pallas location. Usually means an execution thread.")
            .def_readonly("id", &PyLocation::id)
            .def_readonly("name", &PyLocation::name)
            .def_readonly("parent", &PyLocation::parent)
            .def("__repr__", [](const PyLocation& self) { return "<pallas_python.Location " + std::to_string(self.id) + ": '" + self.name + "'>"; });

    py::class_<PyRegion>(m, "Region", "A Pallas region.")
            .def_readonly("id", &PyRegion::id)
            .def_readonly("name", &PyRegion::name)
            .def("__repr__", [](const PyRegion& self) {
                return "<pallas_python.Region " + std::to_string(self.id) + ": '" + self.name + "'>";
            });

    py::class_<pallas::Archive>(m, "Archive", "A Pallas archive. If it exists, it's already been loaded.")
            .def_readonly("dir_name", &pallas::Archive::dir_name)
            .def_readonly("id", &pallas::Archive::id)
            .def_readonly("metadata", &pallas::Archive::metadata)
            .def_property_readonly("locations", &Archive_get_locations)
            .def_property_readonly("strings", &Archive_get_strings)
            .def_property_readonly("regions", &Archive_get_regions)
            .def_property_readonly("threads", &Archive_get_threads);

    m.def("open_trace", &open_trace, "Open a Pallas trace")
            .def("get_ABI", []() { return PALLAS_ABI_VERSION; })
            .def("get_communication_matrix", get_communication_matrix,
                 "Returns an MPI communication matrix for given trace.\n"
                 "Doesn't read more than the grammar.\n")
            .def("get_communication_matrix", get_communication_matrix_timed,
                 "Returns an MPI communication matrix for given trace between the given timestamps.\n"
                 "Doesn't read more than the grammar.\n")
            .def("get_communication_matrix", [](pallas::GlobalArchive& trace, double_t start, double_t end) {
                     // Enables autoconvert from double to uint64, in case of linspace
                     return get_communication_matrix_timed(trace, start, end);
                 }, "Returns an MPI communication matrix for given trace between the given timestamps.\n"
                 "Doesn't read more than the grammar.\n")
            .def("get_message_size_histogram", get_message_size_histogram, py::arg("trace"), py::kw_only(), py::arg("count_data_amount") = false,
                 "Returns a histogram of the message sizes sent in this trace.\n"
                 ":param count_data_amount: If true, the histogram doesn't count the number of messages, but the amount of data sent.")
            .def("get_message_size_histogram", get_message_size_histogram_local, py::arg("archive"), py::kw_only(), py::arg("count_data_amount") = false,
                 "Returns a histogram of the message sizes sent in this archive.\n"
                 ":param count_data_amount: If true, the histogram doesn't count the number of messages, but the amount of data sent.")
            .def("get_communication_over_time", get_communication_over_time, py::arg("trace"), py::arg("timestamps"),
                 py::kw_only(), py::arg("count_messages") = false,
                 "Returns a binned histogram for the given timestamps.\n"
                 ":param timestamps: Bins of timestamps. Beware that the last given timestamp is the end of the last bin.\n"
                 ":param count_messages: If False, count the number of messages rather than the data amount.")
            .def("get_communication_over_time", get_communication_over_time_archive, py::arg("archive"), py::arg("timestamps"),
                 py::kw_only(), py::arg("count_messages") = false,
                 "Returns a binned histogram for the given timestamps.\n"
                 ":param timestamps: Bins of timestamps. Beware that the last given timestamp is the end of the last bin.\n"
                 ":param count_messages: If False, count the number of messages rather than the data amount.")
            .def("get_sequences_statistics", get_sequences_statistics);

    py::class_<pallas::GlobalArchive>(m, "Trace", "A Pallas Trace file.")
            .def(py::init(&open_trace), "Open a trace file and read its structure.")
            .def_readonly("dir_name", &pallas::GlobalArchive::dir_name)
            .def_readonly("trace_name", &pallas::GlobalArchive::trace_name)
            .def_readonly("fullpath", &pallas::GlobalArchive::fullpath)
            .def_readonly("metadata", &pallas::GlobalArchive::metadata)
            .def_property_readonly("locations", &Trace_get_locations)
            .def_property_readonly("location_groups", &Trace_get_location_groups)
            .def_property_readonly("strings", &Trace_get_strings)
            .def_property_readonly("regions", &Trace_get_regions)
            .def_property_readonly("archives", &Trace_get_archives)
            .def_property_readonly("starting_timestamp", &pallas::GlobalArchive::get_starting_timestamp)
            .def_property_readonly("ending_timestamp", &pallas::GlobalArchive::get_ending_timestamp);
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
