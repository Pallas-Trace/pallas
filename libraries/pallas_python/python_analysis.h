/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_record.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct SequenceStatisticsLine {
    uint32_t sequence_id;
    pallas_duration_t min;
    pallas_duration_t mean;
    pallas_duration_t max;
    uint64_t nb_occurrences;
};

struct MPIMessageLine {
    /** Message ID. Simply used as an identifier in the grand scheme of things. */
    uint32_t id = -1;
    /** Sender of the message. -1 if matching hasn't been done. */
    uint32_t sender = -1;
    /** Receiver of the message. -1 if matching hasn't been done. */
    uint32_t receiver = -1;

    /** Tag of the message. */
    uint32_t tag = -1;
    /** Message length ( bytes ). */
    uint64_t msg_length = -1;
    /** Timestamp of the Send/ISend call. - 1 if matching hasn't been done.*/
    pallas_timestamp_t isend_ts = -1;
    /** Start of the Wait call. - 1 if matching hasn't been done.*/
    pallas_timestamp_t start_swait_ts = -1;
    /** End of the Wait call. - 1 if matching hasn't been done.*/
    pallas_timestamp_t end_swait_ts = -1;
    /** Timestamp of the Recv/IRecv call. - 1 if matching hasn't been done.*/
    pallas_timestamp_t irecv_ts = -1;
    /** Start of the Wait call. - 1 if matching hasn't been done.*/
    pallas_timestamp_t start_rwait_ts = -1;
    /** End of the Wait call. - 1 if matching hasn't been done.*/
    pallas_timestamp_t end_rwait_ts = -1;
};

/** Returns a communication matrix of all the messages received. */
py::array_t<uint64_t> get_communication_matrix(pallas::GlobalArchive& trace);

/** Returns a communication matrix of all the messages exchanged between start and end. */
py::array_t<uint64_t> get_communication_matrix_timed(pallas::GlobalArchive& trace, pallas_timestamp_t start, pallas_timestamp_t end);

/** Returns a histogram ( in the form of a sorted map ) of all communications received. */
std::map<uint64_t, uint64_t> get_message_size_histogram(pallas::GlobalArchive& trace, bool count_data_amount = false);

/** Returns a histogram ( in the form of a sorted map ) of all communications in a single archive. */
std::map<uint64_t, uint64_t> get_message_size_histogram_local(pallas::Archive& archive, bool count_data_amount = false);

py::array_t<uint64_t> get_communication_over_time(pallas::GlobalArchive& trace, py::array_t<uint64_t> timestamps, bool count_messages = false);

py::array_t<uint64_t> get_communication_over_time_archive(pallas::Archive& archive, py::array_t<uint64_t> timestamps, bool count_messages = false);

py::object get_sequences_statistics(pallas::Thread& thread);

py::object get_mpi_message_list(pallas::GlobalArchive &trace);
