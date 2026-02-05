/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_record.h>
#include <pallas/utils/pallas_storage.h>

#define READ(data, cursor, type, name)                              \
    {                                                               \
        type name;                                                  \
        pallas_event_pop_data(data, &name, sizeof(type), &cursor);  \
        dict[#name] = name;                                         \
    }


#define IS_MPI_SEND(e) (e.data.record == pallas::PALLAS_EVENT_MPI_ISEND || \
e.data.record == pallas::PALLAS_EVENT_MPI_SEND )

#define IS_MPI_RECV(e) (e.data.record == pallas::PALLAS_EVENT_MPI_IRECV || \
e.data.record == pallas::PALLAS_EVENT_MPI_RECV )

#define IS_MPI_COMM(e) (IS_MPI_RECV(e) || IS_MPI_SEND(e))

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
