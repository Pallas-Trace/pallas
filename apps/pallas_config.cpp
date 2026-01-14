//
// Created by khatharsis on 23/09/25.
//
#include <iostream>

#include "pallas_config.h"

int main() {
    std::cout << "Pallas tracing library v" << PALLAS_VERSION_MAJOR << "." << PALLAS_VERSION_MINOR << "." << PALLAS_VERSION_PATCH << std::endl
            << "\tABI = " << PALLAS_ABI_VERSION << std::endl
            << "\tGit Branch = " << PALLAS_BRANCH << std::endl
            << "\tGit Hash = " << PALLAS_COMMIT_HASH << std::endl
            << "\tVECTOR_SIZE = " << VECTOR_SIZE << std::endl
            << "\tMAP_SIZE = " << MAP_SIZE << std::endl
            << "\tUNO_MAP_SIZE = " << UNO_MAP_SIZE << std::endl
            << "\tTIMEPOINT_SIZE = " << TIMEPOINT_SIZE << std::endl
            << "\tNB_EVENT_DEFAULT = " << NB_EVENT_DEFAULT << std::endl
            << "\tNB_SEQUENCE_DEFAULT = " << NB_SEQUENCE_DEFAULT << std::endl
            << "\tNB_LOOP_DEFAULT = " << NB_LOOP_DEFAULT << std::endl
            << "\tNB_STRING_DEFAULT = " << NB_STRING_DEFAULT << std::endl
            << "\tNB_REGION_DEFAULT = " << NB_REGION_DEFAULT << std::endl
            << "\tNB_TIMESTAMP_DEFAULT = " << NB_TIMESTAMP_DEFAULT << std::endl
            << "\tNB_ATTRIBUTE_DEFAULT = " << NB_ATTRIBUTE_DEFAULT << std::endl
            << "\tSEQUENCE_SIZE_DEFAULT = " << SEQUENCE_SIZE_DEFAULT << std::endl
            << "\tLOOP_SIZE_DEFAULT = " << LOOP_SIZE_DEFAULT << std::endl
            << "\tCALLSTACK_DEPTH_DEFAULT = " << CALLSTACK_DEPTH_DEFAULT << std::endl
            << "\tNB_ARCHIVES_DEFAULT = " << NB_ARCHIVES_DEFAULT << std::endl
            << "\tNB_THREADS_DEFAULT = " << NB_THREADS_DEFAULT << std::endl
            << "\tNB_LOCATION_GROUPS_DEFAULT = " << NB_LOCATION_GROUPS_DEFAULT << std::endl
            << "\tNB_LOCATIONS_DEFAULT = " << NB_LOCATIONS_DEFAULT << std::endl
            << "\tPALLAS_CONFIG_PATH = " << PALLAS_CONFIG_PATH << std::endl;
}

/**
* #define PALLAS_VERSION_MAJOR    @CMAKE_PROJECT_VERSION_MAJOR@
#define PALLAS_VERSION_MINOR    @CMAKE_PROJECT_VERSION_MINOR@
#define PALLAS_VERSION_PATCH    @CMAKE_PROJECT_VERSION_PATCH@
#define PALLAS_VERSION          @CMAKE_PROJECT_VERSION@
#define PALLAS_ABI_VERSION      15

#ifndef VECTOR_SIZE
#define VECTOR_SIZE @SIZEOF_VECTOR@
#endif
#ifndef MAP_SIZE
#define MAP_SIZE @SIZEOF_MAP@
#endif
#ifndef UNO_MAP_SIZE
#define UNO_MAP_SIZE @SIZEOF_UNO_MAP@
#endif
#ifndef TIMEPOINT_SIZE
#define TIMEPOINT_SIZE @SIZEOF_TIMEPOINT@
#endif

#define NB_EVENT_DEFAULT 1000
#define NB_SEQUENCE_DEFAULT 1000
#define NB_LOOP_DEFAULT 1000
#define NB_STRING_DEFAULT 100
#define NB_REGION_DEFAULT 100
#define NB_TIMESTAMP_DEFAULT 1000
#define NB_ATTRIBUTE_DEFAULT 1000
#define SEQUENCE_SIZE_DEFAULT 1024
#define LOOP_SIZE_DEFAULT 16
#define CALLSTACK_DEPTH_DEFAULT 128
#define NB_ARCHIVES_DEFAULT 1
#define NB_THREADS_DEFAULT 16
#define NB_LOCATION_GROUPS_DEFAULT 16
#define NB_LOCATIONS_DEFAULT NB_THREADS_DEFAULT


#define PALLAS_CONFIG_PATH "@CMAKE_INSTALL_FULL_SYSCONFDIR@/pallas.config"
 */
