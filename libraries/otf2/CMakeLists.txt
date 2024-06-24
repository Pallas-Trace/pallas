cmake_minimum_required(VERSION 3.14.0)

# POLICIES #####################
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

#####################################################
# otf2-config

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/otf2_config.in
        ${CMAKE_CURRENT_BINARY_DIR}/otf2-config
        @ONLY # ask cmake to change only variables in the form @VAR@ (ie. don't touch the ${VAR} variables, which can be problematic for bash scripts)
)

list(APPEND SCRIPTS
        ${CMAKE_CURRENT_BINARY_DIR}/otf2-config
)

#####################################################
# otf2-print

add_executable(otf2-print src/otf2_print.c)

target_compile_options(otf2-print
        PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused -Wno-unused-parameter
)

target_link_libraries(otf2-print
        PRIVATE
        pallas
        otf2
)


#####################################################
# otf2-estimator

add_executable(otf2-estimator src/otf2_estimator.cpp)

target_compile_options(otf2-estimator
        PRIVATE
        -Wall -Wextra -Wpedantic
)

target_link_libraries(otf2-estimator
        PRIVATE
        pallas
        otf2
)

target_include_directories(otf2-estimator PRIVATE include)


#####################################################
# libotf2

add_library(otf2 SHARED)

set(OTF2_HEADERS
        include/otf2/OTF2_Archive.h
        include/otf2/OTF2_AttributeList.h
        include/otf2/OTF2_AttributeValue.h
        include/otf2/OTF2_Callbacks.h
        include/otf2/otf2_compiler.h
        include/otf2/OTF2_Definitions.h
        include/otf2/OTF2_DefReaderCallbacks.h
        include/otf2/OTF2_DefReader.h
        include/otf2/OTF2_DefWriter.h
        include/otf2/OTF2_ErrorCodes.h
        include/otf2/OTF2_Events.h
        include/otf2/OTF2_EventSizeEstimator.h
        include/otf2/OTF2_EvtReaderCallbacks.h
        include/otf2/OTF2_EvtReader.h
        include/otf2/OTF2_EvtWriter.h
        include/otf2/OTF2_GeneralDefinitions.h
        include/otf2/OTF2_GlobalDefReaderCallbacks.h
        include/otf2/OTF2_GlobalDefReader.h
        include/otf2/OTF2_GlobalDefWriter.h
        include/otf2/OTF2_GlobalEvtReaderCallbacks.h
        include/otf2/OTF2_GlobalEvtReader.h
        include/otf2/OTF2_GlobalSnapReaderCallbacks.h
        include/otf2/OTF2_GlobalSnapReader.h
        include/otf2/otf2.h
        include/otf2/OTF2_IdMap.h
        include/otf2/OTF2_Marker.h
        include/otf2/OTF2_MarkerReaderCallbacks.h
        include/otf2/OTF2_MarkerReader.h
        include/otf2/OTF2_MarkerWriter.h
        include/otf2/OTF2_MPI_Collectives.h
        include/otf2/OTF2_OpenMP_Locks.h
        include/otf2/OTF2_Pthread_Locks.h
        include/otf2/OTF2_Reader.h
        include/otf2/OTF2_SnapReaderCallbacks.h
        include/otf2/OTF2_SnapReader.h
        include/otf2/OTF2_SnapWriter.h
        include/otf2/OTF2_Thumbnail.h
)

target_sources(
        otf2
        PRIVATE
        src/otf2.c
        src/OTF2_Archive.c
        src/OTF2_AttributeList.c
        src/OTF2_AttributeValue.c
        src/OTF2_DefReaderCallbacks.c
        src/OTF2_DefReader.c
        src/OTF2_DefWriter.c
        src/OTF2_ErrorCodes.c
        src/OTF2_EventSizeEstimator.c
        src/OTF2_EvtReaderCallbacks.c
        src/OTF2_EvtReader.c
        src/OTF2_EvtWriter.c
        src/OTF2_GlobalDefReaderCallbacks.c
        src/OTF2_GlobalDefReader.c
        src/OTF2_GlobalDefWriter.c
        src/OTF2_GlobalEvtReaderCallbacks.c
        src/OTF2_GlobalEvtReader.c
        src/OTF2_GlobalSnapReaderCallbacks.c
        src/OTF2_GlobalSnapReader.c
        src/OTF2_IdMap.c
        src/OTF2_MarkerReaderCallbacks.c
        src/OTF2_MarkerReader.c
        src/OTF2_MarkerWriter.c
        src/OTF2_Reader.c
        src/OTF2_SnapReaderCallbacks.c
        src/OTF2_SnapReader.c
        src/OTF2_SnapWriter.c
        src/OTF2_Thumbnail.c
        PUBLIC
        ${OTF2_HEADERS}
)

set_property(TARGET otf2
        PROPERTY PUBLIC_HEADER ${OTF2_HEADERS})

target_compile_options(otf2
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wno-unused-parameter -Wno-unused
        -Wall -Wextra -Wno-sign-compare
)

target_include_directories(otf2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} include)

target_link_libraries(otf2 PRIVATE pallas)

install(TARGETS otf2 EXPORT OTF2LibTargets
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}/otf2)

install(
        TARGETS otf2-print otf2-estimator
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        RUNTIME DESTINATION ${INSTALL_BINDIR}
        INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
)

install(PROGRAMS ${SCRIPTS} DESTINATION ${INSTALL_BINDIR})
