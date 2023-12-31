cmake_minimum_required(VERSION 3.14.0)

# POLICIES #####################
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/htf/htf_config.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/htf/htf_config.h
        @ONLY
)

add_library(htf SHARED)

set(HTF_HEADERS
        include/htf/htf_archive.h
        include/htf/htf_attribute.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/htf/htf_config.h
        include/htf/htf_dbg.h
        include/htf/htf.h
        include/htf/htf_hash.h
        include/htf/htf_linked_vector.h
	include/htf/htf_parameter_handler.h
        include/htf/htf_read.h
        include/htf/htf_storage.h
        include/htf/htf_timestamp.h
        include/htf/htf_write.h
)

target_sources(htf
        PRIVATE
        src/htf.cpp
        src/htf_archive.cpp
        src/htf_attribute.cpp
        src/htf_dbg.cpp
        src/htf_hash.cpp
        src/htf_read.cpp
        src/htf_storage.cpp
        src/htf_timestamp.cpp
        src/htf_write.cpp
        src/htf_linked_vector.cpp
        src/htf_parameter_handler.cpp
        PUBLIC
        ${HTF_HEADERS}
)

target_include_directories(htf
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        include
        ${JSONCPP_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}/include/htf/
)
if (ZFP_FOUND)
    target_include_directories(htf PUBLIC ${ZFP_INCLUDE_DIRS})
    target_link_libraries(htf PRIVATE ${ZFP_LIBRARIES})
endif ()


if (SZ_FOUND)
    target_include_directories(htf PUBLIC ${SZ_INCLUDE_DIRS})
    target_link_libraries(htf PRIVATE ${SZ_LIBRARIES})
endif ()

set_property(TARGET htf
        PROPERTY PUBLIC_HEADER ${HTF_HEADERS})


target_compile_options(htf
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wall -Wextra -Wno-sign-compare -Wno-unused -Wno-class-memaccess
)

target_link_libraries(htf
        PUBLIC
        dl
        PRIVATE
        atomic
        ${CMAKE_DL_LIBS}
        rt
        m
        zstd
        ${zstd_LIBRARIES}
        ${JSONCPP_LIBRARIES}
)
if (DOXYGEN_FOUND AND BUILD_DOC)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")

    # note the option ALL which allows to build the docs together with the application
    doxygen_add_docs(htf_doc ALL
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating API documentation with Doxygen"
			CONFIG_FILE ${DOXYGEN_OUT})

    install(DIRECTORY "${DOXYGEN_OUTPUT_DIRECTORY}" TYPE DOC)
else ()
    if (NOT DOXYGEN_FOUND)
        message("Doxygen need to be installed to generate the doxygen documentation")
    endif ()
endif ()

install(TARGETS htf EXPORT HTFLibTargets
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}/htf)
