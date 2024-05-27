cmake_minimum_required(VERSION 3.14.0)

# POLICIES #####################
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/pallas/pallas_config.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/pallas/pallas_config.h
        @ONLY
)

add_library(pallas SHARED)

set(PALLAS_HEADERS
        include/pallas/pallas_archive.h
        include/pallas/pallas_attribute.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/pallas/pallas_config.h
        include/pallas/pallas_dbg.h
        include/pallas/pallas.h
        include/pallas/pallas_hash.h
        include/pallas/pallas_linked_vector.h
	    include/pallas/pallas_parameter_handler.h
        include/pallas/pallas_read.h
        include/pallas/pallas_storage.h
        include/pallas/pallas_timestamp.h
        include/pallas/pallas_write.h
)

target_sources(pallas
        PRIVATE
        src/pallas.cpp
        src/pallas_archive.cpp
        src/pallas_attribute.cpp
        src/pallas_dbg.cpp
        src/pallas_hash.cpp
        src/pallas_read.cpp
        src/pallas_storage.cpp
        src/pallas_timestamp.cpp
        src/pallas_write.cpp
        src/pallas_linked_vector.cpp
        src/pallas_parameter_handler.cpp
        PUBLIC
        ${PALLAS_HEADERS}
)

target_include_directories(pallas
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        include
        ${ZSTD_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}/include/pallas/
)
if (ZFP_FOUND)
    target_include_directories(pallas PUBLIC ${ZFP_INCLUDE_DIRS})
    target_link_libraries(pallas PRIVATE ${ZFP_LIBRARIES})
endif ()


if (SZ_FOUND)
    target_include_directories(pallas PUBLIC ${SZ3_INCLUDE_DIRS})
    target_link_libraries(pallas PRIVATE ${SZ3_LIBRARIES})
endif ()

set_property(TARGET pallas
        PROPERTY PUBLIC_HEADER ${PALLAS_HEADERS})


target_compile_options(pallas
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wall -Wextra -Wno-sign-compare -Wno-unused -Wno-class-memaccess
)

target_link_libraries(pallas
        PUBLIC
        dl
        PRIVATE
        atomic
        ${CMAKE_DL_LIBS}
        rt
        m
        ${ZSTD_LIBRARIES}
        ${JSONCPP_LIBRARIES}
)
target_link_directories(pallas PRIVATE ${JSONCPP_LIBRARY_DIRS} ${ZSTD_LIBRARY_DIRS})
if (DOXYGEN_FOUND AND BUILD_DOC)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")

    # note the option ALL which allows to build the docs together with the application
    doxygen_add_docs(pallas_doc ALL
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating API documentation with Doxygen"
			CONFIG_FILE ${DOXYGEN_OUT})

    install(DIRECTORY "${DOXYGEN_OUTPUT_DIRECTORY}" TYPE DOC)
else ()
    if (NOT DOXYGEN_FOUND)
        message("Doxygen need to be installed to generate the doxygen documentation")
    endif ()
endif ()

install(TARGETS pallas EXPORT PALLASLibTargets
        LIBRARY DESTINATION ${INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}/pallas
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pallas.config DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/)
