cmake_minimum_required(VERSION 3.13)

enable_testing()

if (ENABLE_MPI_TEST)
    add_subdirectory(mpi)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-sign-compare ${WARNINGS})
include_directories(${CMAKE_SOURCE_DIR}/src/)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(pallas Threads::Threads)


#####################################################
# write_benchmark

add_executable(write_benchmark write_benchmark.c)
add_executable(write_benchmark_CPP write_benchmark.cpp)
add_executable(test_snapshot test_snapshot.cpp)
add_executable(write_pattern write_pattern.cpp)

SET(TRACE_NAME ${CMAKE_CURRENT_BINARY_DIR}/write_benchmark_trace/main.pallas)
SET(TRACE_NO_COMP_NAME ${CMAKE_CURRENT_BINARY_DIR}/write_benchmark_trace_None/main.pallas)
SET(CPP_TRACE_NAME ${CMAKE_CURRENT_BINARY_DIR}/write_benchmark_CPP_trace/main.pallas)
SET(PATTERN_TRACE_NAME ${CMAKE_CURRENT_BINARY_DIR}/write_pattern_trace/main.pallas)

SET(N_THREADS 4)
SET(N_ITER 2000)
# Twice the size of linked-vectors

add_test(NAME write_benchmark COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS})
add_test(NAME info_benchmark COMMAND pallas_info ${TRACE_NAME})
add_test(NAME print_benchmark COMMAND pallas_print ${TRACE_NAME})
add_test(NAME print_benchmark_structure COMMAND pallas_print -S ${TRACE_NAME})
add_test(NAME print_benchmark_thread COMMAND pallas_print -T ${TRACE_NAME})
add_test(NAME test_snapshot COMMAND test_snapshot ${TRACE_NAME} 10)
add_test(NAME edit_benchmark COMMAND pallas_editor -c None ${TRACE_NAME})

add_test (benchmark_checks bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark print_benchmark_thread print_benchmark print_benchmark_structure edit_benchmark test_snapshot PROPERTIES
        REQUIRED_FILES ${TRACE_NAME}
        DEPENDS write_benchmark
)
set_tests_properties(benchmark_checks PROPERTIES
        REQUIRED_FILES ${TRACE_NAME}
        DEPENDS "write_benchmark;info_benchmark;print_benchmark;print_benchmark_structure;print_benchmark_thread;edit_benchmark;test_snapshot"
)

add_test(NAME info_edited_benchmark COMMAND pallas_info ${TRACE_NO_COMP_NAME})
add_test(NAME print_edited_benchmark COMMAND pallas_print ${TRACE_NO_COMP_NAME})
add_test(NAME print_edited_benchmark_structure COMMAND pallas_print -S ${TRACE_NO_COMP_NAME})
add_test(NAME print_edited_benchmark_thread COMMAND pallas_print -T ${TRACE_NO_COMP_NAME})

add_test (edited_benchmark_checks bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NO_COMP_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_edited_benchmark print_edited_benchmark_thread print_edited_benchmark_structure print_edited_benchmark PROPERTIES
        REQUIRED_FILES ${TRACE_NO_COMP_NAME}
        DEPENDS edit_benchmark
)
set_tests_properties(edited_benchmark_checks PROPERTIES
        REQUIRED_FILES ${TRACE_NO_COMP_NAME}
        DEPENDS "info_edited_benchmark;print_edited_benchmark;print_edited_benchmark_structure;print_edited_benchmark_thread"
)


add_test(NAME write_benchmark_CPP COMMAND write_benchmark_CPP -n ${N_ITER} -t ${N_THREADS})
add_test(NAME info_benchmark_CPP COMMAND pallas_info ${CPP_TRACE_NAME})
add_test(NAME print_benchmark_CPP COMMAND pallas_print ${CPP_TRACE_NAME})
add_test(NAME print_benchmark_structure_CPP COMMAND pallas_print -S ${CPP_TRACE_NAME})
add_test(NAME print_benchmark_thread_CPP COMMAND pallas_print -T ${CPP_TRACE_NAME})

add_test (benchmark_checks_CPP bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${CPP_TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark_CPP print_benchmark_thread_CPP print_benchmark_CPP print_benchmark_structure_CPP PROPERTIES
        REQUIRED_FILES ${CPP_TRACE_NAME}
        DEPENDS write_benchmark_CPP
)
set_tests_properties(benchmark_checks_CPP PROPERTIES
        REQUIRED_FILES ${CPP_TRACE_NAME}
        DEPENDS "info_benchmark_CPP;print_benchmark_CPP;print_benchmark_structure_CPP;print_benchmark_thread_CPP"
)


add_test(NAME write_benchmark_logical COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS} -l)

add_test(NAME info_benchmark_logical COMMAND pallas_info ${TRACE_NAME})
add_test(NAME print_benchmark_logical COMMAND pallas_print ${TRACE_NAME})
add_test(NAME print_benchmark_structure_logical COMMAND pallas_print -S ${TRACE_NAME})
add_test(NAME print_benchmark_thread_logical COMMAND pallas_print -T ${TRACE_NAME})


add_test (benchmark_checks_logical bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS} -l)

set_tests_properties(write_benchmark_logical PROPERTIES DEPENDS benchmark_checks)

set_tests_properties(info_benchmark_logical print_benchmark_thread_logical print_benchmark_logical print_benchmark_structure_logical PROPERTIES
        REQUIRED_FILES ${TRACE_NAME}
        DEPENDS write_benchmark_logical
)
set_tests_properties(benchmark_checks_logical PROPERTIES
        REQUIRED_FILES ${TRACE_NAME}
        DEPENDS "info_benchmark_logical;print_benchmark_logical;print_benchmark_structure_logical;print_benchmark_thread_logical"
)

add_test(NAME write_pattern_benchmark COMMAND write_pattern)
add_test(NAME info_pattern_benchmark COMMAND pallas_info ${PATTERN_TRACE_NAME})
add_test(NAME print_pattern_benchmark COMMAND pallas_print ${PATTERN_TRACE_NAME})
add_test(NAME print_pattern_benchmark_structure COMMAND pallas_print -S ${PATTERN_TRACE_NAME})
add_test(NAME print_pattern_benchmark_thread COMMAND pallas_print -T ${PATTERN_TRACE_NAME})

add_test (benchmark_checks bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark print_benchmark_thread print_benchmark print_benchmark_structure edit_benchmark test_snapshot PROPERTIES
        REQUIRED_FILES ${TRACE_NAME}
        DEPENDS write_benchmark
)
set_tests_properties(benchmark_checks PROPERTIES
        REQUIRED_FILES ${TRACE_NAME}
        DEPENDS "write_benchmark;info_benchmark;print_benchmark;print_benchmark_structure;print_benchmark_thread;edit_benchmark;test_snapshot"
)




add_executable(find_loop find_loop.cpp)
add_test(NAME find_loop COMMAND find_loop 50 100)

add_executable(sequence_duration sequence_durations.cpp)
add_test(NAME sequence_duration COMMAND sequence_duration)

add_executable(test_vector test_vector.cpp)
add_test(NAME test_vector COMMAND test_vector 100)

add_executable(test_hash test_hash.cpp)
#add_test(NAME test_hash COMMAND test_hash)

# Instantaneous
add_test(check_stark_bt.A.4 bash "${CMAKE_CURRENT_SOURCE_DIR}/test_stark_trace.sh" "${CMAKE_BINARY_DIR}" "bt.A.4")
# Takes 4 / 8 / 8 s
add_test(check_stark_bt.B.16 bash "${CMAKE_CURRENT_SOURCE_DIR}/test_stark_trace.sh" "${CMAKE_BINARY_DIR}" "bt.B.16")
# Takes 8 / 11 / 11 s
add_test(check_stark_amg.4.16 bash "${CMAKE_CURRENT_SOURCE_DIR}/test_stark_trace.sh" "${CMAKE_BINARY_DIR}" "amg.4.16")

if (GIT_BRANCH STREQUAL "main")
    # Takes 40 / 60 / 60 s
    add_test(check_stark_amg.16.4 bash "${CMAKE_CURRENT_SOURCE_DIR}/test_stark_trace.sh" "${CMAKE_BINARY_DIR}" "amg.16.4")
    # Takes 180 / 210 / 410 s
    add_test(check_stark_amg.64.1 bash "${CMAKE_CURRENT_SOURCE_DIR}/test_stark_trace.sh" "${CMAKE_BINARY_DIR}" "amg.64.1")
    # Takes 50 / 80 / 160 s
    add_test(check_stark_bt.C.64 bash "${CMAKE_CURRENT_SOURCE_DIR}/test_stark_trace.sh" "${CMAKE_BINARY_DIR}" "bt.C.64")
    set_tests_properties(check_stark_amg.16.4 check_stark_amg.64.1 check_stark_bt.C.64 PROPERTIES
            COST 50
    )
endif()

set_tests_properties(write_benchmark write_benchmark_CPP write_benchmark_logical PROPERTIES
        COST 100
)
set_tests_properties(check_stark_bt.B.16 check_stark_amg.4.16 PROPERTIES
        COST 150
)