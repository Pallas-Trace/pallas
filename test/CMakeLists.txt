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

SET(TRACE_NAME ${CMAKE_CURRENT_BINARY_DIR}/write_benchmark_trace/main.pallas)
SET(N_THREADS 4)
SET(N_ITER 40)

add_test(NAME write_benchmark COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS})

add_test(NAME info_benchmark COMMAND pallas_info ${TRACE_NAME})
add_test(NAME print_benchmark COMMAND pallas_print ${TRACE_NAME})
add_test(NAME print_benchmark_structure COMMAND pallas_print -S ${TRACE_NAME})

add_test (benchmark_checks bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark print_benchmark print_benchmark_structure PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(benchmark_checks PROPERTIES
        REQUIRED "info_benchmark;print_benchmark;print_benchmark_structure")


add_test(NAME write_benchmark_CPP COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS})

add_test(NAME info_benchmark_CPP COMMAND pallas_info ${TRACE_NAME})
add_test(NAME print_benchmark_CPP COMMAND pallas_print ${TRACE_NAME})
add_test(NAME print_benchmark_structure_CPP COMMAND pallas_print -S ${TRACE_NAME})

add_test (benchmark_checks_CPP bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark_CPP print_benchmark_CPP print_benchmark_structure_CPP PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(benchmark_checks_CPP PROPERTIES
        REQUIRED "info_benchmark_CPP;print_benchmark_CPP;print_benchmark_structure_CPP")


add_test(NAME write_benchmark_logical COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS} -l)

add_test(NAME info_benchmark_logical COMMAND pallas_info ${TRACE_NAME})
add_test(NAME print_benchmark_logical COMMAND pallas_print ${TRACE_NAME})
add_test(NAME print_benchmark_structure_logical COMMAND pallas_print -S ${TRACE_NAME})

add_test (benchmark_checks_logical bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS} -l)


set_tests_properties(info_benchmark_logical print_benchmark_logical print_benchmark_structure_logical PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(benchmark_checks_logical PROPERTIES
        REQUIRED "info_benchmark_logical;print_benchmark_logical;print_benchmark_structure_logical")

add_executable(find_loop find_loop.cpp)
add_test(NAME find_loop COMMAND find_loop 50 100)

add_executable(sequence_duration sequence_durations.cpp)
add_test(NAME sequence_duration COMMAND sequence_duration)

add_executable(test_vector test_vector.cpp)
add_test(NAME test_vector COMMAND test_vector 100)

add_executable(test_hash test_hash.cpp)
#add_test(NAME test_hash COMMAND test_hash)
