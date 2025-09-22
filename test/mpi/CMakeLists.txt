cmake_minimum_required(VERSION 3.9)

find_package(MPI REQUIRED)

SET(N_MPI 2)
SET(N_THREADS 3)
SET(N_THREADS_TOTAL 6)
SET(N_ITER 2000)


add_executable(mpi_benchmark mpi_benchmark.c)
target_link_libraries(mpi_benchmark PUBLIC pallas MPI::MPI_C)
target_include_directories(mpi_benchmark PUBLIC MPI::MPI_C_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/pallas/include)
add_test(NAME mpi_benchmark COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${N_MPI} ${MPIEXEC_PREFLAGS}
        ./mpi_benchmark -n ${N_ITER} -t ${N_THREADS}
        ${MPIEXEC_POSTFLAGS})


SET(TRACE_NAME ${CMAKE_CURRENT_BINARY_DIR}/mpi_benchmark_trace/main.pallas)
add_test(NAME mpi_info_benchmark COMMAND pallas_info ${TRACE_NAME})
add_test(NAME mpi_print_benchmark COMMAND pallas_print ${TRACE_NAME})
add_test(NAME mpi_print_benchmark_structure COMMAND pallas_print -S ${TRACE_NAME})
add_test(NAME mpi_print_benchmark_thread COMMAND pallas_print -T ${TRACE_NAME})

add_test (mpi_benchmark_checks bash
        "${CMAKE_CURRENT_SOURCE_DIR}/../write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS_TOTAL})
set_tests_properties(mpi_info_benchmark mpi_print_benchmark_thread mpi_print_benchmark mpi_print_benchmark_structure PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(mpi_benchmark_checks PROPERTIES
        REQUIRED "mpi_info_benchmark;mpi_print_benchmark;mpi_print_benchmark_structure")

