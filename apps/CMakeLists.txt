cmake_minimum_required(VERSION 3.13)

add_compile_options(-Wall -Wextra -Wpedantic)
include_directories(${CMAKE_SOURCE_DIR}/src/)
link_libraries(htf zstd)
configure_file(../config.json config.json COPYONLY)

add_executable(htf_print htf_print.cpp)
add_executable(htf_info htf_info.cpp)

install(
  TARGETS htf_print htf_info
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${INSTALL_BINDIR}
  INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
)
