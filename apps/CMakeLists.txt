cmake_minimum_required(VERSION 3.13)

add_compile_options(-Wall -Wextra -Wpedantic ${WARNINGS})
link_libraries(pallas)

add_executable(pallas_print pallas_print.cpp)
add_executable(pallas_info pallas_info.cpp)
add_executable(pallas_test_durations pallas_test_durations.cpp)
add_executable(pallas_editor pallas_editor.cpp)
add_executable(pallas_durations pallas_durations.cpp)
add_executable(pallas_getTimestamps pallas_getTimestamps.cpp)


install(
  TARGETS pallas_print pallas_info pallas_editor  pallas_durations pallas_getTimestamps
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${INSTALL_BINDIR}
  INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
)
