cmake_minimum_required(VERSION 3.13)

add_compile_options(-Wall -Wextra -Wpedantic -g -O0 ${WARNINGS})
link_libraries(pallas)

add_executable(pallas_print pallas_print.cpp)
add_executable(pallas_info pallas_info.cpp)
add_executable(pallas_editor pallas_editor.cpp)
add_executable(pallas_config pallas_config.cpp)
add_executable(pallas_sync pallas_sync.cpp)

install(
  TARGETS pallas_print pallas_info pallas_editor pallas_config pallas_sync
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${INSTALL_BINDIR}
  INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
)
