cmake_minimum_required(VERSION 3.14.0)
set (CMAKE_CXX_STANDARD 20)

find_package(Python3 COMPONENTS Interpreter Development NumPy)

if (Python3_FOUND)
    if (Python3_VERSION VERSION_LESS 3.10)
        message("Python version was too low: ${Python3_VERSION} <= 3.10")
        message("Skipping Python Library Generation")
        return(-1)
    endif()
endif()
option(PythonInstallLocal "Install the Python Lib locally instead of globally" ON)
set(PythonLocalLib $ENV{HOME}/.local/lib/python3.${Python3_VERSION_MINOR}/site-packages)
set(PythonGlobalLib ${Python3_SITELIB})

execute_process(COMMAND pybind11-config --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)


pybind11_add_module(pallas_python MODULE pallas_python.cpp)
set_target_properties(pallas_python PROPERTIES OUTPUT_NAME "pallas")
pybind11_extension(pallas_python)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(pallas_python)
endif()

list(APPEND WARNINGS "-Wno-missing-field-initializers")

target_compile_options(pallas_python
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wall -Wextra ${WARNINGS}
)

target_include_directories(pallas_python PRIVATE ${Python3_INCLUDE_DIRS})
target_link_directories(pallas_python PRIVATE ${Python3_LIBRARY_DIRS})

target_link_libraries(pallas_python
        PRIVATE
        dl
        pallas
        ${Python3_LIBRARIES}
)

if (PythonInstallLocal)
    install(TARGETS pallas_python DESTINATION ${PythonLocalLib})
else()
    install(TARGETS pallas_python DESTINATION ${PythonGlobalLib})
endif()