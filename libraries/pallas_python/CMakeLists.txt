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

option(Python3_INSTALL_LOCALLY "Install the Pallas Python library in the user local directory instead of system-wide" ON)

execute_process(COMMAND pybind11-config --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)


pybind11_add_module(pallas_python MODULE pallas_python.cpp)
set_target_properties(pallas_python PROPERTIES OUTPUT_NAME "pallas_trace")
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

if (Python3_INSTALL_LOCALLY)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m site --user-site OUTPUT_VARIABLE Python3_USERSITELIB OUTPUT_STRIP_TRAILING_WHITESPACE)
    message("Installing Python library locally to ${Python3_USERSITELIB}")
    install(TARGETS pallas_python DESTINATION ${Python3_USERSITELIB})
else()
    execute_process(COMMAND ${Python3_EXECUTABLE}  -c "import site; print(site.getsitepackages()[0])" OUTPUT_VARIABLE Python3_USERSITELIB OUTPUT_STRIP_TRAILING_WHITESPACE)
    message("Installing Python library globally to ${Python3_SITELIB}")
    install(TARGETS pallas_python DESTINATION ${Python3_SITELIB})
endif()
