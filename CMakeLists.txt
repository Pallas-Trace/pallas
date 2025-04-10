cmake_minimum_required(VERSION 3.13.0)

project(Pallas
        VERSION 0.2.0
        LANGUAGES CXX C
)

# include CMake modules
include(CheckLibraryExists)
include(CheckLanguage)
include(CTest)
include(GNUInstallDirs)

# RPATH 
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_SKIP_BUILD_RPATH FALSE) # use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) # when building, don't use the install RPATH already (but later on when installing)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}") # the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# CMakeFind repository
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)
# C++ standard
set (CMAKE_CXX_STANDARD 17)

# Look for libraries

# Options to enable/disable modules
option(ENABLE_OTF2 "Enable OTF2 compatibility layer" ON)
option(BUILD_DOC "Build Doxygen Documentation" ON)
option(ENABLE_SZ "Enables compression using the SZ lib" ON)
option(ENABLE_ZFP "Enables compression using the ZFP lib" ON)
option(ENABLE_PYTHON "Enable the Python interface" ON)
option(ENABLE_MPI_TEST "MPI tests" ON)


find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD libzstd REQUIRED)
find_package(Doxygen)


INCLUDE(CheckTypeSize)
SET(CMAKE_EXTRA_INCLUDE_FILES "map")
check_type_size("std::map<uint32_t, size_t>" SIZEOF_MAP LANGUAGE CXX)
message("Size of std::map<uint32_t, size_t>: ${SIZEOF_MAP}")
add_compile_definitions(MAP_SIZE=${SIZEOF_MAP})

SET(CMAKE_EXTRA_INCLUDE_FILES "unordered_map")
check_type_size("std::unordered_map<uint32_t, size_t>" SIZEOF_UNO_MAP LANGUAGE CXX)
message("std::unordered_map<uint32_t, size_t>: ${SIZEOF_UNO_MAP}")
add_compile_definitions(UNO_MAP_SIZE=${SIZEOF_UNO_MAP})

SET(CMAKE_EXTRA_INCLUDE_FILES "vector")
check_type_size("std::vector<size_t>" SIZEOF_VECTOR LANGUAGE CXX)
message("Size of std::vector<size_t>: ${SIZEOF_VECTOR}")
add_compile_definitions(VECTOR_SIZE=${SIZEOF_VECTOR})

SET(CMAKE_EXTRA_INCLUDE_FILES "chrono")
check_type_size("std::chrono::time_point<std::chrono::high_resolution_clock>" SIZEOF_TIMEPOINT LANGUAGE CXX)
message("Size of TimePoint: ${SIZEOF_TIMEPOINT}")
add_compile_definitions(TIMEPOINT_SIZE=${SIZEOF_TIMEPOINT})

####################################################
# Some flags for compilation
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(WARNINGS
            -Wno-gnu-zero-variadic-macro-arguments
            -Wno-sign-compare
            -Wno-unused
            -Wno-unused-parameter
            -Wno-string-plus-int
            -Wno-return-type-c-linkage
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(WARNINGS
            -Wno-sign-compare
            -Wno-unused-function
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-deprecated-declarations)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(WARNINGS
            -Rno-debug-disables-optimization
            -Wno-return-type-c-linkage
            -Wno-sign-compare
            -Wno-unused-function
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-string-plus-int
            -Wno-gnu-zero-variadic-macro-arguments
    )
    ADD_LINK_OPTIONS(-Rno-debug-disables-optimization)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(WARNINGS)
endif()

set(prefix "${CMAKE_INSTALL_PREFIX}")
set(LIBRARY_PATH "${CMAKE_INSTALL_PREFIX}/lib")
if ("${APPLE}")
    set(DYNLIB_EXT "dylib")
    set(LD_PRELOAD_NAME "DYLD_INSERT_LIBRARIES")
    set(LD_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else () # By default -> Linux
    set(DYNLIB_EXT "so")
    set(LD_PRELOAD_NAME "LD_PRELOAD")
    set(LD_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif ()

if (ENABLE_ZFP)
    find_package(ZFP)
    if (ZFP_FOUND)
        add_compile_definitions(WITH_ZFP)
    endif ()
endif (ENABLE_ZFP)

if (ENABLE_SZ)
    find_package(SZ)
    if (SZ_FOUND)
        add_compile_definitions(WITH_SZ)
    endif ()
endif (ENABLE_SZ)

set(PALLAS_LIB_DIR ${CMAKE_INSTALL_FULL_LIBDIR})

set(INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(INSTALL_PKGCONFIG_DIR "${INSTALL_LIBDIR}/pkgconfig")

# Subdirectory
add_subdirectory(libraries/pallas)
if (ENABLE_OTF2)
    add_subdirectory(libraries/otf2)
endif ()
if (ENABLE_PYTHON)
    add_subdirectory(libraries/pallas_python)
endif()
add_subdirectory(apps)
add_subdirectory(test)


include(PrintOpts)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pallas.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/pallas.pc)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pallas.pc DESTINATION "${INSTALL_PKGCONFIG_DIR}")
