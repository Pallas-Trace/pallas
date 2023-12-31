cmake_minimum_required(VERSION 3.13.0)

project(HTF
        VERSION 0.1.0
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

# CMakeFind repository
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)


# Look for libraries

# Options to enable/disable modules
option(ENABLE_OTF2 "Enable OTF2 compatibilit layer" ON)
option(BUILD_DOC "Build Doxygen Documentation" ON)
option(ENABLE_SZ "Enables compression using the SZ lib" ON)
option(ENABLE_ZFP "Enables compression using the ZFP lib" ON)

# if (ENABLE_MPI)
# find_program(MPICC mpicc REQUIRED)
# endif()

find_package(zstd REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp REQUIRED)
find_package(Doxygen)


INCLUDE(CheckTypeSize)
SET(CMAKE_EXTRA_INCLUDE_FILES "map")
check_type_size("std::map<uint32_t, size_t>" SIZEOF_MAP LANGUAGE CXX)
message("Size of std::map<uint32_t, size_t>: ${SIZEOF_MAP}")
add_compile_definitions(MAP_SIZE=${SIZEOF_MAP})

SET(CMAKE_EXTRA_INCLUDE_FILES "vector")
check_type_size("std::vector<size_t>" SIZEOF_VECTOR LANGUAGE CXX)
message("Size of std::vector<size_t>: ${SIZEOF_VECTOR}")
add_compile_definitions(VECTOR_SIZE=${SIZEOF_VECTOR})

####################################################
# Some flags for compilation

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
	link_libraries(${SZ_LIBRARIES})
    endif ()
endif (ENABLE_SZ)

set(HTF_LIB_DIR ${CMAKE_INSTALL_FULL_LIBDIR})

set(INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(INSTALL_PKGCONFIG_DIR "${INSTALL_LIBDIR}/pkgconfig")

# Subdirectory
add_subdirectory(libraries/htf)
if (ENABLE_OTF2)
    add_subdirectory(libraries/otf2)
endif ()
#add_subdirectory (tracer)
add_subdirectory(apps)
add_subdirectory(test)
# if (ENABLE_MPI)
# add_subdirectory (test/mpi)
# endif()

include(PrintOpts)
configure_file(config.json config.json COPYONLY)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/htf.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/htf.pc)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/htf.pc DESTINATION "${INSTALL_PKGCONFIG_DIR}")
