cmake_minimum_required(VERSION 3.7)
project(InterfaceGrid VERSION 1.0 LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 14)
set(LIB /cs/labs/dina/dina/libs/gamb)
set(MYLIB /cs/labs/dina/dina/libs/DockingLib)
set(CMAKE_PREFIX_PATH /cs/labs/dina/punims/)

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(ENABLE_STATIC ON)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(Boost_USE_STATIC_LIBS ON)
set(HDF5_USE_STATIC_LIBRARIES ON)

FIND_PACKAGE(Boost COMPONENTS program_options system filesystem REQUIRED)
INCLUDE_DIRECTORIES(${LIB} ${MYLIB} ${Boost_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

file(GLOB SOURCE_FILES_GRID
        interface.cc cnpy.cc
        )
link_directories(${LIB} ${MYLIB} /usr/lib/x86_64-linux-gnu/ )
add_executable(interface2grid ${SOURCE_FILES_GRID})
target_link_libraries(interface2grid z dockingLib gamb++ boost_program_options boost_system boost_filesystem m)
