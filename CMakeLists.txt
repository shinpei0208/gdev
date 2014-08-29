CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)

PROJECT(Gdev)
SET(serial "1.0.0")
SET(soserial "1")

## Option
OPTION(driver "select GPU driver. must set driver name in the user-space mode" nouveau)
OPTION(user "user mode(default off)" OFF)
OPTION(runtime "enable CUDA runtime API(default on)" ON)

## Generate gdev_autogen.h
STRING(TOUPPER "${driver}" DRIVER_NAME)
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/common/gdev_autogen.h.in ${CMAKE_CURRENT_BINARY_DIR}/gdev_autogen.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES(util)
INCLUDE_DIRECTORIES(common)

FIND_PACKAGE(Doxygen)
include(UseDoxygen OPTIONAL)

ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(lib)

IF(NOT user)
    ADD_SUBDIRECTORY(mod)
ENDIF(NOT user)

