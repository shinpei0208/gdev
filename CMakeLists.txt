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
OPTION(usched "enable user mode scheduler(default off)" OFF)

MACRO(INCLUDE_DIRECTORY_IF_EXISTS DIR)
    IF(EXISTS "${DIR}/")
        INCLUDE_DIRECTORIES(${DIR})
    ENDIF()
ENDMACRO()

## Generate gdev_autogen.h
STRING(TOUPPER "${driver}" DRIVER_NAME)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/common/gdev_autogen.h.in ${CMAKE_CURRENT_BINARY_DIR}/gdev_autogen.h)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

IF(user)
    INCLUDE_DIRECTORY_IF_EXISTS(/usr/local/include/libdrm)
ENDIF()

INCLUDE_DIRECTORIES(util)
INCLUDE_DIRECTORIES(common)

FIND_PACKAGE(Doxygen)
include(UseDoxygen OPTIONAL)

ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(lib)

IF(NOT user)
    ADD_SUBDIRECTORY(mod)
ENDIF(NOT user)

