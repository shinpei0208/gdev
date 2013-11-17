CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)

PROJECT(Gdev)
SET(serial "1.0.0")
SET(soserial "1")

INCLUDE_DIRECTORIES(util)

FIND_PACKAGE(Doxygen)
include(UseDoxygen OPTIONAL)

ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(lib)

## Option 
OPTION(driver "select GPU driver. must set driver name in the user-space mode" nouveau)
OPTION(user "user mode(default off)" OFF)
OPTION(runtime "enable CUDA runtime API(default on)" ON)

IF(NOT user)
    ADD_SUBDIRECTORY(mod)
ENDIF(NOT user)

