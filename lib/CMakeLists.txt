SET(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)
IF(NOT(user AND usched))
    SET(CMAKE_C_FLAGS "-DGDEV_SCHED_DISABLED")
ENDIF(NOT(user AND usched))

SET(serial "1.0.0")
SET(soserial "1")
include(UseDoxygen OPTIONAL)

ADD_DEFINITIONS("-Wno-unused-but-set-variable")
ADD_DEFINITIONS("-Wno-unused-result")

## Each source files
set(common_src
    ${PROJECT_SOURCE_DIR}/common/gdev_api.c
    ${PROJECT_SOURCE_DIR}/common/gdev_device.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_compute.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_fifo.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_mem.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_nvc0.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_nve4.c
    ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_shm.c
    ${PROJECT_SOURCE_DIR}/common/gdev_sched.c
)
file(GLOB util_src "${PROJECT_SOURCE_DIR}/util/*.c")
file(GLOB nvrm_src "user/nvrm/*.c")
file(GLOB user_src "user/gdev/*.c")
file(GLOB nouveau_src "user/nouveau/*.c")
file(GLOB barra_src "user/barra/*.c")
file(GLOB kernel_src "kernel/*.c")
file(GLOB usched_src "user/usched/*.c")


## include dir
SET(gdev_inc /usr/local/gdev/include)

## select source
IF(user)
    MESSAGE(Selected\ User-space!)
    SET(gdev_src ${gdev_src} ${user_src} ${common_src} ${util_src})
    SET(gdev_inc ${gdev_inc}
        /usr/include/libdrm
        ${PROJECT_SOURCE_DIR}/common/
        ${CMAKE_CURRENT_SOURCE_DIR}/user/gdev/
    )

    IF(driver STREQUAL nouveau)
        SET(gdev_src ${gdev_src} ${nouveau_src})
        SET(gdev_inc ${gdev_inc} ${CMAKE_CURRENT_SOURCE_DIR}/user/gdev/nouveau)
        SET(link_lib drm drm_nouveau)
    ELSEIF(driver STREQUAL nvrm)
        SET(gdev_src ${gdev_src} ${nvrm_src} )
        SET(gdev_inc ${gdev_inc} ${CMAKE_CURRENT_SOURCE_DIR}/user/gdev/nvrm)
    ELSEIF(driver STREQUAL pscnv)
        SET(gdev_src ${gdev_src} ${pscnv_src})
        SET(gdev_inc ${gdev_inc} ${CMAKE_CURRENT_SOURCE_DIR}/user/gdev/pscnv)
    ELSEIF(driver STREQUAL barra)
        SET(gdev_src ${gdev_src} ${barra_src})
        SET(gdev_inc ${gdev_inc} ${CMAKE_CURRENT_SOURCE_DIR}/user/gdev/barra)
        SET(link_lib barra)
    ELSE(driver STREQUAL  pscnv)
        MESSAGE( FATAL_ERROR "Not selected GPU Driver.")
        MESSAGE( FATAL_ERROR "ex: driver=nouveau.")
    ENDIF(driver STREQUAL nouveau)
    SET(gdev_install_headers
        ${PROJECT_SOURCE_DIR}/common/gdev_api.h
        ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_def.h
        ${PROJECT_SOURCE_DIR}/common/gdev_list.h
        ${PROJECT_SOURCE_DIR}/common/gdev_time.h
        ${PROJECT_BINARY_DIR}/gdev_autogen.h)
ELSE(user)
    MESSAGE(Selected\ Kernel-space!)
    SET(gdev_src ${gdev_src} ${kernel_src})
    SET(gdev_inc ${gdev_inc} ${CMAKE_CURRENT_SOURCE_DIR}/kernel ${PROJECT_SOURCE_DIR}/common)
    SET(gdev_install_headers
        ${PROJECT_SOURCE_DIR}/common/gdev_api.h
        ${PROJECT_SOURCE_DIR}/common/gdev_nvidia_def.h
        ${PROJECT_SOURCE_DIR}/common/gdev_list.h
        ${PROJECT_SOURCE_DIR}/common/gdev_time.h
        ${PROJECT_BINARY_DIR}/gdev_autogen.h)
ENDIF(user)

INCLUDE_DIRECTORIES(gdev ${gdev_inc})
ADD_LIBRARY(gdev SHARED ${gdev_src})
SET_TARGET_PROPERTIES(gdev PROPERTIES VERSION ${serial} SOVERSION ${soserial})
TARGET_LINK_LIBRARIES(gdev ${link_lib})
INSTALL(TARGETS gdev LIBRARY DESTINATION gdev/lib64)
INSTALL(FILES ${gdev_install_headers} DESTINATION gdev/include)

IF(user AND usched)
    FIND_PACKAGE(Threads)
    ADD_EXECUTABLE(gdev_usched_monitor ${usched_src})
    TARGET_LINK_LIBRARIES(gdev_usched_monitor ${link_lib} ${CMAKE_THREAD_LIBS_INIT})
    INSTALL(TARGETS gdev_usched_monitor RUNTIME DESTINATION gdev/bin)
ENDIF(user AND usched)

