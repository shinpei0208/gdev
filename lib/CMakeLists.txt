SET(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_C_FLAGS "-DGDEV_SCHED_DISABLED")

SET(serial "1.0.0")
SET(soserial "1")
include(UseDoxygen OPTIONAL)


## Each source files
SET(gdev_src ${common_src})
file(GLOB common_src "../common/*.c~*nv50*")
file(GLOB nvrm_src "user/nvrm/*.c")
file(GLOB user_src "user/gdev/*.c")
file(GLOB nouveau_src "user/nouveau/*.c")
file(GLOB kernel_src "kernel/*.c")

## include dir
SET(gdev_inc /usr/local/gdev/include)


## select source 
IF(user)
    MESSAGE(Selected\ User-space!)
    SET(gdev_src ${gdev_src} ${user_src})
    SET(gdev_inc ${gdev_inc} /usr/include/libdrm ../common user/gdev)
    IF(driver STREQUAL nouveau)
	SET(gdev_src ${gdev_src} ${nouveau_src})
	SET(gdev_inc ${gdev_inc} user/gdev/nouveau)
    ELSE(driver STREQUAL nouveau)
	IF(driver STREQUAL nvrm)
	    SET(gdev_src ${gdev_src} ${nvrm_src})
	    SET(gdev_inc ${gdev_inc} user/gdev/nvrm)
	ELSE(driver STREQUAL nvrm)
	    IF(driver STREQUAL pscnv)
		SET(gdev_src ${gdev_src} ${pscnv_src})
		SET(gdev_inc ${gdev_inc} user/gdev/nvrm)
	    ELSE(driver STREQUAL  pscnv)
		MESSAGE( FATAL_ERROR "Not selected GPU Driver.")
		MESSAGE( FATAL_ERROR "ex: driver=nouveau.")
	    ENDIF(driver STREQUAL pscnv)
	ENDIF(driver STREQUAL nvrm)
    ENDIF(driver STREQUAL nouveau)
    SET(gdev_install_headers 
	${PROJECT_SOURCE_DIR}/common/gdev_api.h 
	${PROJECT_SOURCE_DIR}/common/gdev_nvidia_def.h 
	${PROJECT_SOURCE_DIR}/common/gdev_list.h 
	${PROJECT_SOURCE_DIR}/common/gdev_time.h)
ELSE(user)
    MESSAGE(Selected\ Kernel-space!)
    SET(gdev_src ${gdev_src} ${kernel_src})
    SET(gdev_inc ${gdev_inc} kernel ../common)
    SET(gdev_install_headers 
	${PROJECT_SOURCE_DIR}/common/gdev_api.h 
	${PROJECT_SOURCE_DIR}/common/gdev_nvidia_def.h 
	${PROJECT_SOURCE_DIR}/common/gdev_list.h 
	${PROJECT_SOURCE_DIR}/common/gdev_time.h)
ENDIF(user)


INCLUDE_DIRECTORIES(gdev ${gdev_inc})
ADD_LIBRARY(gdev SHARED ${gdev_src})
SET_TARGET_PROPERTIES(gdev PROPERTIES VERSION ${serial} SOVERSION ${soserial})

INSTALL(TARGETS gdev LIBRARY DESTINATION gdev/lib64)
INSTALL(FILES ${gdev_install_headers} DESTINATION gdev/include)

