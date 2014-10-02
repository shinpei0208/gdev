SET(serial "1.0.0")
SET(soserial "1")

include(UseDoxygen OPTIONAL)

file(GLOB common_src "../common/*")
file(GLOB util_src "../util/*")
file(GLOB gmod_src "./gdev/*")

set(target gdev_module)

set(mod_src ${common_src} ${util_src} ${gmod_src})

include_directories(../common/ ../util/ /usr/local/gdev/include ./gdev)

SET( MODULE_NAME
    gdev
    )

SET( MODULE_SOURCE_FILES
    ${mod_src}
    )

SET( MODULE_FILE
    ${MODULE_NAME}.ko
    )

SET( MODULE_OUTPUT_DIR
    ${CMAKE_BINARY_DIR}/mod/${MODULE_NAME}
   )

SET( KERNEL_DIR
    "/lib/modules/${CMAKE_SYSTEM_VERSION}/build"
   )

FIND_PROGRAM(MAKE_EXECUTABLE make DOC "path to the make executable")
MARK_AS_ADVANCED(MAKE_EXECUTABLE)
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MAKE DEFAULT_MSG MAKE_EXECUTABLE)
IF(NOT MAKE_EXECUTABLE)
    MESSAGE(FATAL_ERROR "Could not find prerequisite `make` executable")
ENDIF()

SET( KBUILD_COMMAND
    ${MAKE_EXECUTABLE} -C ${KERNEL_DIR} M=${MODULE_OUTPUT_DIR} modules
   )

CONFIGURE_FILE(
    Kbuild.in
    ${MODULE_OUTPUT_DIR}/Kbuild
    )

FOREACH( MODULE_SOURCE_FILE ${MODULE_SOURCE_FILES} )
    CONFIGURE_FILE(
	${MODULE_SOURCE_FILE}
	${MODULE_OUTPUT_DIR}
	COPYONLY
	)
ENDFOREACH( MODULE_SOURCE_FILE )

ADD_CUSTOM_COMMAND(
    OUTPUT ${MODULE_OUTPUT_DIR}/${MODULE_FILE}
    COMMAND ${KBUILD_COMMAND}
    WORKING_DIRECTORY ${MODULE_OUTPUT_DIR}
    DEPENDS ${MODULE_SOURCE_FILES} Kbuild.in
    VERBATIM
    )

ADD_CUSTOM_TARGET(
    ${target}
    ALL
    DEPENDS ${MODULE_OUTPUT_DIR}/${MODULE_FILE}
    )
