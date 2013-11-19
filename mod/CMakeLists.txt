SET(serial "1.0.0")
SET(soserial "1")

include(UseDoxygen OPTIONAL)

file(GLOB common_src "../common/*")
file(GLOB gmod_src "./gdev/*")

set(target gdev_module)

set(mod_src ${common_src} ${gmod_src})

include_directories(../common/ /usr/local/gdev/include ./gdev)

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

SET( KBUILD_COMMAND
    ${CMAKE_MAKE_PROGRAM} -C ${KERNEL_DIR} M=${MODULE_OUTPUT_DIR} modules
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
