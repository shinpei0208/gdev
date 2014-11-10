FIND_PACKAGE(FLEX)
FIND_PACKAGE(BISON)
FIND_FILE(PERL4_CORELIBS getopts.pl PATHS /usr/lib/perl5 /usr/share/perl5)
FIND_PROGRAM(TCSH tcsh PATHS /bin)
EXECUTE_PROCESS(
	COMMAND ${CMAKE_C_COMPILER} -dumpversion
	OUTPUT_VARIABLE GCC_VERSION
)
if (GCC_VERSION VERSION_LESS 4.7)
if(EXISTS ${PERL4_CORELIBS})
if(EXISTS ${TCSH})
EXECUTE_PROCESS(
	COMMAND wget -O - https://github.com/shinpei0208/gdev-nvopencc/blob/master/nvopencc_5.0_src_13604779.tar.gz?raw=true
	COMMAND tar xzf -
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
EXECUTE_PROCESS(
	COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/nvopencc_5.0.patch
	COMMAND patch -p0 --read-only=ignore
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
ADD_CUSTOM_TARGET(open64 ALL make
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/targia3264_nvisa
)
ADD_CUSTOM_TARGET(clean make clean
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/targia3264_nvisa
)
INSTALL(
	PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/src/targia3264_nvisa/bin/nvopencc
	DESTINATION gdev/open64/bin
)
FILE(GLOB libs "${CMAKE_CURRENT_BINARY_DIR}/src/targia3264_nvisa/lib/*")
INSTALL(
	PROGRAMS ${libs}
	DESTINATION gdev/open64/lib
)
else()
MESSAGE("skip: Open64 is required tcsh.")
endif()
else()
MESSAGE("skip: Open64 is required perl4-corelibs package.")
endif()
else()
MESSAGE("skip: Open64 is required GCC version 4.6 or older.")
endif()
