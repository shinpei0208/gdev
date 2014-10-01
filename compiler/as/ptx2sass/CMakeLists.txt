SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS "-std=c++0x")
SET(CMAKE_CXX_COMPILER g++)
#SET(CMAKE_VERBOSE_MAKEFILE true)
#MESSAGE("CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")

SET(serial "1.0.0")
SET(soserial "1")
include(UseDoxygen OPTIONAL)

FIND_PACKAGE(FLEX)
FIND_PACKAGE(BISON)
FLEX_TARGET(ptx
    ocelot/parser/implementation/ptx.ll
    ${CMAKE_CURRENT_BINARY_DIR}/ptx.cpp)
BISON_TARGET(ptxgrammar
    ocelot/parser/implementation/ptxgrammar.yy
    ${CMAKE_CURRENT_BINARY_DIR}/ptxgrammar.cpp)
ADD_FLEX_BISON_DEPENDENCY(ptx ptxgrammar)

ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/configure.h
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/config_h_build.sh ${CMAKE_CURRENT_SOURCE_DIR}/configure.h.in ${CMAKE_CURRENT_BINARY_DIR}/configure.h
)

ADD_DEFINITIONS("-Wno-unused-local-typedefs")

FILE(GLOB hydrazine_src "hydrazine/implementation/*.cpp")
FILE(GLOB analysis_src "ocelot/analysis/implementation/*.cpp")
FILE(GLOB api_src "ocelot/api/implementation/*.cpp")
FILE(GLOB cal_src "ocelot/cal/implementation/*.cpp")
FILE(GLOB cuda_src "ocelot/cuda/implementation/*.cpp")
FILE(GLOB executive_src "ocelot/executive/implementation/*.cpp")
FILE(GLOB ir_src "ocelot/ir/implementation/*.cpp")
FILE(GLOB parser_src "ocelot/parser/implementation/*.cpp")
FILE(GLOB trace_src "ocelot/trace/implementation/*.cpp")
FILE(GLOB transforms_src "ocelot/transforms/implementation/*.cpp")
FILE(GLOB translator_src "ocelot/translator/implementation/*.cpp")
FILE(GLOB util_src "ocelot/util/implementation/*.cpp")

SET(ocelot_src
    ${hydrazine_src}
    ${analysis_src}
    ${api_src}
    ${cal_src}
    ${cuda_src}
    ${executive_src}
    ${ir_src}
    ${parser_src}
    ${trace_src}
    ${transforms_src}
    ${translator_src}
    ${util_src}
)
INCLUDE_DIRECTORIES(ocelot . /usr/include ${CMAKE_CURRENT_BINARY_DIR})
ADD_LIBRARY(ocelot SHARED
    ${ocelot_src}
    ${FLEX_ptx_OUTPUTS}
    ${BISON_ptxgrammar_OUTPUTS}
    ${CMAKE_CURRENT_BINARY_DIR}/configure.h
)
TARGET_LINK_LIBRARIES(ocelot
    boost_system boost_filesystem boost_thread dl rt pthread)
SET_TARGET_PROPERTIES(ocelot
    PROPERTIES VERSION ${serial} SOVERSION ${soserial})
INSTALL(TARGETS ocelot DESTINATION gdev/lib64)

ADD_EXECUTABLE(ptx2sass
    ocelot/tools/PTXToSASSTranslator.cpp)
TARGET_LINK_LIBRARIES(ptx2sass
    ocelot boost_system boost_filesystem boost_thread dl rt pthread)

INSTALL(TARGETS ptx2sass DESTINATION gdev/bin)
