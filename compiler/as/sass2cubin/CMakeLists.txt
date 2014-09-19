SET(CMAKE_CXX_FLAGS_RELEASE "-w -O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-w -g")
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_COMPILER g++)

include(UseDoxygen OPTIONAL)

FILE(GLOB main_src "*.cpp")
FILE(GLOB helper_src "helper/*.cpp")
FILE(GLOB instruction_src "RulesInstruction/*.cpp")
FILE(GLOB modifier_src "RulesModifier/*.cpp")
FILE(GLOB operand_src "RulesOperand/*.cpp")

SET(sass2cubin_src
    ${main_src}
    ${helper_src}
    ${instruction_src}
    ${modifier_src}
    ${operand_src}
)

ADD_EXECUTABLE(sass2cubin ${sass2cubin_src})
INSTALL(TARGETS sass2cubin DESTINATION gdev/bin)
