#ifndef RulesOperandMemoryDefined
#define RulesOperandMemoryDefined


//Global Memory Operand
struct OperandRuleGlobalMemoryWithImmediate32;
extern OperandRuleGlobalMemoryWithImmediate32 OPRGlobalMemoryWithImmediate32;

struct OperandRuleGlobalMemoryWithImmediate24;
extern OperandRuleGlobalMemoryWithImmediate24 OPRGlobalMemoryWithImmediate24;

struct OperandRuleGlobalMemoryWithLastWithoutLast2Bits;
extern OperandRuleGlobalMemoryWithLastWithoutLast2Bits OPRGlobalMemoryWithLastWithoutLast2Bits;

struct OperandRuleMemoryForATOM;
extern OperandRuleMemoryForATOM OPRMemoryForATOM;

//Constant Memory Operand
struct OperandRuleConstantMemory;
extern OperandRuleConstantMemory OPRConstantMemory;

#else
#endif