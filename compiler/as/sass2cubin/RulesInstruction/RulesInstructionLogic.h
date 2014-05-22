#ifndef RulesInstructionLogicDefined


struct InstructionRuleLOP;
extern InstructionRuleLOP IRLOP;

struct InstructionRuleLOP32I;
extern InstructionRuleLOP32I IRLOP32I;

struct InstructionRuleSHR;
extern InstructionRuleSHR IRSHR, IRSHL;

struct InstructionRuleBFE;
extern InstructionRuleBFE IRBFE;

struct InstructionRuleBFI;
extern InstructionRuleBFI IRBFI;

struct InstructionRuleSEL;
extern InstructionRuleSEL IRSEL;


#else
#define RulesInstructionLogicDefined
#endif