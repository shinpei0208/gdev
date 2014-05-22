#ifndef RulesInstructionDataMovementDefined
#define RulesInstructionDataMovementDefined

struct InstructionRuleMOV;
extern InstructionRuleMOV IRMOV;

struct InstructionRuleMOV32I;
extern InstructionRuleMOV32I IRMOV32I;



struct InstructionRuleLD;
extern InstructionRuleLD IRLD;


struct InstructionRuleLDU;
extern InstructionRuleLDU IRLDU;


struct InstructionRuleLDL;
extern InstructionRuleLDL IRLDL;

struct InstructionRuleLDS;
extern InstructionRuleLDS IRLDS;

struct InstructionRuleLDC;
extern InstructionRuleLDC IRLDC;

struct InstructionRuleST;
extern InstructionRuleST IRST;


struct InstructionRuleSTL;
extern InstructionRuleSTL IRSTL;

struct InstructionRuleSTS;
extern InstructionRuleSTS IRSTS;

struct InstructionRuleLDLK;
extern InstructionRuleLDLK IRLDLK;

struct InstructionRuleLDSLK;
extern InstructionRuleLDSLK IRLDSLK;

struct InstructionRuleSTUL;
extern InstructionRuleSTUL IRSTUL;

struct InstructionRuleSTSUL;
extern InstructionRuleSTSUL IRSTSUL;

struct InstructionRuleSTSCUL;
extern InstructionRuleSTSCUL IRSTSCUL;

#else
#endif
