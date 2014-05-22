#ifndef RulesOperandOthersDefined
#define RulesOperandOthersDefined


//ignored operand: currently used for NOP
struct OperandRuleIgnored;
extern OperandRuleIgnored OPRIgnored;

struct OperandRuleSCHI;
extern OperandRuleSCHI OPRSCHI0, 
					   OPRSCHI1,
					   OPRSCHI2,
					   OPRSCHI3,
					   OPRSCHI4,
					   OPRSCHI5,
					   OPRSCHI6;

struct OperandRule32I;
extern OperandRule32I OPR32I;

struct OperandRuleLOP;
extern OperandRuleLOP OPRLOP1, OPRLOP2;


struct OperandRuleF2I;
extern OperandRuleF2I OPRF2I, OPRI2F, OPRI2I;


struct OperandRuleISCADDShift;
extern OperandRuleISCADDShift OPRISCADDShift;

struct OperandRuleNOPCC;
extern OperandRuleNOPCC OPRNOPCC;

struct OperandRuleTEXImm;
extern OperandRuleTEXImm OPRTEXImm2, OPRTEXImm3, OPRTEXImm5;
extern OperandRuleTEXImm OPRTEXDEPBARImm;

struct OperandRuleTEXGeom;
extern OperandRuleTEXGeom OPRTEXGeom;

struct OperandRuleBRACCOrInstAddr;
extern OperandRuleBRACCOrInstAddr OPRBRACCOrInstAddr;

struct OperandRuleSUCLAMPImm;
extern OperandRuleSUCLAMPImm OPRSUCLAMPImm;

#endif
