#ifndef RulesOperandRegisterDefined
#define RulesOperandRegisterDefined


struct OperandRuleRegister;
extern OperandRuleRegister OPRRegister0, OPRRegister1, OPRRegister2;

//reg3 used a separate rule because it applies it result to OpcodeWord1 instead of 0
struct OperandRuleRegister3;
extern OperandRuleRegister3 OPRRegister3ForMAD, OPRRegister3ForCMP, OPRRegister3ForATOM, OPRRegister4ForATOM;

//Note that some operands can have modifiers
//This rule deals with registers that can have the .CC modifier
struct OperandRuleRegisterWithCC;
extern OperandRuleRegisterWithCC OPRRegisterWithCC4IADD32I, OPRRegisterWithCCAt16;

struct OperandRuleRegister0ForMemory;
extern OperandRuleRegister0ForMemory OPRRegister0ForMemory;

//Predicate register operand
struct OperandRulePredicate;
extern OperandRulePredicate OPRPredicate1, 
							OPRPredicate0,
							OPRPredicate2NotNegatable,
							OPRPredicateForLDSLK, 
							OPRPredicateForBAR,
							OPRPredicate0ForVOTE,
							OPRPredicate1ForVOTENoteNegatable,
							OPRPredicate0ForSTSCUL,
							OPRPredicateForSUCLAMP;

//Some predicate registers expressions can be negated with !
//this kind of operand is processed separately
struct OperandRulePredicate2;
extern OperandRulePredicate2 OPRPredicate2, OPRPredicate1ForVOTE, OPRPredicate3ForPSETP;

struct OperandRulePredicateForLDLK;
extern OperandRulePredicateForLDLK OPRPredicateForLDLK;


struct OperandRuleFADD32IReg1;
extern OperandRuleFADD32IReg1 OPRFADD32IReg1;

struct OperandRuleRegister1WithSignFlag;
extern OperandRuleRegister1WithSignFlag OPRIMADReg1, OPRISCADDReg1;



//Register&composite operands for D***
struct OperandRuleRegister0ForDouble;
extern OperandRuleRegister0ForDouble OPRRegister0ForDouble;

struct OperandRuleRegister1ForDouble;
extern OperandRuleRegister1ForDouble OPRRegister1ForDoubleWith2OP, OPRRegister1ForDouble;

struct OperandRuleCompositeOperandForDouble;
extern OperandRuleCompositeOperandForDouble OPRCompositeForDoubleWith2OP, OPRCompositeForDoubleWith1OP;

struct OperandRuleRegister3ForDouble;
extern OperandRuleRegister3ForDouble OPRRegister3ForDouble;

//VADD
struct OperandRuleRegisterForVADD;
extern OperandRuleRegisterForVADD OPRRegister1ForVADD, OPRRegister2ForVADD, OPRRegister2ForI2I;

#endif
