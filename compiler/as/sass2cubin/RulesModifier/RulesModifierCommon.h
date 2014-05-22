#ifndef RulesModifierCommonDefined
#define RulesModifierCommonDefined


struct ModifierRuleSETPLogic;
extern ModifierRuleSETPLogic MRSETPLogicAND, MRSETPLogicOR, MRSETPLogicXOR;

struct ModifierRuleSETPComparison;
extern ModifierRuleSETPComparison 
	MRSETPComparisonLT,
	MRSETPComparisonEQ,
	MRSETPComparisonLE,
	MRSETPComparisonGT,
	MRSETPComparisonNE,
	MRSETPComparisonGE,
	MRSETPComparisonNUM,
	MRSETPComparisonNAN,
	MRSETPComparisonLTU,
	MRSETPComparisonEQU,
	MRSETPComparisonLEU,
	MRSETPComparisonGTU,
	MRSETPComparisonNEU,
	MRSETPComparisonGEU;

struct ModifierRuleS;
extern ModifierRuleS MRS;

struct ModifierRuleALU;
extern ModifierRuleALU MRALU, MRXLU;
#endif