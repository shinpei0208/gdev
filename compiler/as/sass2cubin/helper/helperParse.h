#ifndef helperParseDefined
#define helperParseDefined

void hpParseBreakDirectiveIntoParts();
void hpParseBreakInstructionIntoComponents();
int hpParseComputeInstructionNameIndex(SubString &name);
int hpParseFindInstructionRuleArrayIndex(int Index);
int hpParseComputeDirectiveNameIndex(SubString &name);
int hpParseFindDirectiveRuleArrayIndex(int Index);
void hpParseApplyModifier(ModifierRule &rule);
void hpParseProcessPredicate();

#else
#endif