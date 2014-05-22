void inline ApplyModifierRuleUnconditional(ModifierRule* rule)
{
	if(rule->Apply0)
	{
		csCurrentInstruction.OpcodeWord0 &= rule->Mask0;
		csCurrentInstruction.OpcodeWord0 |= rule->Bits0;
	}
	if(csCurrentInstruction.Is8 && rule->Apply1)
	{
		csCurrentInstruction.OpcodeWord1 &= rule->Mask1;
		csCurrentInstruction.OpcodeWord1 |= rule->Bits1;
	}
}


#include "RulesModifier/RulesModifierDataMovement.h"
#include "RulesModifier/RulesModifierInteger.h"
#include "RulesModifier/RulesModifierFloat.h"
#include "RulesModifier/RulesModifierConversion.h"
#include "RulesModifier/RulesModifierCommon.h"
#include "RulesModifier/RulesModifierExecution.h"
#include "RulesModifier/RulesModifierLogic.h"
#include "RulesModifier/RulesModifierOthers.h"