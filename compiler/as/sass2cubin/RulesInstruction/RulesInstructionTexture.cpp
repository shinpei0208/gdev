
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesInstructionTexture.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"


struct InstructionRuleTEX: InstructionRule
{
	InstructionRuleTEX(): InstructionRule("TEX", 7, true, false)
	{
		hpBinaryStringToOpcode8("01100 00 00 0 111 0 000000 000000 111111 00000000 00000 0 0000 0 000 0 0 0 000 0001", OpcodeWord0, OpcodeWord1);
		SetOperands(6, &OPRRegister0,
						&OPRRegister1,
						&OPRTEXImm2,
						&OPRTEXImm3,
						&OPRTEXGeom,
						&OPRTEXImm5);
		ModifierGroups[0].Initialize(true, 1, &MRTEXI);
		ModifierGroups[1].Initialize(true, 5,
						&MRTEXLZ,
						&MRTEXLB,
						&MRTEXLL,
						&MRTEXLBA,
						&MRTEXLLA);
		ModifierGroups[2].Initialize(true, 1, &MRTEXAOFFI);
		ModifierGroups[3].Initialize(true, 1, &MRTEXDC);
		ModifierGroups[4].Initialize(true, 1, &MRTEXNDV);
		ModifierGroups[5].Initialize(true, 1, &MRTEXNODEP);
		ModifierGroups[6].Initialize(true, 2,
						&MRTEXP,
						&MRTEXT);
	}
}IRTEX;

struct InstructionRuleTEXDEPBAR: InstructionRule
{
	InstructionRuleTEXDEPBAR(): InstructionRule("TEXDEPBAR", 0, true, false)
	{
		hpBinaryStringToOpcode8("01100 11 11 0 111 0 000000 000000 000000 00000000 00000 0 0000 0 000 0 0 0 000 1111", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRTEXDEPBARImm);
	}
}IRTEXDEPBAR;

struct InstructionRuleTLD: InstructionRule
{
	InstructionRuleTLD(): InstructionRule("TLD", 7, true, false)
	{
		hpBinaryStringToOpcode8("01100 00 00 0 111 0 000000 000000 111111 00000000 00000 0 0000 0 000 0 0 0 000 1001", OpcodeWord0, OpcodeWord1);
		SetOperands(5, &OPRRegister0,
						&OPRRegister1,
						&OPRTEXImm2,
						&OPRTEXGeom,
						&OPRTEXImm5);
		ModifierGroups[0].Initialize(true, 1, &MRTEXI);
		ModifierGroups[1].Initialize(false, 2,
						&MRTLDLZ,
						&MRTLDLL);
		ModifierGroups[2].Initialize(true, 1, &MRTEXAOFFI);
		ModifierGroups[3].Initialize(true, 1, &MRTLDMS);
		ModifierGroups[4].Initialize(true, 1, &MRTLDCL);
		ModifierGroups[5].Initialize(true, 1, &MRTEXNODEP);
		ModifierGroups[6].Initialize(true, 2,
						&MRTEXP,
						&MRTEXT);
	}
}IRTLD;
