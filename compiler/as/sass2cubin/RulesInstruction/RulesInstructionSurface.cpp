
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesInstructionSurface.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"


struct InstructionRuleSUST: InstructionRule
{
	InstructionRuleSUST(): InstructionRule("SUST", 5, true, false)
	{
		hpBinaryStringToOpcode8("10100 001 10 111 0 000000 000000 000000 00000000 0000000 00 0000000 0011 1011", OpcodeWord0, OpcodeWord1);
		SetOperands(3, &OPRRegister2,
				&OPRRegister1,
				&OPRImmediate16HexOrInt);
		ModifierGroups[0].Initialize(true, 1, &MRSUST2D);
		ModifierGroups[1].Initialize(true, 1, &MRSUSTB);
		ModifierGroups[2].Initialize(true, 2,
				&MRSUSTNEAR,
				&MRSUSTTRAP);
		ModifierGroups[3].Initialize(true, 3,
				&MRSUSTWB,
				&MRSUSTCS,
				&MRSUSTWT);
		ModifierGroups[4].Initialize(true, 6,
				&MRSUSTU8,
				&MRSUSTS8,
				&MRSUSTU16,
				&MRSUSTS16,
				&MRSUST64,
				&MRSUST128);
	}
} IRSUST;

struct InstructionRuleSUEAU: InstructionRule
{
	InstructionRuleSUEAU(): InstructionRule("SUEAU", 0, true, false)
	{
		hpBinaryStringToOpcode8("0010 0000 0011 1000 0000 0000 0000 0000  0000 0000 0000 0000 0000 0000 0000 0110", OpcodeWord0, OpcodeWord1);
		SetOperands(4, &OPRRegister0,
				&OPRRegister1,
				&OPRRegister2,
				&OPRSUEAUStyle);
	}
} IRSUEAU;

struct InstructionRuleSUBFM: InstructionRule
{
	InstructionRuleSUBFM(): InstructionRule("SUBFM", 1, true, false)
	{
		hpBinaryStringToOpcode8("0010 0000 0011 1000 0000 0000 0000 0000  0000 0000 0000 0000 0000 0000 0011 1010", OpcodeWord0, OpcodeWord1);
		ModifierGroups[0].Initialize(true, 1, &MRSUBFM3D);
		SetOperands(5, &OPRPredicate2NotNegatable,
				&OPRRegister0,
				&OPRRegister1,
				&OPRRegister2,
				&OPRSUEAUStyle);
	}
} IRSUBFM;

struct InstructionRuleSUCLAMP: InstructionRule
{
	InstructionRuleSUCLAMP(): InstructionRule("SUCLAMP", 2, true, false)
	{
		hpBinaryStringToOpcode8("0010 0000 0111 1000 0000 0000 0000 0000  0000 0000 0000 0000 1000 0000 0001 1010", OpcodeWord0, OpcodeWord1);
		SetOperands(5, &OPRPredicateForSUCLAMP,
				&OPRRegister0,
				&OPRRegister1,
				&OPRLOP2,
				&OPRSUCLAMPImm);
		ModifierGroups[0].Initialize(true, 3, &MRSUCLAMPSD,
				&MRSUCLAMPPL,
				&MRSUCLAMPBL);
		ModifierGroups[1].Initialize(true, 5, &MRSUCLAMPR1,
				&MRSUCLAMPR2,
				&MRSUCLAMPR4,
				&MRSUCLAMPR8,
				&MRSUCLAMPR16);
	}
} IRSUCLAMP;
