#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesInstructionDataMovement.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"

struct InstructionRuleMOV: InstructionRule
{
	InstructionRuleMOV(): InstructionRule("MOV", 1, true, false)
	{
		hpBinaryStringToOpcode8("0010 011110111000000000000000000000000000000000000000000000010100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,
					&OPRMOVStyle);
		ModifierGroups[0].Initialize(true, 1, &MRS);
	}
}IRMOV;

struct InstructionRuleMOV32I: InstructionRule
{
	InstructionRuleMOV32I(): InstructionRule("MOV32I", 1, true, false)
	{
		hpBinaryStringToOpcode8("0100 011110111000000000000000000000000000000000000000000000011000", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 1, &MRS);
	}
}IRMOV32I;



struct InstructionRuleLD: InstructionRule
{
	InstructionRuleLD() : InstructionRule("LD", 3, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
		//2 operands
		SetOperands(2, 
					&OPRRegister0ForMemory,					 //register
					&OPRGlobalMemoryWithImmediate32);//global memory
		//3 modifier groups
		ModifierGroups[0].Initialize(true, 1, &MRE);
		ModifierGroups[1].Initialize(true, 3, //3 modifiers in this group
					&MRLDCopCG, //.CG
					&MRLDCopCS, //.CS
					&MRLDCopCV);//.CV
		ModifierGroups[2].Initialize(true, 6, //6 modifiers in this group
					&MRLDU8,  //.U8
					&MRLDS8,  //.S8
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLD;


struct InstructionRuleLDU: InstructionRule
{
	InstructionRuleLDU() : InstructionRule("LDU", 2, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100111000000000000000000000000000000000000000000000010001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0ForMemory,					
					&OPRGlobalMemoryWithImmediate32);
		ModifierGroups[0].Initialize(true, 1, &MRE);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8,
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLDU;


struct InstructionRuleLDL: InstructionRule
{
	InstructionRuleLDL() : InstructionRule("LDL", 2, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100111000000000000000000000000000000000000000000000000011", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0ForMemory,					
					&OPRGlobalMemoryWithImmediate24);
		ModifierGroups[0].Initialize(true, 4,
					&MRLDCopCG,
					&MRLDCopCS,
					&MRLDCopLU,
					&MRLDCopCV);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8,
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLDL;

struct InstructionRuleLDS : InstructionRule
{
	InstructionRuleLDS(): InstructionRule("LDS", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100111000000000000000000000000000000000000000000010000011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0ForMemory,
					&OPRGlobalMemoryWithImmediate24);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRLDS;

struct InstructionRuleLDC : InstructionRule
{
	InstructionRuleLDC(): InstructionRule("LDC", 1, true, false)
	{
		hpBinaryStringToOpcode8("0110000100111000000000000000000000000000000000000000000000101000", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0ForMemory,
					&OPRConstantMemory);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRLDC;

struct InstructionRuleST: InstructionRule
{
	InstructionRuleST() : InstructionRule("ST", 3, true, false)
	{
		hpBinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRGlobalMemoryWithImmediate32,
					&OPRRegister0ForMemory);
		ModifierGroups[0].Initialize(true, 1, &MRE);
		ModifierGroups[1].Initialize(true, 3,
					&MRSTCopCG,
					&MRSTCopCS,
					&MRSTCopWT);
		ModifierGroups[2].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRST;


struct InstructionRuleSTL: InstructionRule
{
	InstructionRuleSTL() : InstructionRule("STL", 2, true, false)
	{
		hpBinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000010011", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRGlobalMemoryWithImmediate24,
					&OPRRegister0ForMemory);
		ModifierGroups[0].Initialize(true, 3,
					&MRSTCopCG,
					&MRSTCopCS,
					&MRSTCopWT);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRSTL;

struct InstructionRuleSTS : InstructionRule
{
	InstructionRuleSTS(): InstructionRule("STS", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000010010011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRGlobalMemoryWithImmediate24,
					&OPRRegister0ForMemory);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRSTS;


struct InstructionRuleLDLK: InstructionRule
{
	InstructionRuleLDLK(): InstructionRule("LDLK", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 0001  00 1110 000000 000000 00000000000000000000000000000000   0 00101", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRPredicateForLDLK,
					&OPRRegister0ForMemory,
					&OPRGlobalMemoryWithImmediate32);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLDLK;

struct InstructionRuleLDSLK: InstructionRule
{
	InstructionRuleLDSLK(): InstructionRule("LDSLK", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100 1110 000000 000000 000000000000000000000000 000 00000 100011", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRPredicateForLDSLK,
					&OPRRegister0ForMemory,
					&OPRGlobalMemoryWithImmediate24);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLDSLK;


struct InstructionRuleSTUL: InstructionRule
{
	InstructionRuleSTUL() : InstructionRule("STUL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100 1110 000000 000000 0000000000000000 0000000000000000 010111", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRGlobalMemoryWithImmediate32,
					&OPRRegister0ForMemory);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRSTUL;

struct InstructionRuleSTSUL : InstructionRule
{
	InstructionRuleSTSUL(): InstructionRule("STSUL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100 1110 000000 000000 0000000000000000 0000000000000000 110011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRGlobalMemoryWithImmediate24,
					&OPRRegister0ForMemory);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRSTSUL;

struct InstructionRuleSTSCUL : InstructionRule
{
	InstructionRuleSTSCUL(): InstructionRule("STSCUL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000100 1110 000000 000000 0000000000000000 0000000000000000 011101", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRPredicate0ForSTSCUL,
					&OPRGlobalMemoryWithImmediate24,
					&OPRRegister0ForMemory);
		ModifierGroups[0].Initialize(true, 1, &MRSUST64);
	}
}IRSTSCUL;
