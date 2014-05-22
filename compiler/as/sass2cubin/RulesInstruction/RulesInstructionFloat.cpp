#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesInstructionFloat.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"


struct InstructionRuleFADD: InstructionRule
{
	InstructionRuleFADD() : InstructionRule("FADD", 4, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRFADD32IReg1, 
					&OPRFADDCompositeWithOperator);
		ModifierGroups[0].Initialize(true, 1, &MRFADD32IFTZ);
		ModifierGroups[1].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[2].Initialize(true, 1, &MRFADDSAT);
		ModifierGroups[3].Initialize(true, 1, &MRS);

	}
}IRFADD;

struct InstructionRuleFADD32I: InstructionRule
{
	InstructionRuleFADD32I() : InstructionRule("FADD32I", 1, true, false)
	{
		hpBinaryStringToOpcode8("0100 000000111000000000000000000000000000000000000000000000010100", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRFADD32IReg1,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 1, &MRFADD32IFTZ);
	}
}IRFADD32I;



struct InstructionRuleFMUL: InstructionRule
{
	InstructionRuleFMUL(): InstructionRule("FMUL", 4, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000111000000000000000000000000000000000000000000000011010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRRegister1,
					&OPRFMULAllowNegative);
		ModifierGroups[0].Initialize(true, 1, &MRFMUL32IFTZ);
		ModifierGroups[1].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[2].Initialize(true, 1, &MRFMULSAT);
		ModifierGroups[3].Initialize(true, 1, &MRS);
	}
}IRFMUL;

struct InstructionRuleFMUL32I: InstructionRule
{
	InstructionRuleFMUL32I(): InstructionRule("FMUL32I", 3, true, false)
	{
		hpBinaryStringToOpcode8("0100 000000 1110 000000 000000 00000000000000000000000000000000 001100", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRRegister1,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 1, &MRFMUL32IFTZ);
		ModifierGroups[1].Initialize(true, 1, &MRFMULSAT);
		ModifierGroups[2].Initialize(true, 1, &MRS);
					
	}
}IRFMUL32I;


struct InstructionRuleFFMA: InstructionRule
{
	InstructionRuleFFMA(): InstructionRule("FFMA", 4, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000111000000000000000000000000000000000000000000000001100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister0,
					&OPRRegister1,
					&OPRFFMAAllowNegative, 
					&OPRMAD3);
		ModifierGroups[0].Initialize(true, 1, &MRFMUL32IFTZ);
		ModifierGroups[1].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[2].Initialize(true, 1, &MRFMULSAT);
		ModifierGroups[3].Initialize(true, 1, &MRS);
	}
}IRFFMA;

struct InstructionRuleFSETP: InstructionRule
{
	InstructionRuleFSETP(): InstructionRule("FSETP", 4, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000 1110 111 000 000000 0000000000000000000000 0 1110 000000 00100", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRFADD32IReg1, 
					&OPRFADDCompositeWithOperator,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 14, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE,
					&MRSETPComparisonNUM,
					&MRSETPComparisonNAN,
					&MRSETPComparisonLTU,
					&MRSETPComparisonEQU,
					&MRSETPComparisonLEU,
					&MRSETPComparisonGTU,
					&MRSETPComparisonNEU,
					&MRSETPComparisonGEU);
		ModifierGroups[1].Initialize(true, 1, &MRFSETPFTZ);
		ModifierGroups[2].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
		ModifierGroups[3].Initialize(true, 1, &MRS);
	}
}IRFSETP;

struct InstructionRuleFCMP: InstructionRule
{
	InstructionRuleFCMP(): InstructionRule("FCMP", 2, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000 1110 000000 000000 0000000000000000000000 0 000000 0000 11100", OpcodeWord0, OpcodeWord1);
		SetOperands(4, 
					&OPRRegister0,
					&OPRRegister1,
					&OPRFMULStyle, 
					&OPRRegister3ForCMP);
		ModifierGroups[0].Initialize(false, 14, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE,
					&MRSETPComparisonNUM,
					&MRSETPComparisonNAN,
					&MRSETPComparisonLTU,
					&MRSETPComparisonEQU,
					&MRSETPComparisonLEU,
					&MRSETPComparisonGTU,
					&MRSETPComparisonNEU,
					&MRSETPComparisonGEU);
		ModifierGroups[1].Initialize(true, 1, &MRFADD32IFTZ);
	}
}IRFCMP;

struct InstructionRuleMUFU: InstructionRule
{
	InstructionRuleMUFU(): InstructionRule("MUFU", 2, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000 1110 000000 000000 0000 0000000000000000000000000000 010011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRFADD32IReg1);
		ModifierGroups[0].Initialize(false, 8, 
					&MRMUFUCOS,
					&MRMUFUSIN,
					&MRMUFUEX2,
					&MRMUFULG2,
					&MRMUFURCP,
					&MRMUFURSQ,
					&MRMUFURCP64H,
					&MRMUFURSQ64H);
		ModifierGroups[1].Initialize(true, 1, &MRFADD32IFTZ);
	}
}IRMUFU;


struct InstructionRuleDADD: InstructionRule
{
	InstructionRuleDADD() : InstructionRule("DADD", 2, true, false)
	{
		hpBinaryStringToOpcode8("1000 000000 1110 000000 000000 0000000000000000000000 0000000000 010010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0ForDouble,
					&OPRRegister1ForDoubleWith2OP, 
					&OPRCompositeForDoubleWith2OP);
		ModifierGroups[0].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[1].Initialize(true, 1, &MRS);

	}
}IRDADD;


struct InstructionRuleDMUL: InstructionRule
{
	InstructionRuleDMUL(): InstructionRule("DMUL", 2, true, false)
	{
		hpBinaryStringToOpcode8("1000 000000 1110 000000 000000 0000000000000000000000 0000000000 001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0ForDouble,
					&OPRRegister1ForDouble,
					&OPRCompositeForDoubleWith1OP);
		ModifierGroups[0].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[1].Initialize(true, 1, &MRS);
	}
}IRDMUL;

struct InstructionRuleDFMA: InstructionRule
{
	InstructionRuleDFMA(): InstructionRule("DFMA", 1, true, false)
	{
		hpBinaryStringToOpcode8("1000 000000 1110 000000 000000 0000000000000000000000 0 000000  000 000100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister0ForDouble,
					&OPRRegister1ForDouble,
					&OPRCompositeForDoubleWith1OP, 
					&OPRRegister3ForDouble);
		ModifierGroups[0].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
	}
}IRDFMA;


struct InstructionRuleDSETP: InstructionRule
{
	InstructionRuleDSETP(): InstructionRule("DSETP", 2, true, false)
	{
		hpBinaryStringToOpcode8("1000 000000 1110 111 000 000000 0000000000000000000000 0 1110 000000 11000", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRRegister1ForDoubleWith2OP, 
					&OPRCompositeForDoubleWith2OP,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 14, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE,
					&MRSETPComparisonNUM,
					&MRSETPComparisonNAN,
					&MRSETPComparisonLTU,
					&MRSETPComparisonEQU,
					&MRSETPComparisonLEU,
					&MRSETPComparisonGTU,
					&MRSETPComparisonNEU,
					&MRSETPComparisonGEU);
		ModifierGroups[1].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRDSETP;

struct InstructionRuleFMNMX: InstructionRule
{
	InstructionRuleFMNMX() : InstructionRule("FMNMX", 1, true, false)
	{
		hpBinaryStringToOpcode8("0000 0000 0011 1000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 0000", OpcodeWord0, OpcodeWord1);
		SetOperands(4, 
					&OPRRegister0,
					&OPRFADD32IReg1, 
					&OPRFADDCompositeWithOperator,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(true, 1, &MRFADD32IFTZ);
	}
}IRFMNMX;

struct InstructionRuleFSET: InstructionRule
{
	InstructionRuleFSET(): InstructionRule("FSET", 4, true, false)
	{
		hpBinaryStringToOpcode8("0000 000000 1110 000 000 000000 0000000000000000000000 0 1110 000000 01000", OpcodeWord0, OpcodeWord1);
		SetOperands(4, 
					&OPRRegister0,
					&OPRRegister1,
					&OPRRegister2,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(true, 1, &MRFSETBF);
		ModifierGroups[1].Initialize(false, 14, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE,
					&MRSETPComparisonNUM,
					&MRSETPComparisonNAN,
					&MRSETPComparisonLTU,
					&MRSETPComparisonEQU,
					&MRSETPComparisonLEU,
					&MRSETPComparisonGTU,
					&MRSETPComparisonNEU,
					&MRSETPComparisonGEU);
		ModifierGroups[2].Initialize(true, 1, &MRFSETPFTZ);
		ModifierGroups[3].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRFSET;

struct InstructionRuleDMNMX: InstructionRule
{
	InstructionRuleDMNMX() : InstructionRule("DMNMX", 0, true, false)
	{
		hpBinaryStringToOpcode8("1000 0000 0011 1000 0000 0000 0000 0000 0000 0000 0000 0000 0111 0000 0001 0000", OpcodeWord0, OpcodeWord1);
		SetOperands(4, 
					&OPRRegister0,
					&OPRRegister1,
					&OPRRegister2,
					&OPRPredicate2);
	}
}IRDMNMX;

struct InstructionRuleRRO: InstructionRule
{
	InstructionRuleRRO() : InstructionRule("RRO", 1, true, false)
	{
		hpBinaryStringToOpcode8("0000 0000 0011 1000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0110", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,
					&OPRRegister2);
		ModifierGroups[0].Initialize(false, 2,
					&MRRROSINCOS,
					&MRRROEX2);
	}
}IRRRO;

struct InstructionRuleDSET: InstructionRule
{
	InstructionRuleDSET() : InstructionRule("DSET", 2, true, false)
	{
		hpBinaryStringToOpcode8("1000 0000 0011 1000 0000 0000 0000 0000 0000 0000 0000 0000 0111 0000 0000 1000", OpcodeWord0, OpcodeWord1);
		SetOperands(4, 
					&OPRRegisterWithCCAt16,
					&OPRRegister1,
					&OPRRegister2,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 14,
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE,
					&MRSETPComparisonNUM,
					&MRSETPComparisonNAN,
					&MRSETPComparisonLTU,
					&MRSETPComparisonEQU,
					&MRSETPComparisonLEU,
					&MRSETPComparisonGTU,
					&MRSETPComparisonNEU,
					&MRSETPComparisonGEU);
		ModifierGroups[1].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRDSET;
