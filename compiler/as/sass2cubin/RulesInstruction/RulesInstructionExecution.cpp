#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesInstructionExecution.h"
#include "../RulesOperand/RulesOperandComposite.h"
#include "../RulesOperand/RulesOperandRegister.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"


struct InstructionRuleSCHI: InstructionRule
{
	InstructionRuleSCHI(): InstructionRule("SCHI", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000000000000000000000000000000000000000000000000000 000100", OpcodeWord0, OpcodeWord1);
		SetOperands(7,	&OPRSCHI0, 
						&OPRSCHI1,
						&OPRSCHI2,
						&OPRSCHI3,
						&OPRSCHI4,
						&OPRSCHI5,
						&OPRSCHI6);
	}
}IRSCHI;

struct InstructionRuleEXIT: InstructionRule
{
	InstructionRuleEXIT() : InstructionRule("EXIT", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IREXIT;

struct InstructionRuleCAL: InstructionRule
{
	InstructionRuleCAL() : InstructionRule("CAL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 0000 001000 000000 00000000000000000000000000000000 001010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress);
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRCAL;

struct InstructionRuleJCAL: InstructionRule
{
	InstructionRuleJCAL() : InstructionRule("JCAL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 0000 001000 000000 000000000000000000000000 00000000 001000", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress); //absolute address only
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRJCAL;



struct InstructionRuleSSY: InstructionRule
{
	InstructionRuleSSY() : InstructionRule("SSY", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 0000 000000 000000 000000000000000000000000 00000000 000110", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress);
	}
}IRSSY;

struct InstructionRuleBRA: InstructionRule
{
	InstructionRuleBRA() : InstructionRule("BRA", 2, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110 1110 000000 00000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
		// TODO: we need InstructionAddress, but do not check...
		SetOperands(2, &OPRBRACCOrInstAddr, &OPRInstAddrOptional);
		ModifierGroups[0].Initialize(true, 1, &MRBRAU);
		ModifierGroups[1].Initialize(true, 1, &MRBRALMT);
	}
}IRBRA;

struct InstructionRuleJMP: InstructionRule
{
	InstructionRuleJMP() : InstructionRule("JMP", 2, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110 1110 000000 000000 000000000000000000000000 00000000 000000", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress); //absolute address only
		ModifierGroups[0].Initialize(true, 1, &MRBRAU);
		ModifierGroups[1].Initialize(true, 1, &MRBRALMT);
	}
}IRJMP;


struct InstructionRulePRET: InstructionRule
{
	InstructionRulePRET() : InstructionRule("PRET", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 0000 001000 00000000000000000000000000000000000000011110", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress);
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRPRET;



struct InstructionRuleRET: InstructionRule
{
	InstructionRuleRET() : InstructionRule("RET", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
	}
}IRRET;

struct InstructionRulePBK: InstructionRule
{
	InstructionRulePBK() : InstructionRule("PBK", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 0000 0 00000 000000 000000000000000000000000 00000000 010110", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress);
	}
}IRPBK;


struct InstructionRuleBRK: InstructionRule
{
	InstructionRuleBRK() : InstructionRule("BRK", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110 1110 000000 000000 00000000000000000000000000000000 010101", OpcodeWord0, OpcodeWord1);
	}
}IRBRK;

struct InstructionRulePCNT: InstructionRule
{
	InstructionRulePCNT() : InstructionRule("PCNT", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 0000 0 00000 000000 000000000000000000000000 00000000 001110", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress);
	}
}IRPCNT;


struct InstructionRuleCONT: InstructionRule
{
	InstructionRuleCONT() : InstructionRule("CONT", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110 1110 000000 000000 00000000000000000000000000000000 001101", OpcodeWord0, OpcodeWord1);
	}
}IRCONT;


struct InstructionRulePLONGJMP: InstructionRule
{
	InstructionRulePLONGJMP() : InstructionRule("PLONGJMP", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000 1110 000000 000000 000000000000000000000000 00000000 011010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRInstructionAddress);
	}
}IRPLONGJMP;


struct InstructionRuleLONGJMP: InstructionRule
{
	InstructionRuleLONGJMP() : InstructionRule("LONGJMP", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110 1110 000000 000000 00000000000000000000000000000000 010001", OpcodeWord0, OpcodeWord1);
	}
}IRLONGJMP;




struct InstructionRuleNOP: InstructionRule
{
	InstructionRuleNOP(): InstructionRule("NOP", 3, true, false)
	{
		hpBinaryStringToOpcode8("0010 011110 111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRNOPCC, &OPRImmediate16HexOrIntOptional);
		ModifierGroups[0].Initialize(true, 1, &MRNOPTRIG);
		ModifierGroups[1].Initialize(true, 8, 
										&MRNOPFMA64,
										&MRNOPFMA32,
										&MRNOPXLU  ,
										&MRNOPALU  ,
										&MRNOPAGU  ,
										&MRNOPSU   ,
										&MRNOPFU   ,
										&MRNOPFMUL);
		ModifierGroups[2].Initialize(true, 1, &MRS);
	}
}IRNOP;

struct InstructionRuleBAR: InstructionRule
{
	InstructionRuleBAR(): InstructionRule("BAR", 0, true, true)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000000 000000 111111000000 00000000   00 0 1110 111 00 001010", OpcodeWord0, OpcodeWord1);
	}
	virtual void CustomProcess()
	{
		int nComp = csCurrentInstruction.Components.size();
		int nMod = csCurrentInstruction.Modifiers.size();
		std::list<SubString>::iterator component = csCurrentInstruction.Components.begin();
		
		//skip instruction name/predicate
		nComp--;
		component++;
		if(csCurrentInstruction.Predicated)
		{
			nComp--;
			component++;
		}

		//start reading modifier to determine whether it's ARV or RED
		if(nMod==0)
			throw 127; //insufficient number of modifiers
		std::list<SubString>::iterator modifier = csCurrentInstruction.Modifiers.begin();
		//.arrive. single modifier, 3 components
		if(modifier->Compare("ARV"))
		{
			if(nMod>1)
				throw 122;//too many modifiers present
			if(nComp>2)
				throw 102;//too many operands
			else if(nComp<2)
				throw 103;//insufficient number of operands

			//start processing for ARV
			csCurrentInstruction.OpcodeWord0 |= 1 << 7; //mod 2
			//reg0
			csCurrentInstruction.OpcodeWord0 |= 63 << 14;
			//bar, tcount
			((OperandRule*)&OPRBAR)->Process(*component);
			component++;
			((OperandRule*)&OPRTCount)->Process(*component);
		}
		//.RED.POPC: 2 - 4 operands
		//.RED.LogicOp: 3-5 operands
		else if(modifier->Compare("RED"))
		{
			if(nMod<2)
				throw 127;
			else if(nMod>2)
				throw 122;
			//number of operands will depend on .Op
			modifier++;
			unsigned int Op;
			bool popc = false;
			//POPC
			if(modifier->Compare("POPC"))
			{
				popc = true;
				Op = 0;
				if(nComp<2)
					throw 103;
				if(nComp>4)
					throw 102;

			}
			//AND and OR
			else if(modifier->Compare("AND"))
				Op = 1;
			else if(modifier->Compare("OR"))
				Op = 2;
			else
				throw 101;
			//reg0
			((OperandRule*)&OPRRegister0)->Process(*component);
			component++;
			nComp--;


			if(!popc)
			{
				if(nComp<3)
					throw 103;
				if(nComp>5)
					throw 102;
				//p
				((OperandRule*)&OPRPredicateForBAR)->Process(*component);
				component++;
				nComp--;
			}
			//bar (,tcount) (,(!)c)
			((OperandRule*)&OPRBAR)->Process(*component);
			component++;
			nComp--;
			if(nComp!=0&&(*component)[0]!='!'&&(*component)[0]!='p'&&(*component)[0]!='P')
			{
				((OperandRule*)&OPRTCount)->Process(*component);
				component++;
				nComp--;
			}
			if(nComp!=0)
			{
				((OperandRule*)&OPRPredicate2)->Process(*component);
				component++;
				nComp--;
			}
			if(nComp!=0)
				throw 142; //unrecognised operand at the end

			//set .Op
			csCurrentInstruction.OpcodeWord0 |= Op << 5;
		}
		else if(modifier->Compare("SYNC"))
		{
			if(nMod != 1) {
				throw (nMod > 1) ? 122:127;
			}
			if(nComp != 1) {
				throw (nComp > 1) ? 102:103;
			}
			// bar.sync Imm
			csCurrentInstruction.OpcodeWord1 &= 0xf80ec000;
			csCurrentInstruction.OpcodeWord0 &= 0x03f03fff;
			((OperandRule*)&OPRBAR)->Process(*component);
			csCurrentInstruction.OpcodeWord1 |= 0xc000;
		}
		else throw 101;//unsupported modifier
	}
}IRBAR;

struct InstructionRuleB2R: InstructionRule
{
	InstructionRuleB2R(): InstructionRule("B2R", 2, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000000 000000 00000000000000000000000000000000 011100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRRegister0, &OPRBARNoRegister);
		ModifierGroups[0].Initialize(true, 2, &MRXLU, &MRALU);
		ModifierGroups[1].Initialize(true, 1, &MRS);
	}
}IRB2R;

struct InstructionRuleMEMBAR: InstructionRule
{
	InstructionRuleMEMBAR(): InstructionRule("MEMBAR", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000000 1110 000000 000000 00000000000000000000000000000000 000111", OpcodeWord0, OpcodeWord1);
		ModifierGroups[0].Initialize(false, 3, &MRMEMBARCTA, &MRMEMBARGL, &MRMEMBARSYS);
	}

}IRMEMBAR;

struct InstructionRuleATOM: InstructionRule
{
	InstructionRuleATOM(): InstructionRule("ATOM", 5, true, false)
	{
		hpBinaryStringToOpcode8("1010 000000 1110 000000 000000 00000000000000000 000000 111111 000 0010 10", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister3ForATOM,
					&OPRMemoryForATOM,
					&OPRRegister0,
					&OPRRegister4ForATOM);
		ModifierGroups[0].Initialize(true, 1, &MRE);
		ModifierGroups[1].Initialize(false, 10,
										&MRATOMADD,	
										&MRATOMMIN,
										&MRATOMMAX,
										&MRATOMINC,
										&MRATOMDEC,
										&MRATOMAND,
										&MRATOMOR,
										&MRATOMXOR,
										&MRATOMEXCH,
										&MRATOMCAS);
		ModifierGroups[2].Initialize(true, 3, 
										&MRATOMTypeU64,
										&MRATOMTypeS32,
										&MRATOMTypeF32);
		ModifierGroups[3].Initialize(true, 1, &MRATOMIgnoredFTZ);
		ModifierGroups[4].Initialize(true, 1, &MRATOMIgnoredRN);
	}
}IRATOM;


struct InstructionRuleRED: InstructionRule
{
	InstructionRuleRED(): InstructionRule("RED", 5, true, false)
	{
		hpBinaryStringToOpcode8("1010 000000 1110 000000 000000 00000000000000000000000000000000 0010 00", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRGlobalMemoryWithImmediate32,
					&OPRRegister0);
		ModifierGroups[0].Initialize(true, 1, &MRE);
		ModifierGroups[1].Initialize(false, 8,
										&MRATOMADD,	
										&MRATOMMIN,
										&MRATOMMAX,
										&MRATOMINC,
										&MRATOMDEC,
										&MRATOMAND,
										&MRATOMOR,
										&MRATOMXOR);
		ModifierGroups[2].Initialize(true, 3, 
										&MRATOMTypeU64,
										&MRATOMTypeS32,
										&MRATOMTypeF32);
		ModifierGroups[3].Initialize(true, 1, &MRATOMIgnoredFTZ);
		ModifierGroups[4].Initialize(true, 1, &MRATOMIgnoredRN);
	}
}IRRED;

struct InstructionRuleVOTE: InstructionRule
{
	InstructionRuleVOTE(): InstructionRule("VOTE", 1, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000000 0000 00 0000000000000000000000000000 000 0 010010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRPredicate0ForVOTE,
					&OPRPredicate1ForVOTE);
		ModifierGroups[0].Initialize(false, 4,
										&MRVOTEALL,
										&MRVOTEANY,
										&MRVOTEEQ);
										//&MRVOTEVTG);
		/*ModifierGroups[1].Initialize(true, 3,
										&MRVOTEVTGR,
										&MRVOTEVTGA,
										&MRVOTEVTGRA);*/
	}
}IRVOTE;

struct InstructionRuleBPT: InstructionRule
{
	InstructionRuleBPT(): InstructionRule("BPT", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000000000000000000000000000000000000000000000000000 001011", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate16HexOrIntOptional);
		ModifierGroups[0].Initialize(false, 4,
					&MRBPTDRAIN,
					&MRBPTCAL,
					&MRBPTPAUSE,
					&MRBPTTRAP);
	}
}IRBPT;

struct InstructionRuleBRX: InstructionRule
{
	InstructionRuleBRX() : InstructionRule("BRX", 0, true, true)
	{
		hpBinaryStringToOpcode8("1110 011110 1110 000000 00000000000000000000000000000000000000010010", OpcodeWord0, OpcodeWord1);
	}
	virtual void CustomProcess()
	{
		int nComp = csCurrentInstruction.Components.size();
		std::list<SubString>::iterator component = csCurrentInstruction.Components.begin();
		
		// skip instruction name/predicate
		nComp--;
		component++;
		if (csCurrentInstruction.Predicated) {
			nComp--;
			component++;
		}

		// check
		if (nComp != 1) {
			throw (nComp < 1) ? 103:102;
		}

		// Register0
		int blankPos = component->FindBlank(0);
		if (blankPos == -1) {
			throw 116; // Invalid operand.
		}
		SubString regStr = component->SubStr(0, blankPos);
		((OperandRule*)&OPRRegister0)->Process(regStr);

		// InstructionAddress
		int addrPos = blankPos + 1;
		while ((*component)[addrPos] < 33) {
			addrPos++;
		}
		SubString addrStr = component->SubStr(addrPos, component->Length - addrPos);
		((OperandRule*)&OPRInstructionAddress)->Process(addrStr);
	}
}IRBRX;
