#include "../DataTypes.h"
#include "../GlobalVariables.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "../RulesOperand.h"
#include "RulesOperandComposite.h"
#include "RulesOperandRegister.h"
#include "RulesOperandOthers.h"

//ignored operand: currently used for NOP
struct OperandRuleIgnored: OperandRule
{
	OperandRuleIgnored() : OperandRule(Optional){}
	virtual void Process(SubString &component)
	{
		//do nothing
	}
}OPRIgnored;

struct OperandRuleSCHI: OperandRule
{
	unsigned Offset;
	OperandRuleSCHI(unsigned offset): OperandRule(Custom)
	{
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		unsigned result;
		if(component.IsHex())
			result = component.ToImmediate32FromHexConstant(false);
		else
			result = component.ToImmediate32FromBinary();
		if(result>0xff)
			throw 155; //cannot exceed 0xff
		if(Offset < 32)
		{
			csCurrentInstruction.OpcodeWord0 |= result<< Offset;
			if(Offset > 24)
				csCurrentInstruction.OpcodeWord1 |= result>> 32 - Offset;
		}
		else
			csCurrentInstruction.OpcodeWord1 |= result<< Offset - 32;
	}
};
OperandRuleSCHI OPRSCHI0(4),
				OPRSCHI1(12),
				OPRSCHI2(20),
				OPRSCHI3(28),
				OPRSCHI4(36),
				OPRSCHI5(44),
				OPRSCHI6(52);



struct OperandRule32I: OperandRule
{
	//this constructor is not really so useful. However, Optional operand can be indicated
	//here with a type Optional instead of Custom
	OperandRule32I() : OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		int startPos = 0;
		//floating point number expression
		if(component[0]=='F')
		{
			result = component.ToImmediate32FromFloatConstant();
			goto write;
		}

		//'-' here is not operator. It's part of a constant expression
		if(component[0]=='-')
			startPos=1;
		//hex constant
		if(component.Length-startPos>2 && component[startPos] == '0' && (component[startPos+1]=='x' || component[startPos+1]=='X'))
		{
			result = component.ToImmediate32FromHexConstant(true);
		}
		//int
		else
		{
			result = component.ToImmediate32FromIntConstant();
		}
		write:
		WriteToImmediate32(result);
	}
}OPR32I;





struct OperandRuleLOP: OperandRule
{
	int ModShift;
	OperandRuleLOP(int modShift): OperandRule(Custom)
	{
		ModShift = modShift;
	}
	virtual void Process(SubString &component)
	{
		bool negate = false;
		if(component[0]=='~')
		{
			negate = true;
			component.Start++;
			component.Length--;
		}
		if(component.Length<1)
			throw 132; //empty operand
		if(ModShift==8)
		{
			if(component[0]=='c')
			{
				SetConstMem(component, 0x1f, true);
				MarkConstantMemoryForImmediate32();
			}
			else
				((OperandRule*)&OPRMOVStyle)->Process(component);
		}
		else
			((OperandRule*)&OPRRegister1)->Process(component);
		if(negate)
		{
			csCurrentInstruction.OpcodeWord0 |= 1<<ModShift;
			component.Start--;
			component.Length++;
		}
	}
}OPRLOP1(9), OPRLOP2(8);


struct OperandRuleF2I: OperandRule
{
	int OPType;
	OperandRuleF2I(int type): OperandRule(Custom)
	{
		OPType = type;
	}
	virtual void Process(SubString &component)
	{
		bool operated = false;
		if(component[0]=='-')
		{
			operated = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<8;
			component.Start++;
			component.Length--;
		}
		else if(component[0]=='|')
		{
			operated = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<6;
			component.Start++;
			component.Length--;
		}
		switch(OPType) {
		case 0: // F2I
			((OperandRule*)&OPRFMULStyle)->Process(component);
			break;
		case 1: // I2F
			((OperandRule*)&OPRIMULStyle)->Process(component);
			break;
		case 2: // I2I
			((OperandRule*)&OPRI2IStyle)->Process(component);
			break;
		}
		if(operated)
		{
			component.Start--;
			component.Length++;
		}
	}
}OPRF2I(0), OPRI2F(1), OPRI2I(2);


struct OperandRuleISCADDShift: OperandRule
{
	OperandRuleISCADDShift(): OperandRule(Custom)
	{
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		if(component.IsHex())
			result = component.ToImmediate32FromHexConstant(false);
		else
			result = component.ToImmediate32FromInt32();
		if(result>=32)
			throw 133;//shift can be no larger than 31
		csCurrentInstruction.OpcodeWord0 |= result << 5; //assumes that the opcode0 has unwritten field
	}
}OPRISCADDShift;


struct OperandRuleNOPCC: OperandRule
{
	bool Initialized;
	SortElement *SortedList;
	unsigned int *IndexList;
	unsigned int ElementCount;
	void Initialize()
	{
		Initialized = true;
		list<SortElement> sElements;

		sElements.push_back(SortElement((void*)0,"F"));
		sElements.push_back(SortElement((void*)1,"LT")); 
		sElements.push_back(SortElement((void*)2,"EQ")); 
		sElements.push_back(SortElement((void*)3,"LE")); 
		sElements.push_back(SortElement((void*)4,"GT")); 
		sElements.push_back(SortElement((void*)5,"NE")); 
		sElements.push_back(SortElement((void*)6,"GE")); 
		sElements.push_back(SortElement((void*)7,"NUM")); 
		sElements.push_back(SortElement((void*)8,"NAN")); 
		sElements.push_back(SortElement((void*)9,"LTU")); 
		sElements.push_back(SortElement((void*)10,"EQU")); 
		sElements.push_back(SortElement((void*)11,"LEU")); 
		sElements.push_back(SortElement((void*)12,"GTU")); 
		sElements.push_back(SortElement((void*)13,"NEU")); 
		sElements.push_back(SortElement((void*)14,"GEU")); 
		sElements.push_back(SortElement((void*)15,"T")); 
		sElements.push_back(SortElement((void*)16,"OFF")); 
		sElements.push_back(SortElement((void*)17,"LO")); 
		sElements.push_back(SortElement((void*)18,"SFF")); 
		sElements.push_back(SortElement((void*)19,"LS")); 
		sElements.push_back(SortElement((void*)20,"HI")); 
		sElements.push_back(SortElement((void*)21,"SFT")); 
		sElements.push_back(SortElement((void*)22,"HS"));
		sElements.push_back(SortElement((void*)23,"OFT")); 
		sElements.push_back(SortElement((void*)24,"CSM_TA")); 
		sElements.push_back(SortElement((void*)25,"CSM_TR")); 
		sElements.push_back(SortElement((void*)26,"CSM_MX")); 
		sElements.push_back(SortElement((void*)27,"FCSM_TA")); 
		sElements.push_back(SortElement((void*)28,"FCSM_TR")); 
		sElements.push_back(SortElement((void*)29,"FCSM_MX")); 
		sElements.push_back(SortElement((void*)30,"RLE")); 
		sElements.push_back(SortElement((void*)31,"RGT")); 
		SortInitialize(sElements, SortedList, IndexList);
		ElementCount = sElements.size();
		Initialized = true;
	}
	OperandRuleNOPCC(): OperandRule(Optional)
	{
		Initialized = false;
	}
	virtual void Process(SubString &component)
	{
		if(component.Length<4 || component[0]!='C' || component[1] != 'C' || component[2] != '.')
			throw 135;//incorrect NOP operand
		SubString mod = component.SubStr(3, component.Length - 3);
		
		if(!Initialized)
			Initialize();

		SortElement found = SortFind(SortedList, IndexList, ElementCount, mod);
		if(found.ExtraInfo==SortNotFound.ExtraInfo)
			throw 135;


		unsigned int type = *((unsigned int*)&found.ExtraInfo);
		csCurrentInstruction.OpcodeWord0 &= ~(15<<5);
		csCurrentInstruction.OpcodeWord0 |= type<<5;
	}
	~OperandRuleNOPCC()
	{
		if(Initialized)
		{
			delete[] IndexList;
			delete[] SortedList;
		}
	}
}OPRNOPCC;

struct OperandRuleTEXImm: OperandRule
{
	int Offset;
	unsigned int Mask;

	OperandRuleTEXImm(int offset, unsigned int mask) : OperandRule(Custom)
	{
		Offset = offset;
		Mask = mask;
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		result = component.ToImmediate32FromHexConstant(false);
		if (Offset >= 32) {
			csCurrentInstruction.OpcodeWord1 |=
				(result & Mask)<<(Offset-32);
		} else {
			csCurrentInstruction.OpcodeWord0 |=
				(result & Mask)<<Offset;
		}
	}
} OPRTEXImm2(32, 0xff), OPRTEXImm3(40, 0x1f), OPRTEXImm5(46, 0xf),
	OPRTEXDEPBARImm(26, 0x3f);

struct OperandRuleTEXGeom: OperandRule
{
	bool Initialized;
	SortElement *SortedList;
	unsigned int *IndexList;
	unsigned int ElementCount;
	OperandRuleTEXGeom(): OperandRule(Custom)
	{
		Initialized = false;
	}
	void TEXGeomInit()
	{
		list<SortElement> Geoms;
		Geoms.push_back(SortElement((void*)0, "1D"));
		Geoms.push_back(SortElement((void*)1, "ARRAY_1D"));
		Geoms.push_back(SortElement((void*)2, "RECT"));
		Geoms.push_back(SortElement((void*)3, "ARRAY_2D"));
		Geoms.push_back(SortElement((void*)4, "3D"));
		Geoms.push_back(SortElement((void*)6, "CUBE"));
		Geoms.push_back(SortElement((void*)7, "ARRAY_CUBE"));
		ElementCount = Geoms.size();
		SortInitialize(Geoms, SortedList, IndexList);
		Initialized = true;
	}
	virtual void Process(SubString &component)
	{
		if(!Initialized) {
			TEXGeomInit();
		}
		component.RemoveBlankAtEnd();
		if(!component.Length) {
			throw 128;
		}
		unsigned int result;
		SortElement found = SortFind(SortedList, IndexList, ElementCount, component);
		if(found.ExtraInfo==SortNotFound.ExtraInfo) {
			throw 128;
		}
		else {
			result = *((unsigned int*)&found.ExtraInfo);
		}
		csCurrentInstruction.OpcodeWord1 |= result<<(51-32);
	}
	~OperandRuleTEXGeom()
	{
		if(Initialized)
		{
			delete[] SortedList;
			delete[] IndexList;
		}
	}
} OPRTEXGeom;

struct OperandRuleBRACCOrInstAddr: OperandRule
{
	OperandRuleBRACCOrInstAddr(): OperandRule(Custom)
	{
	}
	virtual void Process(SubString &component)
	{
		try {
			((struct OperandRule *)&OPRNOPCC)->Process(component);
		} catch (int e) {
			if (e != 135) {
				throw e;
			}
			((struct OperandRule *)&OPRInstructionAddress)->Process(component);
		}
	}
} OPRBRACCOrInstAddr;

struct OperandRuleSUCLAMPImm: OperandRule
{
	OperandRuleSUCLAMPImm(): OperandRule(Custom)
	{
	}
	virtual void Process(SubString &component)
	{
		unsigned result;
		if(component.IsHex())
			result = component.ToImmediate32FromHexConstant(false);
		else
			throw 106;
		csCurrentInstruction.OpcodeWord1 &= 0xff81ffff;
		csCurrentInstruction.OpcodeWord1 |= (result&0x3f)<<(49-32);

		// set correct values for (SD|PL|BL).R(1|2|4|8|16)
		unsigned sucm1 = (csCurrentInstruction.OpcodeWord0>>7)&3;
		unsigned sucm2 = (csCurrentInstruction.OpcodeWord0>>4)&7;
		csCurrentInstruction.OpcodeWord0 &= 0xfffffe0f;
		sucm1 = sucm1*5 + sucm2;
		csCurrentInstruction.OpcodeWord0 |= (sucm1<<5);
	}
} OPRSUCLAMPImm;
