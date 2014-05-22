
#include "../DataTypes.h"
#include "../GlobalVariables.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "../RulesOperand.h"
#include "RulesOperandConstant.h"


//24-bit Hexadecimal Constant Operand
struct OperandRuleImmediate24HexConstant: OperandRule
{
	OperandRuleImmediate24HexConstant() :OperandRule(Immediate32HexConstant){} //issue: wrong type
	virtual void Process(SubString &component)
	{
		bool negate = false;
		if(component[0]=='-')
		{
			negate = true;
			component.Start++;
			component.Length--;
		}
		unsigned int result = component.ToImmediate32FromHexConstant(false);
		if(negate)
		{
			if(result>0x7FFFFF)
				throw 131; //too large offset
			result ^= 0xFFFFFF;
			result += 1;
			component.Start--;
			component.Length++;
		}
		else
		{
			if(result>0xFFFFFF)
				throw 131;
		}
		WriteToImmediate32(result);
	}	
}OPRImmediate24HexConstant;

/*
struct OperandRuleImmediateHexConstant: OperandRule
{
	unsigned int MaxNum;
	OperandRuleImmediateHexConstant(int maxBit)
	{
		MaxNum = 1<<maxBit - 1;
	}
	virtual void Process(SubString &component)
	{
		bool negate = false;		
		if(component[0]=='-')
		{
			negate = true;
			component.Start++;
			component.Length--;
		}
		unsigned int result = component.ToImmediate32FromHexConstant(false);
		
		if(!negate)
		{
			if(result > MaxNum)
				throw 131;
		}
		else
		{
			if(result>MaxNum>>1)
				throw 131;
		}
	}
};
*/

struct OperandRuleImmediate32HexConstant: OperandRule
{
	OperandRuleImmediate32HexConstant() :OperandRule(Immediate32HexConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromHexConstant(true);
		WriteToImmediate32(result);
	}	
}OPRImmediate32HexConstant;



//32-bit Integer Constant Operand
struct OperandRuleImmediate32IntConstant: OperandRule
{
	OperandRuleImmediate32IntConstant():OperandRule(Immediate32IntConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromIntConstant();
		WriteToImmediate32(result);
	}
}OPRImmediate32IntConstant;

//32-bit Floating Number Constant Operand
struct OperandRuleImmediate32FloatConstant: OperandRule
{
	OperandRuleImmediate32FloatConstant():OperandRule(Immediate32FloatConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromFloatConstant();
		WriteToImmediate32(result);
	}	
}OPRImmediate32FloatConstant;

//32-bit Constant: Hex || Int || Float
struct OperandRuleImmediate32AnyConstant: OperandRule
{
	OperandRuleImmediate32AnyConstant():OperandRule(Immediate32AnyConstant){}
	virtual void Process(SubString &component)
	{
		//Issue: not yet implemented
	}	
}OPRImmediate32AnyConstant;



struct OperandRuleS2R: OperandRule
{
	bool Initialized;
	SortElement *SortedList;
	unsigned int *IndexList;
	unsigned int ElementCount;
	OperandRuleS2R(): OperandRule(Custom)
	{
		Initialized = false;
	}
	void S2RInitialize()
	{
		list<SortElement> SRs;
		SRs.push_back(SortElement((void*)0, "SR_LaneId"));
		SRs.push_back(SortElement((void*)2, "SR_VirtCfg"));
		SRs.push_back(SortElement((void*)3, "SR_VirtId"));
		SRs.push_back(SortElement((void*)4, "SR_PM0"));
		SRs.push_back(SortElement((void*)5, "SR_PM1"));
		SRs.push_back(SortElement((void*)6, "SR_PM2"));
		SRs.push_back(SortElement((void*)7, "SR_PM3"));
		SRs.push_back(SortElement((void*)8, "SR_PM4"));
		SRs.push_back(SortElement((void*)9, "SR_PM5"));
		SRs.push_back(SortElement((void*)10, "SR_PM6"));
		SRs.push_back(SortElement((void*)11, "SR_PM7"));
		SRs.push_back(SortElement((void*)16, "SR_PRIM_TYPE"));
		SRs.push_back(SortElement((void*)17, "SR_INVOCATION_ID"));
		SRs.push_back(SortElement((void*)18, "SR_Y_DIRECTION"));
		SRs.push_back(SortElement((void*)24, "SR_MACHINE_ID_0"));
		SRs.push_back(SortElement((void*)25, "SR_MACHINE_ID_1"));
		SRs.push_back(SortElement((void*)26, "SR_MACHINE_ID_2"));
		SRs.push_back(SortElement((void*)27, "SR_MACHINE_ID_3"));
		SRs.push_back(SortElement((void*)28, "SR_AFFINITY"));
		SRs.push_back(SortElement((void*)32, "SR_Tid"));
		SRs.push_back(SortElement((void*)33, "SR_Tid_X"));
		SRs.push_back(SortElement((void*)34, "SR_Tid_Y"));
		SRs.push_back(SortElement((void*)35, "SR_Tid_Z"));
		SRs.push_back(SortElement((void*)36, "SR_CTAParam"));
		SRs.push_back(SortElement((void*)37, "SR_CTAid_X"));
		SRs.push_back(SortElement((void*)38, "SR_CTAid_Y"));
		SRs.push_back(SortElement((void*)39, "SR_CTAid_Z"));
		SRs.push_back(SortElement((void*)40, "SR_NTid"));
		SRs.push_back(SortElement((void*)41, "SR_NTid_X"));
		SRs.push_back(SortElement((void*)42, "SR_NTid_Y"));
		SRs.push_back(SortElement((void*)43, "SR_NTid_Z"));
		SRs.push_back(SortElement((void*)44, "SR_GridParam"));
		SRs.push_back(SortElement((void*)45, "SR_NCTAid_X"));
		SRs.push_back(SortElement((void*)46, "SR_NCTAid_Y"));
		SRs.push_back(SortElement((void*)47, "SR_NCTAid_Z"));
		SRs.push_back(SortElement((void*)48, "SR_SWinLo"));
		SRs.push_back(SortElement((void*)49, "SR_SWINSZ"));
		SRs.push_back(SortElement((void*)50, "SR_SMemSz"));
		SRs.push_back(SortElement((void*)51, "SR_SMemBanks"));
		SRs.push_back(SortElement((void*)52, "SR_LWinLo"));
		SRs.push_back(SortElement((void*)53, "SR_LWINSZ"));
		SRs.push_back(SortElement((void*)54, "SR_LMemLoSz"));
		SRs.push_back(SortElement((void*)55, "SR_LMemHiOff"));
		SRs.push_back(SortElement((void*)56, "SR_EqMask"));
		SRs.push_back(SortElement((void*)57, "SR_LtMask"));
		SRs.push_back(SortElement((void*)58, "SR_LeMask"));
		SRs.push_back(SortElement((void*)59, "SR_GtMask"));
		SRs.push_back(SortElement((void*)60, "SR_GeMask"));
		SRs.push_back(SortElement((void*)80, "SR_ClockLo"));
		SRs.push_back(SortElement((void*)81, "SR_ClockHi"));
		ElementCount = SRs.size();
		SortInitialize(SRs, SortedList, IndexList);
		Initialized = true;
	}
	virtual void Process(SubString &component)
	{
		if(!Initialized)
			S2RInitialize();
		component.RemoveBlankAtEnd();
		if(!component.Length)
			throw 128;
		unsigned int result;
		SortElement found = SortFind(SortedList, IndexList, ElementCount, component);
		if(found.ExtraInfo==SortNotFound.ExtraInfo)
		{
			//try SRnum
			if(component.Length>=3&&component[0]=='S'&&component[1]=='R'&&component[2]!='_')
			{
				int srnum=0;
				try
				{
					srnum = component.SubStr(2, component.Length-2).ToImmediate32FromInt32();
				}
				catch(int e)
				{
					throw 128;
				}
				if(srnum>255||srnum<0)
					throw 128;
				result = (unsigned int)srnum;
			}
			else
				throw 128;
		}
		else
			result = *((unsigned int*)&found.ExtraInfo);
		WriteToImmediate32(result);
		
	}
	~OperandRuleS2R()
	{
		if(Initialized)
		{
			delete[] SortedList;
			delete[] IndexList;
		}
	}
}OPRS2R;


struct OperandRuleImmediate16HexOrInt: OperandRule
{
	OperandRuleImmediate16HexOrInt(bool optional): OperandRule(Custom)
	{
		if(optional)
			Type = Optional;
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		if(component.Length>2 && component[0]=='0' &&(component[1]=='x'||component[1]=='X'))
			result = component.ToImmediate32FromHexConstant(false);
		else
			result = component.ToImmediate32FromInt32();
		if(result>0xffff)
			throw 136;//limited to 16 bits
		WriteToImmediate32(result);
	}
}OPRImmediate16HexOrInt(false), OPRImmediate16HexOrIntOptional(true);