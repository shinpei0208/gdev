
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesModifierConversion.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"


void X2XRegCheck(ModifierRule *rule, bool is64, int compPos)
{
	ApplyModifierRuleUnconditional(rule);
	if(!is64)
		return;
	if(csCurrentInstruction.Predicated)
		compPos++;
	if(csCurrentInstruction.Components.size()<=compPos)
		throw 103;//insufficient operands
	list<SubString>::iterator component = csCurrentInstruction.Components.begin();
	for(int i =0; i<compPos; i++)
		component++;
	if(component->IsRegister())
	{
		int reg;
		reg = component->ToRegister();
		if(reg==63)return;
		reg++;
		if(reg==63) //will be wrongly ignored 
			throw 147; //cannot be used with 64
		CheckRegCount(reg);
	}
}


struct ModifierRuleF2IDest: ModifierRule
{
	bool Is64;
	ModifierRuleF2IDest(int type, bool sign): ModifierRule("", true, false, true)
	{
		Is64 = type==3;
		hpBinaryStringToOpcode4("11111110111111111111001111111111", Mask0);
		Bits0 = type << 20;
		Bits0 |= (int)sign << 7;
		if(sign)
		{
			if(type==0)
				Name = "S8";
			else if(type==1)
				Name = "S16";
			else if(type==2)
				Name = "S32";
			else if(type==3)
				Name = "S64";
			else
				throw exception();
		}
		else
		{
			if(type==0)
				Name = "U8";
			else if(type==1)
				Name = "U16";
			else if(type==2)
				Name = "U32";
			else if(type==3)
				Name = "U64";
			else
				throw exception();
		}
	}
	
	virtual void CustomProcess()
	{
		X2XRegCheck(this, Is64, 1);
	}
}	MRF2IDestU8(0, false), 
	MRF2IDestU16(1, false), 
	MRF2IDestU32(2, false), 
	MRF2IDestU64(3, false), 
	MRF2IDestS8(0, true), 
	MRF2IDestS16(1, true), 
	MRF2IDestS32(2, true), 
	MRF2IDestS64(3, true);

struct ModifierRuleF2FPASS: ModifierRule
{
	ModifierRuleF2FPASS(bool pass): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 111011 1111 111111 111111 111111", Mask0);
		if(!pass)
		{
			Bits0 |= 1<<7;
			Name = "ROUND";
		}
		else
			Name = "PASS";
	}
}MRF2FPASS(true), MRF2FROUND(false);

struct ModifierRuleF2ISource: ModifierRule
{
	bool Is64;
	ModifierRuleF2ISource(int type): ModifierRule("", true, false, true)
	{
		Is64 = type==3;
		hpBinaryStringToOpcode4("1111 111111 1111 111111 111001 111111", Mask0);
		Bits0 = type<<23;
		if(type==1)
			Name = "F16";
		else if(type==2)
			Name = "F32";
		else if(type==3)
			Name = "F64";
		else throw exception();
	}	
	virtual void CustomProcess()
	{
		X2XRegCheck(this, Is64, 2);
	}
}MRF2ISourceF16(1),MRF2ISourceF32(2),MRF2ISourceF64(3);

struct ModifierRuleF2IRound: ModifierRule
{
	ModifierRuleF2IRound(int type, bool f2f): ModifierRule("", true, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111001111111111111", Mask1);
		Bits1 = type <<17;
		Mask0 = 0xffffffff;
		Bits0 = 0;
		if(!f2f)
		{
			if(type >= 4) {
				// set ROUND for sm_30
				Mask0 = 0xffffff7f;
				Bits0 = 1<<7;
				type -= 4;
				Bits1 = type <<17;
			}
			if(type==1)
				Name = "FLOOR";
			else if(type==2)
				Name = "CEIL";
			else if(type==3)
				Name = "TRUNC";
			else throw exception();
		}
		else
		{
			if(type==1)
				Name = "RM";
			else if(type==2)
				Name = "RP";
			else if(type==3)
				Name = "RZ";
			else throw exception();
		}
	}
}MRF2IFLOOR(1, false), MRF2ICEIL(2, false), MRF2ITRUNC(3, false),
	MRF2FFLOOR(5, false), MRF2FCEIL(6, false), MRF2FTRUNC(7, false),
	MRF2FRM(1, true), MRF2FRP(2, true), MRF2FRZ(3, true);

struct ModifierRuleF2IFTZ: ModifierRule
{
	ModifierRuleF2IFTZ(): ModifierRule("FTZ", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask1);
		Bits1 = 1 <<23;
	}
}MRF2IFTZ;



struct ModifierRuleI2FSource: ModifierRule
{
	bool Is64;
	ModifierRuleI2FSource(int type, bool sign): ModifierRule("", true, false, true)
	{
		Is64 = type==3;
		hpBinaryStringToOpcode4("1111 111110 1111 111111 111001 111111", Mask0);
		Bits0 = type << 23;
		Bits0 |= (int)sign << 9;
		if(sign)
		{
			if(type==0)
				Name = "S8";
			else if(type==1)
				Name = "S16";
			else if(type==2)
				Name = "S32";
			else if(type==3)
				Name = "S64";
			else
				throw exception();
		}
		else
		{
			if(type==0)
				Name = "U8";
			else if(type==1)
				Name = "U16";
			else if(type==2)
				Name = "U32";
			else if(type==3)
				Name = "U64";
			else
				throw exception();
		}
	}
	virtual void CustomProcess()
	{
		X2XRegCheck(this, Is64, 2);
	}
}	MRI2FSourceU8(0, false),
	MRI2FSourceU16(1, false),
	MRI2FSourceU32(2, false), 
	MRI2FSourceU64(3, false), 
	MRI2FSourceS8(0, true), 
	MRI2FSourceS16(1, true), 
	MRI2FSourceS32(2, true), 
	MRI2FSourceS64(3, true);

struct ModifierRuleI2FDest: ModifierRule
{
	bool Is64;
	ModifierRuleI2FDest(int type): ModifierRule("", true, false, true)
	{
		Is64 = type==3;
		hpBinaryStringToOpcode4("11111111111111111111001111111111", Mask0);
		Bits0 = type<<20;
		if(type==1)
			Name = "F16";
		else if(type==2)
			Name = "F32";
		else if(type==3)
			Name = "F64";
		else throw exception();
	}
	virtual void CustomProcess()
	{
		X2XRegCheck(this, Is64, 1);
	}
}MRI2FDestF16(1),MRI2FDestF32(2),MRI2FDestF64(3);

struct ModifierRuleI2FRound: ModifierRule
{
	ModifierRuleI2FRound(int type): ModifierRule("", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111001111111 111111", Mask1);
		Bits1 = type <<17;
		if(type==1)
			Name = "RM";
		else if(type==2)
			Name = "RP";
		else if(type==3)
			Name = "RZ";
		else throw exception();
	}
}MRI2FRM(1), MRI2FRP(2), MRI2FRZ(3);
