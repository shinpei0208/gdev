
#include "../DataTypes.h"
#include "../helper/helperMixed.h"


#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesModifierLogic.h"


struct ModifierRuleLOP: ModifierRule
{
	ModifierRuleLOP(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111100111111111111111111111111", Mask0);
		Bits0 = type << 6;
		if(type==0)
			Name = "AND";
		else if(type==1)
			Name = "OR";
		else if(type==2)
			Name = "XOR";
		else if(type ==3)
			Name = "PASS_B";
	}
}MRLOPAND(0), MRLOPOR(1), MRLOPXOR(2), MRLOPPASS(3);

struct ModifierRuleSHR: ModifierRule
{
	ModifierRuleSHR(bool u32): ModifierRule("", true, false, false)
	{
		Bits0 = 0;
		if(u32)
		{
			Name = "U32";
			hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		}
		else
		{
			Name = "W";
			hpBinaryStringToOpcode4("1111 111110 1111111111111111111111", Mask0);
			hpBinaryStringToOpcode4("0000 000001 0000000000000000000000", Bits0);
		}
	}
}MRSHRU32(true), MRSHRW(false);

struct ModifierRuleBFEBREV :ModifierRule
{
	ModifierRuleBFEBREV(): ModifierRule("BREV", true, false, false)
	{
		Mask0 = 0xffffffff;
		Bits0 = 1 << 8;
	}
}MRBFEBREV;
