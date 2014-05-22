
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesModifierFloat.h"




struct ModifierRuleFADD32IFTZ: ModifierRule
{
	//ModifierRule(char *name, bool apply0, bool apply1, bool needCustomProcessing)
	ModifierRuleFADD32IFTZ(): ModifierRule("FTZ", true, false, false)
	{
		//Setting the mask. No bits are to be cleared for FTZ, so it's just all 1s
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask0);
		//mod 4 is to be set to 1
		Bits0 = 1<<5;
	}
}MRFADD32IFTZ;


struct ModifierRuleFMUL32IFTZ: ModifierRule
{
	ModifierRuleFMUL32IFTZ(): ModifierRule("FTZ", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask0);
		Bits0 = 1<<6;
	}
}MRFMUL32IFTZ;

struct ModifierRuleFSETPFTZ: ModifierRule
{
	ModifierRuleFSETPFTZ(): ModifierRule("FTZ", false, true, false)
	{
		Mask1 = 0xffffffff;
		Bits1 = 1<<27; // 1<<0x3b
	}
}MRFSETPFTZ;

struct ModifierRuleFMULR: ModifierRule
{
	ModifierRuleFMULR(int type, char* name): ModifierRule("", false, true, false)
	{
		Name = name;
		//2 bits are to be cleared
		hpBinaryStringToOpcode4("11111111111111111111111001111111", Mask1);
		//immeb 1:2 to be set to 10, 01 or 11
		Bits1 = type<<23;
	}
}MRFMULRM(1, "RM"), MRFMULRP(2, "RP"), MRFMULRZ(3, "RZ");


struct ModifierRuleFADDSAT : ModifierRule
{
	ModifierRuleFADDSAT() : ModifierRule("SAT", false, true, false)
	{
		Mask1 = 0xffffffff;
		Bits1 = 1<<17;
	}
}MRFADDSAT;


struct ModifierRuleFMULSAT: ModifierRule
{
	ModifierRuleFMULSAT(): ModifierRule("SAT", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		Bits0 = 1<<5;
	}
}MRFMULSAT;


struct ModifierRuleMUFU: ModifierRule
{
	ModifierRuleMUFU(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 111111 1111 111111 111111 0000 11", Mask0);
		Bits0 = type<<26;
		if(type==0)
			Name = "COS";
		else if(type==1)
			Name = "SIN";
		else if(type==2)
			Name = "EX2";
		else if(type==3)
			Name = "LG2";
		else if(type==4)
			Name = "RCP";
		else if(type==5)
			Name = "RSQ";
		else if(type==6)
			Name = "RCP64H";
		else if(type==7)
			Name = "RSQ64H";
	}
}	MRMUFUCOS(0),
	MRMUFUSIN(1),
	MRMUFUEX2(2),
	MRMUFULG2(3),
	MRMUFURCP(4),
	MRMUFURSQ(5),
	MRMUFURCP64H(6),
	MRMUFURSQ64H(7);

struct ModifierRuleFSETBF: ModifierRule
{
	ModifierRuleFSETBF(): ModifierRule("BF", true, false, false)
	{
		Mask0 = 0xffffffdf;
		Bits0 = 1<<5;
	}
} MRFSETBF;

struct ModifierRuleRRO: ModifierRule
{
	ModifierRuleRRO(int type): ModifierRule("", true, false, false)
	{
		Mask0 = 0xffffffdf;
		Bits0 = 0;
		if(type==0)
			Name = "SINCOS";
		else if(type==1) {
			Name = "EX2";
			Bits0 = 1<<5;
		}
	}
} MRRROSINCOS(0), MRRROEX2(1);
