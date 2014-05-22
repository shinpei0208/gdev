
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesModifierDataMovement.h"


struct ModifierRuleLDType: ModifierRule
{
	ModifierRuleLDType(int type): ModifierRule("", true, false, false)
	{		
		hpBinaryStringToOpcode4("11111000111111111111111111111111", Mask0);
		Bits0 = type<<5;
		switch(type)
		{
		case 0:
			Name = "U8";
			break;
		case 1:
			Name = "S8";
			break;
		case 2:
			Name = "U16";
			break;
		case 3:
			Name = "S16";
			break;
		case 5:
			Name = "64";
			break;
		case 6:
			Name = "128";
			break;
		default:
			throw exception();
			break;
		}
	}
}MRLDU8(0), MRLDS8(1), MRLDU16(2), MRLDS16(3), MRLD64(5), MRLD128(6);

struct ModifierRuleLDCop: ModifierRule
{
	ModifierRuleLDCop(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111001111111111111111111111", Mask0);
		Bits0 = type << 8;
		if(type==1)
			Name = "CG";
		else if(type==2)
			Name = "CS";
		else if(type == 3)
			Name = "CV";
		else if(type == 5)
		{
			Name = "LU";
			Bits0 = 2<<8;
		}
	}
}MRLDCopCG(1),MRLDCopCS(2),MRLDCopCV(3), MRLDCopLU(5);


struct ModifierRuleSTCop: ModifierRule
{
	ModifierRuleSTCop(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111001111111111111111111111", Mask0);
		Bits0 = type << 8;
		if(type==1)
			Name = "CG";
		else if(type==2)
			Name = "CS";
		else if(type == 3)
			Name = "WT";
	}
}MRSTCopCG(1),MRSTCopCS(2),MRSTCopWT(3);


struct ModifierRuleE: ModifierRule
{
	ModifierRuleE(): ModifierRule("E", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask1);
		Bits1 = 1 << 26;
	}
}MRE;