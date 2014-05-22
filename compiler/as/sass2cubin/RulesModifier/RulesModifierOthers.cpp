
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesModifierOthers.h"


struct ModiferRuleCCTLOp1: ModifierRule
{
	ModiferRuleCCTLOp1(int type): ModifierRule("", true , false, false)
	{
		hpBinaryStringToOpcode4("1111 111111 1111 111111 111111 00 1111", Mask0);
		Bits0 = type<<26;
		if(type==1)
			Name = "U";
		else if(type == 2)
			Name = "C";
		else if(type == 3)
			Name = "I";
		else
			throw exception();
	}
}	MRCCTLOp1U(1),
	MRCCTLOp1C(2),
	MRCCTLOp1I(3);

struct ModifierRuleCCTLOp2: ModifierRule
{
	ModifierRuleCCTLOp2(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 100011 1111 111111 111111 111111", Mask0);
		Bits0 = type<<5;
		if(type==0)
			Name = "QRY1";
		else if(type==1)
			Name = "PF1";
		else if(type==2)
			Name = "PF1_5";
		else if(type==3)
			Name = "PR2";
		else if(type==4)
			Name = "WB";
		else if(type==5)
			Name = "IV";
		else if(type==6)
			Name = "IVALL";
		else if(type==7)
			Name = "RS";
		else
			throw exception();
	}
}	MRCCTLOp2QRY1(0),
	MRCCTLOp2PF1 (1) ,
	MRCCTLOp2PF1_5(2),
	MRCCTLOp2PR2(3),
	MRCCTLOp2WB(4),
	MRCCTLOp2IV(5),
	MRCCTLOp2IVALL(6),
	MRCCTLOp2RS(7);

struct ModifierRulePSETPMainop: ModifierRule
{
	ModifierRulePSETPMainop(int type): ModifierRule("", true, false, false)
	{
		Mask0 = 0x3fffffff;
		Bits0 = type << 30;
		if(type==0)
			Name = "AND";
		else if(type==1)
			Name = "OR";
		else if(type == 2)
			Name= "XOR";
		else throw exception();
	}
}MRPSETPAND(0), MRPSETPOR(1), MRPSETPXOR(2);

struct ModifierRuleTEX1: ModifierRule
{
	ModifierRuleTEX1(int type): ModifierRule("", false, true, false)
	{
		switch(type) {
		case 0: // I:bit50
			Mask1 = 0xfffbffff;
			Bits1 = 1 << (50-32);
			Name = "I";
			break;
		case 1: // LZ:bit59-57
			Mask1 = 0xf1ffffff;
			Bits1 = 1 << (57-32);
			Name = "LZ";
			break;
		case 2: // LB:bit59-57
			Mask1 = 0xf1ffffff;
			Bits1 = 2 << (57-32);
			Name = "LB";
			break;
		case 3: // LL:bit59-57
			Mask1 = 0xf1ffffff;
			Bits1 = 3 << (57-32);
			Name = "LL";
			break;
		case 4: // LBA:bit59-57
			Mask1 = 0xf1ffffff;
			Bits1 = 6 << (57-32);
			Name = "LBA";
			break;
		case 5: // LLA:bit59-57
			Mask1 = 0xf1ffffff;
			Bits1 = 7 << (57-32);
			Name = "LLA";
			break;
		case 6: // AOFFI:bit54
			Mask1 = 0xffbfffff;
			Bits1 = 1 << (54-32);
			Name = "AOFFI";
			break;
		case 7: // DC:bit56
			Mask1 = 0xfeffffff;
			Bits1 = 1 << (56-32);
			Name = "DC";
			break;
		case 8: // NDV:bit45
			Mask1 = 0xffffdfff;
			Bits1 = 1 << (45-32);
			Name = "NDV";
			break;
		default:
			throw exception();
		}
	}
} MRTEXI(0), MRTEXLZ(1), MRTEXLB(2), MRTEXLL(3), MRTEXLBA(4),
  MRTEXLLA(5), MRTEXAOFFI(6), MRTEXDC(7), MRTEXNDV(8);

struct ModifierRuleTEX0: ModifierRule
{
	ModifierRuleTEX0(int type): ModifierRule("", true, false, false)
	{
		switch(type) {
		case 0: // NODEP:bit9
			Mask0 = 0xfffffdff;
			Bits0 = 1 << 9;
			Name = "NODEP";
			break;
		case 1: // P:bit8-7
			Mask0 = 0xfffffe7f;
			Bits0 = 2 << 7;
			Name = "P";
			break;
		case 2: // T:bit8-7
			Mask0 = 0xfffffe7f;
			Bits0 = 1 << 7;
			Name = "T";
			break;
		default:
			throw exception();
		}
	}
} MRTEXNODEP(0), MRTEXP(1), MRTEXT(2);

struct ModifierRuleSUST1: ModifierRule
{
	ModifierRuleSUST1(int type): ModifierRule("", false, true, false)
	{
		switch(type) {
		case 0: // 2D:maybe bit46-44=101
			Mask1 = 0xffff8fff;
			Bits1 = 5 << (44-32);
			Name = "2D";
			break;
		case 1: // B:unknown
			Mask1 = 0xffffffff;
			Bits1 = 0;
			Name = "B";
			break;
		case 2: // NEAR
			Mask1 = 0xfffe7fff;
			Bits1 = 1 << (47-32);
			Name = "NEAR";
			break;
		case 3: // TRAP
			Mask1 = 0xfffe7fff;
			Bits1 = 2 << (47-32);
			Name = "TRAP";
			break;
		default:
			throw exception();
		}
	}
} MRSUST2D(0), MRSUSTB(1), MRSUSTNEAR(2), MRSUSTTRAP(3);

struct ModifierRuleSUST0: ModifierRule
{
	ModifierRuleSUST0(int type): ModifierRule("", true, false, false)
	{
		switch(type) {
		case 0: // WB
			Mask0 = 0xfffffcff;
			Bits0 = 0 << 8;
			Name = "WB";
			break;
		case 1: // CS
			Mask0 = 0xfffffcff;
			Bits0 = 2 << 8;
			Name = "CS";
			break;
		case 2: // WT
			Mask0 = 0xfffffcff;
			Bits0 = 3 << 8;
			Name = "WT";
			break;
		case 3: // U8
			Mask0 = 0xffffff1f;
			Bits0 = 0 << 5;
			Name = "U8";
			break;
		case 4: // S8
			Mask0 = 0xffffff1f;
			Bits0 = 1 << 5;
			Name = "S8";
			break;
		case 5: // U16
			Mask0 = 0xffffff1f;
			Bits0 = 2 << 5;
			Name = "U16";
			break;
		case 6: // S16
			Mask0 = 0xffffff1f;
			Bits0 = 3 << 5;
			Name = "S16";
			break;
		case 7: // 64
			Mask0 = 0xffffff1f;
			Bits0 = 5 << 5;
			Name = "64";
			break;
		case 8: // 128
			Mask0 = 0xffffff1f;
			Bits0 = 6 << 5;
			Name = "128";
			break;
		default:
			throw exception();
		}
	}
} MRSUSTWB(0), MRSUSTCS(1), MRSUSTWT(2),
  MRSUSTU8(3), MRSUSTS8(4), MRSUSTU16(5), MRSUSTS16(6),
  MRSUST64(7), MRSUST128(8);

struct ModifierRuleTLD: ModifierRule
{
	ModifierRuleTLD(int type): ModifierRule("", false, true, false)
	{
		switch(type) {
		case 0: // LZ:bit57=0
			Mask1 = 0xfdffffff;
			Bits1 = 0 << (57-32);
			Name = "LZ";
			break;
		case 1: // LL:bit57
			Mask1 = 0xfdffffff;
			Bits1 = 1 << (57-32);
			Name = "LL";
			break;
		case 2: // MS:bit55
			Mask1 = 0xff7fffff;
			Bits1 = 1 << (55-32);
			Name = "MS";
			break;
		case 3: // CL:bit56
			Mask1 = 0xfeffffff;
			Bits1 = 1 << (56-32);
			Name = "CL";
			break;
		default:
			throw exception();
		}
	}
} MRTLDLZ(0), MRTLDLL(1), MRTLDMS(2), MRTLDCL(3);

struct ModifierRuleSUBFM: ModifierRule
{
	ModifierRuleSUBFM(): ModifierRule("3D", false, true, false)
	{
		Mask1 = 0xfffeffff;
		Bits1 = 1<<(48-32);
	}
} MRSUBFM3D;

struct ModifierRuleSUCLAMP: ModifierRule
{
	// NOTICE: sets dummy values, see OPRSUCLAMPImm
	ModifierRuleSUCLAMP(int type): ModifierRule("", true, false, false)
	{
		if (type < 3)
			Mask0 = 0xfffffe7f;
		else
			Mask0 = 0xffffff8f; // use bit4!
		switch(type) {
		case 0: // SD 0,2,4,6,8
			Bits0 = 0<<7;
			Name = "SD";
			break;
		case 1: // PL a,c,e,10,12
			Bits0 = 1<<7;
			Name = "PL";
			break;
		case 2: // BL 14,16,18,1a,1c
			Bits0 = 2<<7;
			Name = "BL";
			break;
		case 3:
			Bits0 = 0<<4;
			Name = "R1";
			break;
		case 4:
			Bits0 = 1<<4;
			Name = "R2";
			break;
		case 5:
			Bits0 = 2<<4;
			Name = "R4";
			break;
		case 6:
			Bits0 = 3<<4;
			Name = "R8";
			break;
		case 7:
			Bits0 = 4<<4;
			Name = "R16";
			break;
		default:
			throw exception();
		}
	}
} MRSUCLAMPSD(0), MRSUCLAMPPL(1), MRSUCLAMPBL(2),
	MRSUCLAMPR1(3), MRSUCLAMPR2(4), MRSUCLAMPR4(5),
	MRSUCLAMPR8(6), MRSUCLAMPR16(7);
