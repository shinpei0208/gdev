
#ifndef RulesModifierOthersDefined
#define RulesModifierOthersDefined

struct ModiferRuleCCTLOp1;
extern ModiferRuleCCTLOp1 MRCCTLOp1U,
						  MRCCTLOp1C,
						  MRCCTLOp1I;

struct ModifierRuleCCTLOp2;
extern ModifierRuleCCTLOp2 MRCCTLOp2QRY1,
						   MRCCTLOp2PF1,
						   MRCCTLOp2PF1_5,
						   MRCCTLOp2PR2,
						   MRCCTLOp2WB,
						   MRCCTLOp2IV,
						   MRCCTLOp2IVALL,
						   MRCCTLOp2RS;


struct ModifierRulePSETPMainop;
extern ModifierRulePSETPMainop MRPSETPAND, MRPSETPOR, MRPSETPXOR;

struct ModifierRuleTEX1;
extern ModifierRuleTEX1 MRTEXI, MRTEXLZ, MRTEXLB, MRTEXLL,
	MRTEXLBA, MRTEXLLA, MRTEXAOFFI, MRTEXDC, MRTEXNDV;
struct ModifierRuleTEX0;
extern ModifierRuleTEX0 MRTEXNODEP, MRTEXP, MRTEXT;

struct ModifierRuleSUST1;
extern ModifierRuleSUST1 MRSUST2D, MRSUSTB, MRSUSTNEAR, MRSUSTTRAP;
struct ModifierRuleSUST0;
extern ModifierRuleSUST0 MRSUSTWB, MRSUSTCS, MRSUSTWT,
	MRSUSTU8, MRSUSTS8, MRSUSTU16, MRSUSTS16, MRSUST64, MRSUST128;

struct ModifierRuleTLD;
extern ModifierRuleTLD MRTLDLZ, MRTLDLL, MRTLDMS, MRTLDCL;

struct ModifierRuleSUBFM;
extern ModifierRuleSUBFM MRSUBFM3D;

struct ModifierRuleSUCLAMP;
extern ModifierRuleSUCLAMP MRSUCLAMPSD, MRSUCLAMPPL, MRSUCLAMPBL;
extern ModifierRuleSUCLAMP MRSUCLAMPR1, MRSUCLAMPR2, MRSUCLAMPR4,
	MRSUCLAMPR8, MRSUCLAMPR16;

#endif
