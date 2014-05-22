#ifndef RulesModifierExecutionDefined
#define RulesModifierExecutionDefined

struct ModifierRuleCALNOINC;
extern ModifierRuleCALNOINC MRCALNOINC;

struct ModifierRuleBRAU;
extern ModifierRuleBRAU MRBRAU, MRBRALMT;

struct ModifierRuleNOPTRIG;
extern ModifierRuleNOPTRIG MRNOPTRIG;

struct ModifierRuleNOPOP;
extern ModifierRuleNOPOP MRNOPFMA64,
						 MRNOPFMA32,
						 MRNOPXLU  ,
						 MRNOPALU  ,
						 MRNOPAGU  ,
						 MRNOPSU   ,
						 MRNOPFU   ,
						 MRNOPFMUL ;

struct ModifierRuleMEMBAR;
extern ModifierRuleMEMBAR MRMEMBARCTA, MRMEMBARGL, MRMEMBARSYS;

struct ModifierRuleATOM;
extern ModifierRuleATOM
						MRATOMADD,	
						MRATOMMIN,
						MRATOMMAX,
						MRATOMDEC,
						MRATOMINC,
						MRATOMAND,
						MRATOMOR,
						MRATOMXOR,
						MRATOMEXCH,
						MRATOMCAS;


struct ModifierRuleATOMType;
extern ModifierRuleATOMType 
						MRATOMTypeU64,
						MRATOMTypeS32,
						MRATOMTypeF32;

struct ModifierRuleATOMIgnored;
extern ModifierRuleATOMIgnored MRATOMIgnoredFTZ, MRATOMIgnoredRN;

struct ModifierRuleVOTE;
extern ModifierRuleVOTE MRVOTEALL, MRVOTEANY, MRVOTEEQ, MRVOTEVTG;

struct ModifierRuleVOTEVTG;
extern ModifierRuleVOTEVTG MRVOTEVTGR, MRVOTEVTGA, MRVOTEVTGRA;

struct ModifierRuleBPT;
extern ModifierRuleBPT MRBPTDRAIN, MRBPTCAL, MRBPTPAUSE, MRBPTTRAP;

#endif
