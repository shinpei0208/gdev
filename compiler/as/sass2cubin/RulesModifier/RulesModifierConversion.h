#ifndef RulesModifierConversionDefined
#define RulesModifierConversionDefined

struct ModifierRuleF2IDest;
extern ModifierRuleF2IDest
	MRF2IDestU8,
	MRF2IDestU16,
	MRF2IDestU32,
	MRF2IDestU64,
	MRF2IDestS8,
	MRF2IDestS16,
	MRF2IDestS32,
	MRF2IDestS64; 

struct ModifierRuleF2ISource;
extern ModifierRuleF2ISource MRF2ISourceF16,MRF2ISourceF32,MRF2ISourceF64;

struct ModifierRuleF2IRound;
extern ModifierRuleF2IRound MRF2IFLOOR, MRF2ICEIL, MRF2ITRUNC,
		MRF2FFLOOR, MRF2FCEIL, MRF2FTRUNC,
		MRF2FRM, MRF2FRP, MRF2FRZ;

struct ModifierRuleF2IFTZ;
extern ModifierRuleF2IFTZ MRF2IFTZ;

struct ModifierRuleF2FPASS;
extern ModifierRuleF2FPASS MRF2FPASS, MRF2FROUND;

struct ModifierRuleI2FSource;
extern ModifierRuleI2FSource 
	MRI2FSourceU8,
	MRI2FSourceU16,
	MRI2FSourceU32,
	MRI2FSourceU64,
	MRI2FSourceS8,
	MRI2FSourceS16,
	MRI2FSourceS32,
	MRI2FSourceS64;

struct ModifierRuleI2FDest;
extern ModifierRuleI2FDest MRI2FDestF16,MRI2FDestF32,MRI2FDestF64;

struct ModifierRuleI2FRound;
extern ModifierRuleI2FRound MRI2FRM, MRI2FRP, MRI2FRZ;

#else
#endif
