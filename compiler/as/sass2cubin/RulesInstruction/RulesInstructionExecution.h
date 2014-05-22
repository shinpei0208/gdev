#ifndef RulesInstructionExecutionDefined
#define RulesInstructionExecutionDefined

struct InstructionRuleSCHI;
extern InstructionRuleSCHI IRSCHI;

struct InstructionRuleEXIT;
extern InstructionRuleEXIT IREXIT;

struct InstructionRuleCAL;
extern InstructionRuleCAL IRCAL;

struct InstructionRuleJCAL;
extern InstructionRuleJCAL IRJCAL;

struct InstructionRuleSSY;
extern InstructionRuleSSY IRSSY;

struct InstructionRuleBRA;
extern InstructionRuleBRA IRBRA;

struct InstructionRuleJMP;
extern InstructionRuleJMP IRJMP;

struct InstructionRulePRET;
extern InstructionRulePRET IRPRET;

struct InstructionRuleRET;
extern InstructionRuleRET IRRET;

struct InstructionRulePBK;
extern InstructionRulePBK IRPBK;

struct InstructionRuleBRK;
extern InstructionRuleBRK IRBRK;

struct InstructionRulePCNT;
extern InstructionRulePCNT IRPCNT;

struct InstructionRuleCONT;
extern InstructionRuleCONT IRCONT;

struct InstructionRulePLONGJMP;
extern InstructionRulePLONGJMP IRPLONGJMP;

struct InstructionRuleLONGJMP;
extern InstructionRuleLONGJMP IRLONGJMP;

struct InstructionRuleNOP;
extern InstructionRuleNOP IRNOP;

struct InstructionRuleBAR;
extern InstructionRuleBAR IRBAR;

struct InstructionRuleB2R;
extern InstructionRuleB2R IRB2R;

struct InstructionRuleMEMBAR;
extern InstructionRuleMEMBAR IRMEMBAR;

struct InstructionRuleATOM;
extern InstructionRuleATOM IRATOM;

struct InstructionRuleRED;
extern InstructionRuleRED IRRED;

struct InstructionRuleVOTE;
extern InstructionRuleVOTE IRVOTE;

struct InstructionRuleBPT;
extern InstructionRuleBPT IRBPT;

struct InstructionRuleBRX;
extern InstructionRuleBRX IRBRX;

#else
#endif
