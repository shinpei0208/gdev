#ifndef RulesDirectiveDefined
#define RulesDirectiveDefined



//Kernel
struct DirectiveRuleKernel;
extern DirectiveRuleKernel DRKernel;

//EndKernel
struct DirectiveRuleEndKernel;
extern DirectiveRuleEndKernel DREndKernel;

struct DirectiveRuleLabel;
extern DirectiveRuleLabel DRLabel;

//Param
struct DirectiveRuleParam;
extern DirectiveRuleParam DRParam;

//Shared
struct DirectiveRuleShared;
extern DirectiveRuleShared DRShared;

//Local
struct DirectiveRuleLocal;
extern DirectiveRuleLocal DRLocal;

//MinStack
struct DirectiveRuleMinStack;
extern DirectiveRuleMinStack DRMinStack;

//MinFrame
struct DirectiveRuleMinFrame;
extern DirectiveRuleMinFrame DRMinFrame;

//Constant2
struct DirectiveRuleConstant2;
extern DirectiveRuleConstant2 DRConstant2;


struct DirectiveRuleConstant;
extern DirectiveRuleConstant DRConstant;


struct DirectiveRuleEndConstant;
extern DirectiveRuleEndConstant DREndConstant;

struct DirectiveRuleRegCount;
extern DirectiveRuleRegCount DRRegCount;

struct DirectiveRuleBarCount;
extern DirectiveRuleBarCount DRBarCount;

struct DirectiveRuleRawInstruction;
extern DirectiveRuleRawInstruction DRRawInstruction;

struct DirectiveRuleArch;
extern DirectiveRuleArch DRArch;

struct DirectiveRuleMachine;
extern DirectiveRuleMachine DRMachine;

struct DirectiveRuleAlign;
extern DirectiveRuleAlign DRAlign;

struct DirectiveRuleSelfDebug;
extern DirectiveRuleSelfDebug DRSelfDebug;


#endif
