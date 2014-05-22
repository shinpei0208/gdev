
#ifndef GlobalVariablesDefined
#define GlobalVariablesDefined


#include <stack>
#include <list>
#include <vector>
#include <iostream>
#include <fstream>
#include "DataTypes.h"
#include "Cubin.h"

using namespace std;



struct Constant2Parser;

extern bool csSelfDebug;

extern int csLineNumber;
extern int csInstructionOffset;
extern Line		csCurrentLine;
extern Instruction csCurrentInstruction;
extern Directive   csCurrentDirective;

extern std::fstream csInput;
extern std::fstream csOutput;
extern char *csSource;
extern int csSourceSize;
extern int csRegCount;
extern int csBarCount;
extern bool csAbsoluteAddressing;

enum OperationMode{Replace, Insert, DirectOutput, Undefined };
extern OperationMode csOperationMode;
extern char* csSourceFilePath;

extern bool csExceptionPrintUsage;
extern bool csErrorPresent;

//the following 3 variables are for Replace mode.
extern int csOutputSectionOffset;
extern int csOutputSectionSize;
extern int csOutputInstructionOffset;

extern std::list<MasterParser*>		csMasterParserList;  //List of parsers to loaded at initialization
extern std::list<LineParser*>		csLineParserList;
extern std::list<InstructionParser*> csInstructionParserList;
extern std::list<DirectiveParser*>	csDirectiveParserList;


extern InstructionRule** csInstructionRules; //sorted array
extern int* csInstructionRuleIndices; //Instruction name index of the corresponding element in csInstructionRules
extern int csInstructionRuleCount;
extern std::list<InstructionRule*>  csInstructionRulePrepList; //used for preperation

extern DirectiveRule** csDirectiveRules; //sorted array
extern int* csDirectiveRuleIndices; //Directive name index of the corresponding element in csDirectiveRules
extern int csDirectiveRuleCount;
extern list<DirectiveRule*>  csDirectiveRulePrepList; //used for preperation

extern stack<MasterParser*>	csMasterParserStack;
extern stack<LineParser*>		csLineParserStack;
extern stack<InstructionParser*>  csInstructionParserStack;
extern stack<DirectiveParser*> csDirectiveParserStack;

extern MasterParser*		csMasterParser;  //curent Master Parser
extern LineParser*			csLineParser;
extern InstructionParser*	csInstructionParser;
extern DirectiveParser*	csDirectiveParser;

extern vector<Line> csLines;
extern vector<Instruction> csInstructions;
extern list<Directive> csDirectives;
extern list<Label> csLabels;
extern list<LabelRequest> csLabelRequests;

//=================

extern ELFSection cubinSectionEmpty, cubinSectionSHStrTab, cubinSectionStrTab, cubinSectionSymTab;
extern ELFSection cubinSectionConstant2, cubinSectionNVInfo;
extern ELFSegmentHeader cubinSegmentHeaderPHTSelf;
extern ELFSegmentHeader cubinSegmentHeaderConstant2;


extern unsigned int cubinCurrentSectionIndex;
extern unsigned int cubinCurrentOffsetFromFirst; //from the end of the end of .symtab
extern unsigned int cubinCurrentSHStrTabOffset;
extern unsigned int cubinCurrentStrTabOffset;
extern unsigned int cubinTotalSectionCount;
extern unsigned int cubinPHTOffset;
extern unsigned int cubinPHCount;
extern unsigned int cubinConstant2Size;
extern unsigned int cubinCurrentConstant2Offset;
extern bool cubinConstant2Overflown;

extern void (*cubinCurrentConstant2Parser)(SubString &content);

enum Architecture{sm_20, sm_21, sm_30};
extern Architecture cubinArchitecture;
extern bool cubin64Bit;



extern char *cubin_str_empty;
extern char *cubin_str_shstrtab;
extern char *cubin_str_strtab;
extern char *cubin_str_symtab;
extern char *cubin_str_extra1;
extern char *cubin_str_extra2;
extern char *cubin_str_text;
extern char *cubin_str_constant0;
extern char *cubin_str_info;
extern char *cubin_str_shared;
extern char *cubin_str_local;
extern char *cubin_str_constant2;
extern char *cubin_str_nvinfo;


extern bool csCurrentKernelOpened;
extern Kernel	csCurrentKernel;
extern list<Kernel> csKernelList;


#else
#endif
