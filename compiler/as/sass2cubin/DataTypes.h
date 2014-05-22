#ifndef DataTypesDefined
#define DataTypesDefined



#include "SubString.h"
#include <list>
#include <vector>




//	1
//-----Basic structures used by the assembler: Line, Instruction, Directive

struct Line
{
	SubString LineString;
	int LineNumber;
	Line(){}
	Line(SubString lineString, int lineNumber);
};

struct Instruction
{
	SubString InstructionString;
	int LineNumber;
	std::list<SubString> Modifiers;
	std::list<SubString> Components; //predicate is the first optional component. Then instruction name (without modifier), then operands(unprocesed, may contain modifier)
	bool Is8;	//true: 8-byte opcode. OpcodeWord1 is used as well
	unsigned int OpcodeWord0;
	unsigned int OpcodeWord1;
	int Offset;	//Instruction offset in assembly
	bool Predicated;//indicates whether @Px is present at the beginning
	
	Instruction(){}
	Instruction(SubString instructionString, int offset, int lineNumber);
	void Reset(SubString instructionString, int offset, int lineNumber);
};

struct Directive
{
	SubString DirectiveString;
	int LineNumber;
	std::list<SubString> Parts; //Same as Components in Instruction
	Directive(){}
	Directive(SubString directiveString, int lineNumber);
	void Reset(SubString directiveString, int lineNumber);
};
//-----End of basic types





//	2
//-----Structures for line analysis: ModifierRule, OperandRule, InstructionRule, DirectiveRule
//this enum is largely useless for now
typedef enum OperandType
{
	Register, Immediate32HexConstant, Predicate,
	Immediate32IntConstant, Immediate32FloatConstant, Immediate32AnyConstant, 
	GlobalMemoryWithImmediate32, ConstantMemory, SharedMemoryWithImmediate20, Optional, Custom, 
	MOVStyle, FADDStyle, IADDStyle
};

//Rule for specific modifier
struct ModifierRule
{
	SubString Name;// .RZ would have a name of RZ

	bool Apply0; //apply on OpcodeWord0?
	unsigned int Mask0; // Does an AND operation with opcode first
	unsigned int Bits0; //then an OR operation

	bool Apply1; //Apply on OpcodeWord1?
	unsigned int Mask1;
	unsigned int Bits1;

	bool NeedCustomProcessing;
	virtual void CustomProcess(){}
	ModifierRule(){}
	ModifierRule(char* name, bool apply0, bool apply1, bool needCustomProcessing);
};

//Modifiers are grouped. Modifiers must be present in the correct sequence in which modifier groups are arranged
//Different modifiers from the same group cannot appear at the same time
//For example, FMUL has 2 modifier groups
//1st group: .RP/.RM/.RZ
//2nd group: .SAT
//So if .SAT is to be present, it must be the last modifier
//Only one modifier from the first group could be present
struct ModifierGroup
{
	int ModifierCount; //number of possible modifiers there are in this group
	ModifierRule**  ModifierRules; //point to the modifier rules
	bool Optional; //whether this modifier group is optional
	~ModifierGroup();
	void Initialize(bool optional, int modifierCount, ...);
};

//Rule for processing specific operand
struct OperandRule
{
	OperandType Type;//not really useful. 
	//bool Optional;

	OperandRule();
	OperandRule(OperandType type);
	//the custom processing function used to process a specific operand(component) that must be defined in child classes
	virtual void Process(SubString &component) = 0;
};

//When an instruction rule is initialized, the ComputeIndex needs to be called. 
//They need to be sorted according to their indices and then placed in csInstructionRules
struct InstructionRule
{
	char* Name;
	int OperandCount;
	OperandRule** Operands;

	int ModifierGroupCount;
	ModifierGroup *ModifierGroups;
	

	bool Is8;
	unsigned int OpcodeWord0;
	unsigned int OpcodeWord1;

	//If NeedCustomProcessing is set to true, the components of an instruction SubString
	//will not be processed according to the operand rules. Instead, the CustomProcess function
	//will be called
	bool NeedCustomProcessing;
	virtual void CustomProcess(){}
	int ComputeIndex();
	InstructionRule();
	InstructionRule(char* name, int modifierGroupCount, bool is8, bool needCustomProcessing);
	void SetOperands(int operandCount, ...);
	~InstructionRule();
};

struct DirectiveRule
{
	char* Name;
	virtual void Process() = 0;
	int ComputeIndex();
};
//-----End of structures for line analysis






//	3
//-----Abstract parser structures: Parser, MasterParser, LineParser, InstructionParser, DirectiveParser
struct Parser
{
	char* Name;
};
struct MasterParser: Parser
{
	virtual void Parse(unsigned int startinglinenumber) = 0;
};
struct LineParser: Parser
{
	virtual void Parse(Line &line) = 0;
};
struct InstructionParser: Parser
{
	virtual void Parse() = 0;
};
struct DirectiveParser: Parser
{
	virtual void Parse() = 0;
};
//-----End of abstract parser structures




//	9.0
//-----Label structures
struct Label
{
	SubString Name;
	int LineNumber;
	unsigned int Offset;
};
struct LabelRequest
{
	// Index of the requesting instruction in kernel
	unsigned RequestIndex;
	//std::vector<Instruction>::iterator InstructionPointer;
	SubString RequestedLabelName;
	//bool Zero;
};
//-----End of Label structures
#else
#endif
