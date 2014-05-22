
#include "DataTypes.h"
#include "SubString.h"
#include <stdarg.h>

#include "stdafx.h" //Mark



//-----Extern declarations
extern char* csSource; //The array with the entire source file stored in it
extern int csRegCount;   //highest register used
extern int hpParseComputeInstructionNameIndex(SubString &name);// compute instruction index from instruction name
extern int hpParseComputeDirectiveNameIndex(SubString &name);
//-----End of extern declaration

//	1
//-----Basic structures used by the assembler: Line, Instruction, Directive

//---Line
Line::Line(SubString lineString, int lineNumber)
{
	LineString = lineString;
	LineNumber = lineNumber;
}


//---Instruction
Instruction::Instruction(SubString instructionString, int offset, int lineNumber)
{
	InstructionString = instructionString;
	Offset = offset;
	LineNumber = lineNumber;
}
void Instruction::Reset(SubString instructionString, int offset, int lineNumber)
{
	InstructionString = instructionString;
	Offset = offset;
	LineNumber = lineNumber;
	Components.clear();
	Is8 = true;
	OpcodeWord0 = 0;
	OpcodeWord1 = 0;
	Predicated = false;
}


//---Directive
Directive::Directive(SubString directiveString, int lineNumber)
{
	DirectiveString = directiveString;
	LineNumber = lineNumber;
}
void Directive::Reset(SubString directiveString, int lineNumber)
{
	DirectiveString = directiveString;
	LineNumber = lineNumber;
	Parts.clear();
}

//-----End of basic types





//	2
//-----Structures for line analysis: ModifierRule, OperandRule, InstructionRule, DirectiveRule
//this enum is largely useless for now

//ModifierRule
ModifierRule::ModifierRule(char* name, bool apply0, bool apply1, bool needCustomProcessing)
{
	Name = name;
	Apply0 = apply0;
	Apply1 = apply1;
	NeedCustomProcessing = needCustomProcessing;
}


//ModifierGroup
//Modifiers are grouped. Modifiers must be present in the correct sequence in which modifier groups are arranged
//Different modifiers from the same group cannot appear at the same time
//For example, FMUL has 2 modifier groups
//1st group: .RP/.RM/.RZ
//2nd group: .SAT
//So if .SAT is to be present, it must be the last modifier
//Only one modifier from the first group could be present
ModifierGroup::~ModifierGroup()
{
	if(ModifierCount)
		delete[] ModifierRules;
}
void ModifierGroup::Initialize(bool optional, int modifierCount, ...)
{
	Optional = optional;
	va_list modifierRules;
	va_start (modifierRules, modifierCount);
	ModifierCount = modifierCount;
	ModifierRules = new ModifierRule*[modifierCount];
	for(int i =0; i<modifierCount; i++)
		ModifierRules[i] = va_arg(modifierRules, ModifierRule*);
	va_end(modifierRules);

}

//OperandRule
OperandRule::OperandRule(OperandType type)
{
	Type = type;
}


//InstructionRule
//When an instruction rule is initialized, the ComputeIndex needs to be called. 
//They need to be sorted according to their indices and then placed in csInstructionRules
int InstructionRule::ComputeIndex()
{
	int result = 0;
	int len = strlen(Name);
	SubString nameString;
	nameString.Start = Name;
	nameString.Length = len;
	result = hpParseComputeInstructionNameIndex(nameString);
	return result;
}
	
InstructionRule::InstructionRule(char* name, int modifierGroupCount, bool is8, bool needCustomProcessing)
{
	Name = name;
	OperandCount = 0;
	ModifierGroupCount = modifierGroupCount;
	if(modifierGroupCount>0)
		ModifierGroups = new ModifierGroup[modifierGroupCount];
	Is8 = is8;
	NeedCustomProcessing = needCustomProcessing;
}
void InstructionRule::SetOperands(int operandCount, ...)
{
	OperandCount = operandCount;
	Operands = new OperandRule*[operandCount];
	va_list operandRules;
	va_start (operandRules, operandCount);
	for(int i =0; i<operandCount; i++)
		Operands[i] = va_arg(operandRules, OperandRule*);
	va_end(operandRules);

}
InstructionRule::~InstructionRule()
{
	if(OperandCount)
		delete[] Operands;
	if(ModifierGroupCount)
		delete[] ModifierGroups;		
}


//DirectiveRule
int DirectiveRule::ComputeIndex()
{
	int result = 0;
	int len = strlen(Name);
	SubString nameString;
	nameString.Start = Name;
	nameString.Length = len;
	result = hpParseComputeDirectiveNameIndex(nameString);
	return result;
}
//-----End of structures for line analysis


