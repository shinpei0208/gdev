#include "DataTypes.h"
#include "GlobalVariables.h"


#ifndef RulesOperandDefined
#define RulesOperandDefined

inline void WriteToImmediate32(unsigned int content)
{
	csCurrentInstruction.OpcodeWord0 |= content <<26;
	csCurrentInstruction.OpcodeWord1 |= content >> 6;
}
inline void MarkConstantMemoryForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 1<<14; //constant memory flag
}
inline void MarkConstantMemoryForIMAD()
{
	csCurrentInstruction.OpcodeWord1 &= (0xFFFF3FFF);
	csCurrentInstruction.OpcodeWord1 |= 2<<14; //special constant mem flag for IMAD, FFMA, DFMA
}
inline void MarkImmediate20ForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 3<<14; //20-bit immediate flag
}
inline void MarkRegisterForImmediate32()
{
}

inline void CheckRegCount(int reg)
{
	if(reg!=63&&reg>=csRegCount)
		csRegCount=reg+1;
}

inline void SetConstMem(SubString &component, int maxBank=0xf, bool specialLast2=false)
{	
	unsigned int bank, memory;
	int register1;
	component.ToConstantMemory(bank, register1, memory, maxBank); //correct
	if(register1 != 63)
		throw 112;  //register cannot be used in composite constant memory operand
	if(specialLast2)
	{
		if(memory%2!=0)
			throw 148; // should be multiples of 4, multiples of 2 allowed for experiment
		if(memory%4!=0)
			::hpWarning(14); //should be multiples of 4
		if(bank>0xf)
		{
			memory |= 1;
			bank -= 0x10;
		}
	}
	csCurrentInstruction.OpcodeWord1 |= bank<<10;
	WriteToImmediate32(memory);
	//MarkConstantMemoryForImmediate32();
}

#include "RulesOperand/RulesOperandConstant.h"
#include "RulesOperand/RulesOperandRegister.h"
#include "RulesOperand/RulesOperandMemory.h"
#include "RulesOperand/RulesOperandComposite.h"
#include "RulesOperand/RulesOperandOthers.h"

#endif
