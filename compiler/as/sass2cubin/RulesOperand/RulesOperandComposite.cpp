#include "../DataTypes.h"
#include "../GlobalVariables.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "../RulesOperand.h"
#include "RulesOperandRegister.h"
#include "RulesOperandConstant.h"
#include "RulesOperandComposite.h"


struct OperandRuleMOVStyle: OperandRule
{
	OperandRuleMOVStyle() : OperandRule(MOVStyle){}

	virtual void Process(SubString &component)
	{
		//Register
		if(component.IsRegister() || component.IsRegister())
			((OperandRule*)&OPRRegister2)->Process(component);
		//Constant memory
		else if(component[0]=='c'||component[0]=='C')
		{
			SetConstMem(component);
			MarkConstantMemoryForImmediate32();
		}
		//constant
		else
		{
			unsigned int result;
			//hex
			if(component.Length>2 && component[0]=='0' && (component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate20FromHexConstant(false); //does not allows negative as highest bit cannot be included
			//float
			else if(component[0]=='F')
				result = component.ToImmediate20FromFloatConstant();
			//int
			else
				result = component.ToImmediate20FromIntConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
	}
}OPRMOVStyle; //const fixed


#define mArithmeticCommonEnd {if(negative){component.Start--;	component.Length++;	}}

#define mArithmeticCommon\
	bool negative = false;\
	if(component[0] == '-')\
	{\
		if(!AllowNegative)\
			throw 129;\
		negative = true;\
		csCurrentInstruction.OpcodeWord0 |= 1<<8; \
		component.Start++;\
		component.Length--;\
		if(component.Length<1) \
			throw 116; \
	}\
	/*register*/\
	if(component.IsRegister())\
	{\
		int register2 = component.ToRegister();\
		CheckRegCount(register2);\
		csCurrentInstruction.OpcodeWord0 |= register2 << 26;\
		MarkRegisterForImmediate32();\
	}		\
	/*constant memory*/\
	else if(component.IsConstantMemory()) \
	{\
		SetConstMem(component, 0x1f, true);\
		MarkConstantMemoryForImmediate32();\
	}
//FADD: Register, Constant memory without reg, 20-bit Float)
//issue: will not work if the second operand, a register, has a negative sign before it. This is, however, possible
//since it's been observed in the output of cuobjdump. The mod1:0 bits, when = 10, negate third operand; when = 01, negate second operand;
//when = 11, produces.PO, which I do not understand for now
struct OperandRuleFADDStyle: OperandRule
{
	bool AllowNegative;
	OperandRuleFADDStyle(bool allowNegative) :OperandRule(FADDStyle)
	{
		AllowNegative = allowNegative;
	}
	virtual void Process(SubString &component)
	{
		//Register or constant memory
		mArithmeticCommon
		//constants
		else
		{
			unsigned int result;
			//float
			if(component.IsFloat())
				result = component.ToImmediate20FromFloatConstant();
			//hex
			else if(component.IsHex())
				result = component.ToImmediate20FromHexConstant(true);
			else
				throw 116;
			
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		mArithmeticCommonEnd;
	}
}OPRFADDStyle(true), OPRFMULStyle(false);
//Confirmed: all use specialLast2
//FADDStyle: none
//FMULStyle: FCMP
/*

	FMULAllowNegative: FMUL
	FFMAAllowNegative: FFMA, DMUL, DFMA
	OPRFADDCompositeWithOperator: FADD, FSETP, DADD, DSETP, 
	OPRF2I: F2I, F2F
*/

struct OperandRuleFAllowNegative: OperandRule
{
	bool OnWord0;
	int FlagPos;
	OperandRuleFAllowNegative(bool onWord0, int flagPos): OperandRule(Custom)
	{
		OnWord0 = onWord0;
		FlagPos = flagPos;
	}
	virtual void Process(SubString &component)
	{
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			int flag = 1<<FlagPos;
			if(OnWord0)
				csCurrentInstruction.OpcodeWord0 |= flag;
			else
				csCurrentInstruction.OpcodeWord1 |= flag;
			if(component.Length<2)
				throw 116;//issue: bad error message
		}
		OPRFMULStyle.Process(component);
		mArithmeticCommonEnd
	}
}OPRFMULAllowNegative(false, 25), OPRFFMAAllowNegative(true, 9);


//IADD: Register, Constant memory without reg, 20-bit Int)
struct OperandRuleIADDStyle: OperandRule
{
	bool AllowNegative;
	bool IsI2I;
	OperandRuleIADDStyle(bool allowNegative, bool isI2I) :OperandRule(IADDStyle)
	{
		AllowNegative = allowNegative;
		IsI2I = isI2I;
	}
	virtual void Process(SubString &component)
	{
		bool allowNegate = true;
		// process I2I register
		if (IsI2I && component.IsRegister()) {
			((OperandRule*)&OPRRegister2ForI2I)->Process(component);
			return;
		}
		//register or constant memory
		mArithmeticCommon
		//constants
		else
		{
			unsigned int result;
			//hex
			if(component.IsHex())
				result = component.ToImmediate20FromHexConstant(true);
			//int
			else
				result = component.ToImmediate20FromIntConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}	
		mArithmeticCommonEnd
	}
}OPRIADDStyle(true, false), OPRIMULStyle(false, false),
	OPRI2IStyle(false, true);
//IADD: IADD, ISETP, SHR, SHL
//IMUL: IMUL, IMAD, ICMP, BFE, BFI, SEL
/*
OPRISCADDAllowNegative: ISCADD
OPRI2F: I2F, I2I
*/

struct OperandRuleMAD3: OperandRule
{
	OperandRuleMAD3(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		if(component.IsConstantMemory() )
		{
			csCurrentInstruction.OpcodeWord0 &= 0x03FFFFFF; //remove action of third opr
			OPRIMULStyle.Process(component); //issue: temporary fix only. Not the right one to use
			MarkConstantMemoryForIMAD();
			//issue: need to check third operand is register!!
			//reprocess third as fourth opr
			list<SubString>::iterator opr = csCurrentInstruction.Components.end();
			opr--; //third operand
			opr--; //third operand
			((OperandRule*)&OPRRegister3ForMAD)->Process(*opr);
		}
		else
			((OperandRule*)&OPRRegister3ForMAD)->Process(component);
	}
} OPRMAD3;

struct OperandRuleIAllowNegative: OperandRule
{
	bool OnWord0;
	int FlagPos;
	OperandRuleIAllowNegative(bool onWord0, int flagPos): OperandRule(Custom)
	{
		OnWord0 = onWord0;
		FlagPos = flagPos;
	}
	virtual void Process(SubString &component)
	{
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			int flag = 1<<FlagPos;
			if(OnWord0)
				csCurrentInstruction.OpcodeWord0 |= flag;
			else
				csCurrentInstruction.OpcodeWord1 |= flag;
			if(component.Length<2)
				throw 116;//issue: bad error message
		}
		OPRIMULStyle.Process(component);
		mArithmeticCommonEnd
	}
}OPRISCADDAllowNegative(false, 23);

struct OperandRuleFADDCompositeWithOperator: OperandRule
{
	OperandRuleFADDCompositeWithOperator(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		int startPos = 1;
		if(component[0]=='-')
			csCurrentInstruction.OpcodeWord0 |= 1<<8;
		else if(component[0]=='|')
			csCurrentInstruction.OpcodeWord0 |= 1<<6;
		else startPos = 0;
		SubString s = component.SubStr(startPos, component.Length - startPos);
		OPRFMULStyle.Process(s);
	}
}OPRFADDCompositeWithOperator;


bool LabelProcessing = false;
int LabelAbsoluteAddr = 0;
struct OperandRuleInstructionAddress: OperandRule
{
	OperandRuleInstructionAddress(bool optional): OperandRule(Custom)
	{
		if (optional)
			Type = Optional;
	}
	virtual void Process(SubString &component)
	{
		//Label
		int result = 0;
		if(LabelProcessing)
		{
			result = LabelAbsoluteAddr;
			goto labelProcess;
		}
		if(component[0]=='!')
		{
			LabelRequest request;

			SubString labelName = component.SubStr(1, component.Length - 1);
			labelName.RemoveBlankAtEnd();
			if(labelName.Length==0)
				throw 145; //empty label name
			request.RequestedLabelName = labelName;
			request.RequestIndex = csInstructions.size();
#if 0
			//request.InstructionPointer = csInstructions.end();
			if(csInstructions.size()==0)
				request.Zero = true;
			else
			{
				request.InstructionPointer--;
				request.Zero = false;
			}
#endif
			csLabelRequests.push_back(request);
		}
		//constant memory
		else if(component[0]=='c'||component[0]=='C')
		{
			csCurrentInstruction.OpcodeWord0 |= 1 << 14;
			SetConstMem(component, 0x1f, false);
			//MarkConstantMemoryForImmediate32();
		}
		else
		{
			//cuobjdump's output gives absolute address
			//while actually the address stored in the opcode is relative
			if(csAbsoluteAddressing)
			{
				result = component.ToImmediate32FromHexConstant(false);
labelProcess:
				if((csCurrentInstruction.OpcodeWord1>>26&0x3b)!=0) //no JCAL or JMP
					result = result - (csInstructionOffset);
				if(result<0)
				{
					result = -result;
					result ^= 0xFFFFFF;
					result += 1;
					if(result>0xFFFFFF)
						throw 131;
				}
				else if(result>0x7FFFFF)
					throw 131; //too large offset
				WriteToImmediate32((unsigned int)result);
				LabelProcessing = false;
			}
			//input is relative
			else
			{
				((OperandRule*)&OPRImmediate24HexConstant)->Process(component);
			}
		}
	}
}OPRInstructionAddress(false), OPRInstAddrOptional(true);
//confirmed that all users of OPRInstructionAddress do not use specialLast2. Yet they all support maxBank 31


struct OperandRuleBAR: OperandRule
{
	bool AllowRegister;
	OperandRuleBAR(bool allowRegister=true): OperandRule(Custom)
	{
		AllowRegister= allowRegister;
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		//register
		if(component.IsRegister())
		{
			if(!AllowRegister)
				throw 144; //no registers allowed
			result = component.ToRegister();
			if(result==63&&csBarCount==0)
				csBarCount = 1;
		}
		//numerical expression
		else
		{
			//hex
			if(component.IsHex())
				result = component.ToImmediate32FromHexConstant(false);
			//int
			else
				result = component.ToImmediate32FromInt32();
			if(result>127)
				throw 139;//too large barrier identifier
         if(result>16)
            hpWarning(13);
			csCurrentInstruction.OpcodeWord1|= 1<<15;
			if(result>=csBarCount)
				csBarCount = result+1;
		}
		csCurrentInstruction.OpcodeWord0 |= result << 20;
	}
}OPRBAR, OPRBARNoRegister;

struct OperandRuleTCount: OperandRule
{
	OperandRuleTCount(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		//register
		if(component.IsRegister())
		{
			result = component.ToRegister();
		}
		//numerical expression
		else
		{
			//hex
			if(component.IsHex())
				result = component.ToImmediate32FromHexConstant(false);
			//int
			else
				result = component.ToImmediate32FromInt32();
			if(result>0xfff)
				throw 140;//thread count should be no larger than 4095
			csCurrentInstruction.OpcodeWord1|= 1<<14;
		}
		csCurrentInstruction.OpcodeWord0 &= 0x03ffffff;
		csCurrentInstruction.OpcodeWord0 |= result << 26;		
		csCurrentInstruction.OpcodeWord0 |= result >> 6;
	}
}OPRTCount;

struct OperandRuleCompositeForVADD: OperandRule
{
	//this thing deals with either a register or a 16-bit integer.
	//a negative sign could precede things
	OperandRuleCompositeForVADD(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		//this function will deal with the negative sign. So the negative sign
		//process in OPRRegister2ForVADD is redundant
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			csCurrentInstruction.OpcodeWord0 |= 1<<7; //negative bit is the 8th bit from LSB
		}
		//register
		if(component.IsRegister())
		{
			((OperandRule*)&OPRRegister2ForVADD)->Process(component);
			//default composite mode is immediate, so has to do a mark
			csCurrentInstruction.OpcodeWord1 |= 1 << 15; //register mark
		}
		//16-bit constant
		else
		{
			int result;
			//hex
			if(component.IsHex())
				result = component.ToImmediate32FromHexConstant(false);
			//int
			else
				result = component.ToImmediate32FromInt32(); //issue: will work with negative integers. i.e. "--123" will be accepted, and cause 
										//error 154 below
			if((result & 0xFFFF0000)!=0)
				throw 154; //cannot be longer than 16 bits
			//write result, lowest 6 bits to OpcodeWord0
			csCurrentInstruction.OpcodeWord0 |= result << 26;
			csCurrentInstruction.OpcodeWord1 &= ~(7); //clear the x32 modifier bits
			csCurrentInstruction.OpcodeWord1 |= result >> 6;
			//default is immediate operand, so no need to mark
		}
		if(negative)
		{
			component.Start--;
			component.Length++;
		}
		
	}
};
OperandRuleCompositeForVADD OPRCompositeForVADD;

struct OperandRuleSUEAUStyle: OperandRule
{
	OperandRuleSUEAUStyle() : OperandRule(Custom){}

	virtual void Process(SubString &component)
	{
		//Register
		if(component.IsRegister())
			((OperandRule*)&OPRRegister3ForCMP)->Process(component);
		//Constant memory
		else if(component[0]=='c'||component[0]=='C')
		{
			unsigned int regnr;
			// clera Reg2
			regnr = csCurrentInstruction.OpcodeWord0 >> 26;
			csCurrentInstruction.OpcodeWord0 &= 0x03ffffff;
			// set constant
			SetConstMem(component);
			MarkConstantMemoryForImmediate32();
			// set Reg3
			csCurrentInstruction.OpcodeWord1 &= 0xffff3fff;
			csCurrentInstruction.OpcodeWord1 |=
				(1<<(47-32)) | (regnr<<(49-32));
		}
		else
			throw 116;
	}
}OPRSUEAUStyle;
