#include "../DataTypes.h"
#include "../GlobalVariables.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesOperandRegister.h"
#include "../RulesOperand.h"



struct OperandRuleRegister: OperandRule
{
	int Offset;//offset of register bits in OpcodeWord0
	//14 for reg0, 20 for reg1, 26 for reg2. reg3 is not being dealt with here
	
	//this constructor is not really so useful. However, Optional operand can be indicated
	//here with a type Optional instead of Register
	OperandRuleRegister(int offset): OperandRule(Register)
	{
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		//parse the expression using the parsing function defined under SubString
		int result = component.ToRegister();
		//Check if this register is the highest register used so far
		//issue: .128 and .64 will cause the highest register used be higher than the register indicated in the expression
		CheckRegCount(result);
		//apply result to OpcodeWord0
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord0 |= result;
	}
}	OPRRegister0(14), 
	OPRRegister1(20), 
	OPRRegister2(26);

//reg3 used a separate rule because it applies it result to OpcodeWord1 instead of 0
struct OperandRuleRegister3: OperandRule
{
	bool AllowNegative;
	int Offset;
	OperandRuleRegister3(bool allowNegative, int offset, bool optional):OperandRule(Register)
	{
		if(optional)Type = Optional;
		AllowNegative = allowNegative;
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		//parse
		bool negative = false;
		if(component[0]=='-')
		{
			if(!AllowNegative)
				throw 134; //negative now allowed here
			negative = true;
			component.Start++;
			component.Length--;
			csCurrentInstruction.OpcodeWord0|= 1<<8;
		}
		int result = component.ToRegister();
		CheckRegCount(result);
		//apply result
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord1 &= ~(63<<Offset);
		csCurrentInstruction.OpcodeWord1 |=result;
		if(negative)
		{
			component.Start--;
			component.Length++;
		}
	}
}	OPRRegister3ForMAD(true, 17, false), 
	OPRRegister3ForCMP(false, 17, false),
	OPRRegister3ForATOM(false, 11, false),
	OPRRegister4ForATOM(false, 17, true);

//Note that some operands can have modifiers
//This rule deals with registers that can have the .CC modifier
struct OperandRuleRegisterWithCC: OperandRule
{
	int Offset, FlagPos;
	OperandRuleRegisterWithCC(int offset, int flagPos): OperandRule(Register)
	{
		Offset = offset;
		FlagPos = flagPos; //FlagPos is the position of the bit in OpcodeWord1 to be set to 1
	}
	virtual void Process(SubString &component)
	{
		//parse the register expression
		int result = component.ToRegister();
		CheckRegCount(result);
		//apply result
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord0 |= result;
		//look for .CC
		int dotPos = component.Find('.', 0);
		if(dotPos!=-1)
		{
			SubString mod = component.SubStr(dotPos, component.Length - dotPos);
			if(mod.Length>=3 && mod[1] == 'C' && mod[2] == 'C')
			{
				csCurrentInstruction.OpcodeWord1 |= 1<<FlagPos;
			}
		}
	}
}OPRRegisterWithCC4IADD32I(14, 26), OPRRegisterWithCCAt16(14, 16);//for reg0




inline void SetMaxRegisterFor64And128(int reg)
{
	if(reg==63)
		return;

#ifdef DebugMode
	if(reg>63||reg<0)
		throw;
#endif

	unsigned int lengthType = (csCurrentInstruction.OpcodeWord0&0x000000ff);
	lengthType>>=5;
	if(lengthType>4)
	{
		if(lengthType==5) //64
		{
			reg++;
		}
		else if(lengthType==6) //128
		{
			reg+=3;
		}
		else throw; //error in assembler
		if(reg>=63)
			throw 147; //reg too large
	}
	if(reg>=csRegCount)
		csRegCount = reg+1;
}
struct OperandRuleRegister0ForMemory: OperandRule
{
	OperandRuleRegister0ForMemory(): OperandRule(Register){}
	virtual void Process(SubString &component)
	{
		int reg = component.ToRegister();
		SetMaxRegisterFor64And128(reg); //check .64 and .128
		reg = reg<<14; //reg0
		csCurrentInstruction.OpcodeWord0 |= reg;

	}
}OPRRegister0ForMemory;


//Predicate register operand
struct OperandRulePredicate: OperandRule
{
	int Offset; //offset of the predicate's bitfield
	bool Word0; //whether it applies to OpcodeWord0 or Word1
	OperandRulePredicate(int offset, bool word0, bool optional): OperandRule(Predicate)
	{
		//some predicate operands can be optional
		if(optional)
			Type = Optional; //issue: doesn't work for predicate1 as operands in the middle cannot be optional
		Word0 = word0;
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		//No parsing function in SubString is available to process predicate expression
		//So the parsing is done here
		if(component.Length<2 || (component[0] != 'p' && component[0] != 'P'))
			throw 126; //incorrect predicate
		//pt
		if(component[1]=='t' || component[1] == 'T')
			result = 7;
		//Px
		else
		{
			result = component[1] - 48;
			if(result<0 || result > 7)
				throw 126;
		}
		result <<= Offset;
		//clear the bit field
		//apply result
		if(Word0)
		{
			csCurrentInstruction.OpcodeWord0 &= ~(7<<Offset);
			csCurrentInstruction.OpcodeWord0 |= result;
		}
		else
		{
			csCurrentInstruction.OpcodeWord1 &= ~(7<<Offset);
			csCurrentInstruction.OpcodeWord1 |= result;
		}
	}
}	OPRPredicate1(14, true, true), 
	OPRPredicate0(17, true, false), 
	OPRPredicate2NotNegatable(17, false, true),
	OPRPredicateForLDSLK(18, false, false),
	OPRPredicateForBAR(21, false, false),
	OPRPredicate0ForVOTE(22, false, false),
	OPRPredicate1ForVOTENotNegatable(20, true, true),
	OPRPredicate3ForPSETPNotNegatable(26, true, false), //internal
	OPRPredicate0ForSTSCUL(8, true, false),
	OPRPredicateForSUCLAMP(23, false, false);

//Some predicate registers expressions can be negated with !
//this kind of operand is processed separately
struct OperandRulePredicate2: OperandRule
{
	OperandRule* PredRule;
	int NegateOffset;
	bool OnWord0;
	OperandRulePredicate2(OperandRule* predRule, int negateOffset, bool onWord0, bool optional=true): OperandRule(Optional)
	{
		if(!optional)
			Type = Predicate;
		OnWord0 = onWord0;
		NegateOffset = negateOffset;
		PredRule = predRule;
	}
	virtual void Process(SubString &component)
	{
		int startPos = 0;
		if(component[0]=='!')
		{
			startPos = 1;
			if(OnWord0)
				csCurrentInstruction.OpcodeWord0 |= 1<<NegateOffset;
			else
				csCurrentInstruction.OpcodeWord1 |= 1<<NegateOffset;
			component.Start++;
			component.Length--;
		}
		PredRule->Process(component);
		if(startPos)
		{
			component.Start--;
			component.Length++;
		}
	}
}	OPRPredicate2(&OPRPredicate2NotNegatable, 20, false),
	OPRPredicate1ForVOTE(&OPRPredicate1ForVOTENotNegatable, 23, true),
	OPRPredicate3ForPSETP(&OPRPredicate3ForPSETPNotNegatable, 29, true, false);

struct OperandRulePredicateForLDLK: OperandRule
{
	OperandRulePredicateForLDLK(): OperandRule(Predicate)
	{
	}
	virtual void Process(SubString &component)
	{		
		unsigned int result;
		if(component.Length<2 || (component[0] != 'p' && component[0] != 'P'))
			throw 126; //incorrect predicate
		if(component[1]=='t' || component[1] == 'T')
			result = 7;
		else
		{
			result = component[1] - 48;
			if(result<0 || result > 7)
				throw 126;
		}
		//p is split into p_0 and p_1
		csCurrentInstruction.OpcodeWord0 |= (result&0xfffffffb)<<8;
		csCurrentInstruction.OpcodeWord1 |= (result&0xfffffffc)<<24;
	}
}OPRPredicateForLDLK;


struct OperandRuleFADD32IReg1: OperandRule
{
	OperandRuleFADD32IReg1(): OperandRule(Register){}
	virtual void Process(SubString &component)
	{
		int startPos = 1;
		if(component[0]=='-')
			csCurrentInstruction.OpcodeWord0 |= 1<<9;
		else if(component[0]=='|')
			csCurrentInstruction.OpcodeWord0 |= 1<<7;
		else startPos = 0;
		//leave the operator out when processing it using a general-purpose operand rule
		SubString s = component.SubStr(startPos, component.Length - startPos);
		OPRRegister1.Process(s);
	}
}OPRFADD32IReg1;

struct OperandRuleRegister1WithSignFlag : OperandRule
{
	int FlagPos;
	bool OnWord1;
	OperandRuleRegister1WithSignFlag(int flagPos, bool onWord1): OperandRule(Register)
	{
		FlagPos = flagPos;
		OnWord1 = onWord1;
	}
	virtual void Process(SubString &component)
	{
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			unsigned int result = 1<<FlagPos;
			if(OnWord1)
				csCurrentInstruction.OpcodeWord1 |= result;
			else
				csCurrentInstruction.OpcodeWord0 |= result;
			//no need to check length. checked in substring function
		}
		OPRRegister1.Process(component);
		if(negative)
		{
			component.Start--;
			component.Length++;
		}
	}
}OPRIMADReg1(9, false), OPRISCADDReg1(24, true);

inline void RegCheckForDouble(int reg)
{
#ifdef DebugMode
	if(reg>63||reg<0)
		throw;
#endif
	if(reg!=63)
	{
		if(reg==62)
			throw 151; //should be less than 62
		reg++;
		if(reg>=csRegCount)
			csRegCount=reg+1;
	}
}

struct OperandRuleRegister0ForDouble: OperandRule
{
	OperandRuleRegister0ForDouble(): OperandRule(Register){}
	virtual void Process(SubString &component)
	{
		int result = component.ToRegister();
		RegCheckForDouble(result);
		csCurrentInstruction.OpcodeWord0 |= result<<14;
	}
}OPRRegister0ForDouble;

struct OperandRuleRegister1ForDouble: OperandRule
{
	OperandRule* TargetRule;
	OperandRuleRegister1ForDouble(OperandRule* targetRule): OperandRule(Register)
	{
		TargetRule = targetRule;
	}
	virtual void Process(SubString &component)
	{
		TargetRule->Process(component);
		SubString localComp = component;
		if(component[0]=='-'||component[0]=='|')
		{
			localComp = component.SubStr(1, component.Length -1);
			localComp.RemoveBlankAtBeginning();
			if(localComp.Length==0)
				throw 125; //empty operand
		}
		int result = localComp.ToRegister();
		RegCheckForDouble(result);
	}
}OPRRegister1ForDoubleWith2OP(&OPRFADD32IReg1), OPRRegister1ForDouble(&OPRRegister1);

struct OperandRuleCompositeOperandForDouble: OperandRule
{
	OperandRule* TargetRule;
	OperandRuleCompositeOperandForDouble(OperandRule* targetRule): OperandRule(Custom)
	{
		TargetRule = targetRule;
	}
	virtual void Process(SubString &component)
	{
		TargetRule->Process(component);
		SubString localComp = component;
		if(component[0]=='-'||component[0]=='|')
		{
			localComp = component.SubStr(1, component.Length -1);
			localComp.RemoveBlankAtBeginning();
			if(localComp.Length==0)
				throw 125; //empty operand
		}
		if(localComp.IsRegister())
		{
			int result = localComp.ToRegister();
			RegCheckForDouble(result);
		}
	}
}OPRCompositeForDoubleWith2OP((OperandRule*)&OPRFADDCompositeWithOperator), OPRCompositeForDoubleWith1OP((OperandRule*)&OPRFFMAAllowNegative);

struct OperandRuleRegister3ForDouble: OperandRule
{
	OperandRuleRegister3ForDouble(): OperandRule(Register){}
	virtual void Process(SubString &component)
	{
		OPRRegister3ForMAD.Process(component);
		int result = component.ToRegister();
		RegCheckForDouble(result);
	}
}OPRRegister3ForDouble;

struct OperandRuleRegisterForVADD: OperandRule
{
	//this operand supports both the negative sign and the .rxsel subword selection modifier
	int RegOffset; //both reside in OpcodeWord0
	int SelOffset; //both reside in OpcodeWord1
	int NegOffset; //both reside in OpcodeWord0
	OperandRuleRegisterForVADD(int regOffset, int selOffset, int negOffset): OperandRule(Register)
	{
		RegOffset = regOffset;
		SelOffset = selOffset;
		NegOffset = negOffset;
	}
	virtual void Process(SubString &component)
	{
		//check negative sign first
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			csCurrentInstruction.OpcodeWord0 |= 1<<NegOffset;
		}
		//write to register field
		int result = component.ToRegister();
		CheckRegCount(result);
		csCurrentInstruction.OpcodeWord0 |= result << RegOffset;
		//check sub-word selection modifier
		int selector = 0;
		int dotPos = component.Find('.', 0);
		if(dotPos!=-1)
		{
			//get the substring from the dot to the last non-blank character
			SubString mod = component.SubStr(dotPos, component.Length - dotPos);
			mod.RemoveBlankAtEnd();
			if(mod.CompareWithCharArray(".B1", 3))
				selector = 1;
			else if(mod.CompareWithCharArray(".B2", 3))
				selector = 2;
			else if(mod.CompareWithCharArray(".B3", 3))
				selector = 3;
			else if(mod.CompareWithCharArray(".H1", 3))
				selector = 5;
			else
				throw 152; //unrecognised sub-word selector modifier
		}
		//selector has been found out. 0 means no selector is used.
		//if selector is zero, then no need to confirm with the OpType
		if(selector!=0)
		{
			int opType = csCurrentInstruction.OpcodeWord1 & (7 << SelOffset);
			opType >>= SelOffset;
			//H1
			if(selector == 5 && opType != 4) //has to be S/U16, which uses an opType of 4
				throw 153; //operand's sub-word selector incompatible with type modifier
			//B1 to B3 use U8/S8, which should have an opType of 0
			else if(selector <=3 && opType != 0)
				throw 153;
			//U32/S32 uses an opType of 6, but it generates no selector so ignore
			//write back selector
			csCurrentInstruction.OpcodeWord1 |= selector << SelOffset;
		}
		if(negative)
		{
			component.Start--;
			component.Length++;
		}
		
	}
};
OperandRuleRegisterForVADD OPRRegister1ForVADD(20, 12, 8), OPRRegister2ForVADD(26, 0, 7), OPRRegister2ForI2I(26, 23, 8);
