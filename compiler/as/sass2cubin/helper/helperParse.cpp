
#include "../SubString.h"
#include "../GlobalVariables.h"
#include "helperParse.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark


//Break a directive line into its parts (directive name, arguments)
//The '!' at the beginning is not included in the first part
void hpParseBreakDirectiveIntoParts()
{
	SubString &directiveString = csCurrentDirective.DirectiveString;
	int startPos =1;
	int currentPos = 1;
	while(true)
	{
		for(; currentPos<directiveString.Length; currentPos++)
		{
			startPos = currentPos;
			if(directiveString[currentPos]>32)
				break;
		}

		for(; currentPos < directiveString.Length; currentPos++)
		{
			if(directiveString[currentPos] <33)
			{
				csCurrentDirective.Parts.push_back(directiveString.SubStr(startPos, currentPos - startPos));
				currentPos ++;
				startPos = currentPos;
				break;
			}
		}
		if(currentPos == directiveString.Length)
		{
			if(startPos<currentPos)
				csCurrentDirective.Parts.push_back(directiveString.SubStr(startPos, currentPos - startPos));
			return;
		}
	}


}





int b_startPos;				//starting Position of a non-blank character in the instruction string
int b_currentPos;			//current search position in the instruction string. For the macros, this is the position of dot
int b_lineLength;			//length of the instruction string

//b_startPos = b_currentPos = position of first non-blank character found in [b_currentPos, b_lineLength)
//When no non-blank character, it causes hpBreakInstructionIntoComponents to directly return
#define mSkipBlank																							\
{																											\
	for(; b_currentPos < b_lineLength; b_currentPos++)														\
	{																										\
		if((int)csCurrentInstruction.InstructionString[b_currentPos]>32)									\
		{																									\
			b_startPos = b_currentPos;																		\
			break;																							\
		}																									\
	}																										\
	if(b_currentPos == b_lineLength)return;																	\
}

#define mExtract(startPos,cutPos) csCurrentInstruction.InstructionString.SubStr(startPos, cutPos-startPos)
#define mExtractPushComponent {	csCurrentInstruction.Components.push_back(mExtract(b_startPos, b_currentPos));}
#define mExtractPushModifier { if(b_currentPos==b_startPos){throw 123;} csCurrentInstruction.Modifiers.push_back( mExtract(b_startPos, b_currentPos));}

void hpParseBreakInstructionIntoComponents()
{
	b_currentPos = 0;
	b_startPos = 0;
	b_lineLength = csCurrentInstruction.InstructionString.Length;
	csCurrentInstruction.Modifiers.clear();
	csCurrentInstruction.Components.clear();



	//---predicate
	//mSkipBlank; //blank is skipped by line parser
	if(csCurrentInstruction.InstructionString[b_currentPos]=='@')
	{
PRED:
		b_currentPos++;
		if(b_currentPos==b_lineLength)
		{
			csCurrentInstruction.Predicated = true;
			mExtractPushComponent;
			return;
		}
		if(csCurrentInstruction.InstructionString[b_currentPos] < 33)
		{
			csCurrentInstruction.Predicated = true;
			mExtractPushComponent;
		}
		else
			goto PRED;
	}
	else
		csCurrentInstruction.Predicated = false;




	//---instruction name and modifiers
	mSkipBlank;
	bool instNameEnded = false;
INST:
	if(b_currentPos==b_lineLength)
	{
		mExtractPushComponent;
		return;
	}
	if( csCurrentInstruction.InstructionString[b_currentPos] < 33 )
	{
		mExtractPushComponent;
		mSkipBlank;
		if(csCurrentInstruction.InstructionString[b_currentPos]=='.')
		{
			b_currentPos++; b_startPos = b_currentPos;
			goto INSTDOT;
		}
		else
			goto OPCONTINUE;
	}
	else if(csCurrentInstruction.InstructionString[b_currentPos] == '.')
	{
		if(b_currentPos==b_startPos)
			throw 124; //empty instruction name
		mExtractPushComponent;
		b_currentPos++; b_startPos = b_currentPos;
INSTDOT:
		//End of line
		if(b_currentPos==b_lineLength)
		{
			mExtractPushModifier;
			return;
		}
		//encounter first blank right after the modifier name
		else if( csCurrentInstruction.InstructionString[b_currentPos] < 33 )
		{
			mExtractPushModifier;
			mSkipBlank;
			if(csCurrentInstruction.InstructionString[b_currentPos]=='.')
			{
				b_currentPos++; b_startPos = b_currentPos;
				goto INSTDOT;
			}
			else
				goto OPCONTINUE;
		}
		// start of another modifier
		else if(csCurrentInstruction.InstructionString[b_currentPos] == '.')
		{
			mExtractPushModifier;
			b_currentPos++; b_startPos = b_currentPos;
			goto INSTDOT;
		}
		else
		{
			b_currentPos++;
			goto INSTDOT;
		}
	}
	else
	{
		b_currentPos++;
		goto INST;
	}




	//---Operands
OPNEW:
	mSkipBlank;

OPCONTINUE:
	//EOL. ';' will not be encountered as it is eliminated by default line parser
	if(b_currentPos==b_lineLength)
	{
		if(b_currentPos==b_startPos)
			throw 125; //empty operand
		mExtractPushComponent;
		return;
	}
	//comma, start of another operand
	else if( csCurrentInstruction.InstructionString[b_currentPos] == ',' ) //space can exist in operands
	{
		if(b_currentPos==b_startPos)
			throw 125;
		mExtractPushComponent;
		b_currentPos++; b_startPos = b_currentPos;
		goto OPNEW;
	}
	else
	{
		b_currentPos++;
		goto OPCONTINUE;
	}
}


int hpParseComputeInstructionNameIndex(SubString &name)
{
	int len = name.Length;
	int index = 0;
	if(len>0)
	{
		index += (int)name[0] * 2851;
		if(len>1)
		{
			index += (int)name[1] * 349;
			for(int i =2; i<len; i++)
				index += (int)name[i];
		}
	}
	return index;
}
int hpParseFindInstructionRuleArrayIndex(int Index)
{
	int start = 0; //inclusive
	int end = csInstructionRuleCount; //exclusive
	int mid;
	while(start<end) //still got unchecked numbers
	{
		mid = (start+end)/2;
		if(Index > csInstructionRuleIndices[mid])
			start = mid + 1;
		else if(Index < csInstructionRuleIndices[mid])
			end = mid;
		else
			return mid;
	}
	return -1;
}

int hpParseComputeDirectiveNameIndex(SubString &name)
{
	int len = name.Length;
	int index = 0;
	if(len>0)
	{
		index += (int)name[0] * 128 * 128 * 128;
		if(len>1)
		{
			index += (int)name[1] * 128 * 128;
			for(int i = 2; i<len-1; i++)
				index += (int)name[i];
			index += (int)name[len-1] * 128;
		}
	}
	return index;
}
int hpParseFindDirectiveRuleArrayIndex(int Index)
{
	int start = 0; //inclusive
	int end = csDirectiveRuleCount; //exclusive
	int mid;
	while(start<end) //still got unchecked numbers
	{
		mid = (start+end)/2;
		if(Index > csDirectiveRuleIndices[mid])
			start = mid + 1;
		else if(Index < csDirectiveRuleIndices[mid])
			end = mid;
		else
			return mid;
	}
	return -1;
}

void hpParseApplyModifier(ModifierRule &rule)
{
	if(rule.NeedCustomProcessing)
	{
		rule.CustomProcess();
	}
	else
	{
		if(rule.Apply0)
		{
			csCurrentInstruction.OpcodeWord0 &= rule.Mask0;
			csCurrentInstruction.OpcodeWord0 |= rule.Bits0;
		}
		if(csCurrentInstruction.Is8 && rule.Apply1)
		{
			csCurrentInstruction.OpcodeWord1 &= rule.Mask1;
			csCurrentInstruction.OpcodeWord1 |= rule.Bits1;
		}
	}
}

static unsigned int predRef[]={0u, 1<<10, 2<<10, 3<<10, 4<<10, 5<<10, 6<<10, 7<<10};
static unsigned int predRefNegate = 1<<13;
static unsigned int predRefMask = 0xFFFFC3FF;
void hpParseProcessPredicate()
{
	SubString &predStr = *csCurrentInstruction.Components.begin();
	if(predStr.Length < 3)
		throw 109; //incorrect predicate
	bool negate = false;
	int startPos = 1;
	if(predStr[startPos]=='!')
	{
		if(predStr.Length<4)
			throw 109;
		negate = true;
		startPos++;
	}
	if(predStr[startPos] != 'P' && predStr[startPos] != 'p')
		throw 109;
	int predNumber = (int) predStr[startPos+1];
	if(predNumber < 48 || predNumber > 55)
	{
		if(predNumber != 90 && predNumber != 122)
			throw 109;
		predNumber = 7; //pt is seven
	}
	else
		predNumber -= 48;

	csCurrentInstruction.OpcodeWord0 &= predRefMask;
	csCurrentInstruction.OpcodeWord0 |= predRef[predNumber];
	if(negate)csCurrentInstruction.OpcodeWord0 |= predRefNegate;
}
//-----End of parser helper functions
