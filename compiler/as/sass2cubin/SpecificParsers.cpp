#include "SpecificParsers.h"
#include "GlobalVariables.h"
#include "helper/helperException.h"
#include "helper/helperParse.h"

#include "stdafx.h" //Mark
using namespace std;

struct MasterParserDefault: MasterParser
{
	MasterParserDefault()
	{
		Name = "DefaultMasterParser";
	}
	void Parse(unsigned int startinglinenumber);
}MPDefault;
struct LineParserDefault : LineParser
{
	LineParserDefault()
	{
		Name = "DefaultLineParser";
	}
	void Parse(Line &line);
}LPDefault;
struct InstructionParserDefault: InstructionParser
{
	InstructionParserDefault()
	{
		Name = "DefaultInstructionParser";
	}
	void Parse();
}IPDefault;
struct DirectiveParserDefault: DirectiveParser
{
	DirectiveParserDefault()
	{
		Name = "DefaultDirectiveParser";
	}
	void Parse();
}DPDefault;
//-----End of default parser declarations



//	2
//-----Implementation of the parse() functions for parsers declared in 1.
void MasterParserDefault:: Parse(unsigned int startinglinenumber)
{
	vector<Line>::iterator cLine = csLines.begin(); //current line	
	
	//Going through all lines
	for(unsigned int i =startinglinenumber; i<csLines.size(); i++, cLine++)
	{
		csCurrentLine = *cLine;
		cLine->LineString.RemoveBlankAtBeginning();
		//Jump to next line if there's nothing in this line
		if(cLine->LineString.Length<=0)
			continue;
		csLineParser->Parse(*cLine);
	}
}

//entire thing is done in the masterparser. So this is left empty for now.
void LineParserDefault:: Parse(Line &line)
{
	int lineLength = line.LineString.Length;
	if(line.LineString[0]=='!') //if it's directive, build it, parse it and append it to csDirectives. Issue: the first character of the line must be '!'
	{
		try
		{
			//build the new directive
			csCurrentDirective.Reset(line.LineString, line.LineNumber);
			csDirectiveParser->Parse();						//parse it. the parser will decide whether to append it to csDirectives or not
		}
		catch(int e)
		{
			hpDirectiveErrorHandler(e);
		}
	}
	else //if it's not directive, it's instruction. Break it if it has ';'
	{
		try
		{
			//look for instruction delimiter ';'				
			int startPos = 0;
			int lastfoundpos = line.LineString.Find(';', startPos); //search for ';', starting at startPos
			while(lastfoundpos!=-1)
			{
				//Build an instruction, parse it and the parser will decide whether to append it to csInstructions or not
				csCurrentInstruction.Reset(line.LineString.SubStr(startPos, lastfoundpos - startPos), csInstructionOffset, line.LineNumber);
				csInstructionParser->Parse();
				startPos = lastfoundpos + 1; //starting position of next search
				lastfoundpos = line.LineString.Find(';', startPos); //search for ';', starting at startPos
			}
			//still have to deal with the last part of the line, which may not end with ';'
			if(startPos < lineLength)
			{
				csCurrentInstruction.Reset(line.LineString.SubStr(startPos, lineLength - startPos), csInstructionOffset, line.LineNumber);
				csInstructionParser->Parse();
			}
		}
		catch(int e)
		{
			hpInstructionErrorHandler(e);
		}			
	}
}


void InstructionParserDefault:: Parse()
{
	hpParseBreakInstructionIntoComponents();

	
	//Start

	if(csCurrentInstruction.Components.size()==0)
		return;

	int processedComponent = 0;
	int OPPresent; //number of operands present
	list<SubString>::iterator component = csCurrentInstruction.Components.begin();
	list<SubString>::iterator modifier = csCurrentInstruction.Modifiers.begin();
	
	//---skip predicate expression first, process later
	if(csCurrentInstruction.Predicated)
	{
		component++; processedComponent++;		
	}

	//---instruction name
	if(component == csCurrentInstruction.Components.end())
			throw 100; //no instruction name present
	int nameIndex = hpParseComputeInstructionNameIndex(*component);
	int arrayIndex = hpParseFindInstructionRuleArrayIndex(nameIndex);
	if(arrayIndex == -1)
	{
		throw 108; //instruction not supported
	}

	csCurrentInstruction.Is8 = csInstructionRules[arrayIndex]->Is8;
	csCurrentInstruction.Offset = csInstructionOffset;
	csCurrentInstruction.OpcodeWord0 = csInstructionRules[arrayIndex]->OpcodeWord0;
	csInstructionOffset += 4;
	if(csCurrentInstruction.Is8)
	{
		csCurrentInstruction.OpcodeWord1 = csInstructionRules[arrayIndex]->OpcodeWord1;
		csInstructionOffset += 4;
	}
	if(csCurrentInstruction.Predicated)
		hpParseProcessPredicate();

	if(csInstructionRules[arrayIndex]->NeedCustomProcessing)
	{
		csInstructionRules[arrayIndex]->CustomProcess();
		goto APPEND;
	}

	




	//---instruction modifiers
	if(csCurrentInstruction.Modifiers.size()>csInstructionRules[arrayIndex]->ModifierGroupCount)
		throw 122;//too many modifiers.
	for(int modGroupIndex = 0; modGroupIndex < csInstructionRules[arrayIndex]->ModifierGroupCount; modGroupIndex++)
	{
		if(modifier==csCurrentInstruction.Modifiers.end())
		{
			//ignore if all the following groups are optional
			if(csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].Optional)
				continue;
			else
				throw 127; //insufficient number of modifiers
		}
		int i = 0;
		for( ; i < csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierCount; i++)
		{
			if(modifier->Compare(csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierRules[i]->Name))
				break;
		}
		//modifier name not found in this group
		if(i==csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierCount)
		{
			//Can ignore this modGroup if it is optional
			if(csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].Optional)
				continue;
			else
				throw 101; //unsupported modifier
		}
		hpParseApplyModifier(*csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierRules[i]);
		modifier++;
	}	
	if(modifier!=csCurrentInstruction.Modifiers.end())
		throw 101; //issue: the error line should be something else
	component++; processedComponent++;



	//---Operands
	OPPresent = csCurrentInstruction.Components.size() - processedComponent; //OPPresent is the number of operands that are present
	if(OPPresent > csInstructionRules[arrayIndex]->OperandCount)
	{
		throw 102; //too many operands
		return;
	}
	
	for(int i=0; i<csInstructionRules[arrayIndex]->OperandCount; i++)
	{
		if(component == csCurrentInstruction.Components.end())
		{
			if(csInstructionRules[arrayIndex]->Operands[i]->Type == Optional)
				continue;
			else
				throw 103; //insufficient operands.
		}
		//process operand
		csInstructionRules[arrayIndex]->Operands[i]->Process(*component);
		component++;
		//process modifiers
		//not done yet
	}
APPEND:
	csInstructions.push_back(csCurrentInstruction);
}

void DirectiveParserDefault:: Parse()
{
	hpParseBreakDirectiveIntoParts();
	if(csCurrentDirective.Parts.size()==0)
		throw 1000; //empty directive

	int index = hpParseComputeDirectiveNameIndex(*csCurrentDirective.Parts.begin());
	int arrayIndex = hpParseFindDirectiveRuleArrayIndex(index);
	if(arrayIndex==-1)
		throw 1001; //unsupported directive

	csDirectiveRules[arrayIndex]->Process();

	csDirectives.push_back(csCurrentDirective);
}

//-----End of parse() function implementation for default parsers


struct LineParserConstant2 : LineParser
{
	LineParserConstant2()
	{
		Name = "ConstantLineParser";
	}
	void Parse(Line &line)
	{
		
		try
		{
			if(cubinConstant2Overflown)
				return;
			//directive
			if(line.LineString[0]=='!')
			{
					//build the new directive
				csCurrentDirective.Reset(line.LineString, line.LineNumber);
				hpParseBreakDirectiveIntoParts();
				if(csCurrentDirective.Parts.size()!=1 || !csCurrentDirective.Parts.begin()->Compare("EndConstant"))
					throw 1018; //Next directive can only be EndConstant
				csDirectiveParser->Parse();
			}
			//constant
			else
			{	
				int startPos = 0;
				int lastfoundpos = line.LineString.Find(',', startPos); //search for ';', starting at startPos
				while(lastfoundpos!=-1)
				{
					SubString s = line.LineString.SubStr(startPos, lastfoundpos - startPos);
					cubinCurrentConstant2Parser(s);
					if(cubinCurrentConstant2Offset>cubinConstant2Size)
					{
						cubinConstant2Overflown = true;
						throw 1017; //constant object too large
						return;
					}
					startPos = lastfoundpos + 1;
					lastfoundpos = line.LineString.Find(',', startPos);
				}
				if(startPos < line.LineString.Length)
				{
					SubString s = line.LineString.SubStr(startPos, line.LineString.Length - startPos);
					cubinCurrentConstant2Parser(s);
					if(cubinCurrentConstant2Offset>cubinConstant2Size)
					{
						cubinConstant2Overflown = true;
						throw 1017;
						return;
					}
				}
			}
		}
		catch(int e)
		{
			hpDirectiveErrorHandler(e);
		}
	}
}LPConstant2;

//issue: the parsers do not report error if result 0 is returned

void Constant2ParseInt(SubString &content)
{
	content.SubEndWithNull();
	int result = atoi(content.Start);
	content.RecoverEndWithNull();

	*(int*)(cubinSectionConstant2.SectionContent+cubinCurrentConstant2Offset) = result;
	cubinCurrentConstant2Offset+=4;
}

void Constant2ParseLong(SubString &content)
{
	content.SubEndWithNull();
	long long result = atol(content.Start);
	content.RecoverEndWithNull();

	*(long long *)(cubinSectionConstant2.SectionContent+cubinCurrentConstant2Offset) = result;
	cubinCurrentConstant2Offset+=8;
}

void Constant2ParseFloat(SubString &content)
{
	content.SubEndWithNull();
	float result = atof(content.Start);
	content.RecoverEndWithNull();
	*(float *)(cubinSectionConstant2.SectionContent+cubinCurrentConstant2Offset) = result;
	cubinCurrentConstant2Offset+=4;
}
void Constant2ParseDouble(SubString &content)
{
	content.SubEndWithNull();
	double result = atof(content.Start);
	content.RecoverEndWithNull();

	*(double *)(cubinSectionConstant2.SectionContent+cubinCurrentConstant2Offset) = result;
	cubinCurrentConstant2Offset+=8;
}

void Constant2ParseMixed(SubString &content)
{
	content.RemoveBlankAtBeginning();
	if(content.Length==0)
	{
		cubinCurrentConstant2Offset += 4;
		return; //
	}
	unsigned int result;
	if(content[0]=='F')
		result = content.ToImmediate32FromFloatConstant();
	else if(content.Length>2 && content[0]=='0' && (content[1]=='X' || content[1]=='x'))
		result = content.ToImmediate32FromHexConstant(true);
	else
		result = content.ToImmediate32FromIntConstant();

	*(unsigned int *)(cubinSectionConstant2.SectionContent+cubinCurrentConstant2Offset) = result;
	cubinCurrentConstant2Offset+=4;
}
