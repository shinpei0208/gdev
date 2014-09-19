
#include "DataTypes.h"
#include "GlobalVariables.h"
#include "helper/helperException.h"
#include "SpecificParsers.h"

#include "stdafx.h" //Mark

#include "RulesDirective.h"
#include "RulesOperand/RulesOperandComposite.h"


//Kernel
struct DirectiveRuleKernel: DirectiveRule
{
	DirectiveRuleKernel()
	{
		Name = "Kernel";
	}
	virtual void Process() // Needs 3 arguments: Name
	{
		if(csOperationMode != DirectOutput)
			throw 1003; //only usable in direct output mode
		if(csCurrentDirective.Parts.size()!=2) //!Kernel KernelName
			throw 1002; //Incorrect number of directive arguments.
		if(csCurrentKernelOpened)
			throw 1004; //previous kernel without EndKernel
		if(csInstructions.size()!=0)
			hpWarning(12); //some instructions not included

		csCurrentKernelOpened = true;
		if(csCurrentDirective.Parts.size()!=2)
			throw 1010; //Incorrect number of directive parameters
		csCurrentKernel.KernelName = *csCurrentDirective.Parts.rbegin();
	}
}DRKernel;


void processLabels()
{
	list<SortElement> labelList;
	SortElement element;
	for(list<Label>::iterator label = csLabels.begin(); label!= csLabels.end(); label++)
	{
		element.ExtraInfo = &*label;
		element.Name = label->Name;
		labelList.push_back(element);
	}
	unsigned int *IndexList;
	SortElement *SortedList;
	SortInitialize(labelList, SortedList, IndexList);
	int count = labelList.size();
	//check for repeating label name
	for(int i =1; i<count; i++)
	{
		if(IndexList[i]==IndexList[i-1]&&SortedList[i].Name.Compare(SortedList[i-1].Name))
		{
			csCurrentDirective.Reset(csInstructions.begin()->InstructionString, csInstructions.begin()->LineNumber);
			throw 1023; //repeating label name
		}
	}
	//process each label request
	for(list<LabelRequest>::iterator request = csLabelRequests.begin(); request!=csLabelRequests.end(); request++)
	{
		SortElement found = SortFind(SortedList, IndexList, count, request->RequestedLabelName);
		if(found.ExtraInfo == SortNotFound.ExtraInfo)
		{
#if 0
			if(!request->Zero)
			{
				csCurrentDirective.Reset((++request->InstructionPointer)->InstructionString, request->InstructionPointer->LineNumber);
			}
			else
				csCurrentDirective.Reset(csInstructions.begin()->InstructionString, csInstructions.begin()->LineNumber);
#endif
			throw 1024;//label not found
		}
		int offset = ((Label*)found.ExtraInfo)->Offset;
		
		LabelProcessing = true;
		LabelAbsoluteAddr = offset;
		Instruction *relatedInstruction = &csInstructions[request->RequestIndex];
#if 0
		if(!request->Zero)
		{
			relatedInstruction = request->InstructionPointer; //on the one before the requesting instruction
			relatedInstruction++;//now on the requesting instruction
		}
		else
		{
			relatedInstruction = csInstructions.begin(); // on the requesting instruction
		}
#endif

		csInstructionOffset = relatedInstruction->Offset+(relatedInstruction->Is8?8:4); //offset of the instruction before it
		csCurrentInstruction = *relatedInstruction;
		((OperandRule*)&OPRInstructionAddress)->Process(request->RequestedLabelName);
		relatedInstruction->OpcodeWord0=csCurrentInstruction.OpcodeWord0;
		relatedInstruction->OpcodeWord1=csCurrentInstruction.OpcodeWord1;
	}
	delete[] SortedList;
	delete[] IndexList;
	csLabels.clear();
	csLabelRequests.clear();
}

//EndKernel
struct DirectiveRuleEndKernel: DirectiveRule
{
	DirectiveRuleEndKernel()
	{
		Name = "EndKernel";
	}
	virtual void Process()
	{
		if(!csCurrentKernelOpened)
			throw 1005; //without Kernel directive
		
		csCurrentKernel.TextSize = csInstructionOffset;
		processLabels();
		csCurrentKernel.KernelInstructions = csInstructions;
		
		if(csRegCount>csCurrentKernel.RegCount)
			csCurrentKernel.RegCount = csRegCount;
		if(csBarCount>csCurrentKernel.BarCount)
			csCurrentKernel.BarCount = csBarCount;
		
		csKernelList.push_back(csCurrentKernel);

		csInstructions.clear();
#ifdef DebugMode
		cout<<"RegCount:"<<csRegCount<<endl;
#endif
		csInstructionOffset = 0;
		csRegCount = 0;
		csBarCount = 0;
		csCurrentKernel.Reset();
		csCurrentKernelOpened = false;
	}
}DREndKernel;

struct DirectiveRuleLabel: DirectiveRule
{
	DirectiveRuleLabel()
	{
		Name = "Label";
	}
	virtual void Process() //!Label Name
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on Name
		Label label;
		SubString lName = *currentArg;
		lName.RemoveBlankAtEnd();
		label.Name = lName;
		label.Offset = csInstructionOffset;
		label.LineNumber = csLineNumber;
		csLabels.push_back(label);
	}
}DRLabel;

//Param
struct DirectiveRuleParam: DirectiveRule //!Param Size Count
{
	DirectiveRuleParam()
	{
		Name = "Param";
	}
	virtual void Process()
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()<2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size

		// check Ext : !Param E size1 size2 ...
		if (currentArg->Compare("E")) {
			if(csCurrentDirective.Parts.size()>258)
				throw 1002; //incorrect no. of arguments
			currentArg++; // skip E
			for (int i=2; i<csCurrentDirective.Parts.size();
				i++, currentArg++) {
				unsigned int size =
					currentArg->ToImmediate32FromInt32();
				KernelParameter param;
				param.Size = size;
				csCurrentKernel.ParamTotalSize =
					(csCurrentKernel.ParamTotalSize+size-1)
					& ~(size-1);
				param.Offset = csCurrentKernel.ParamTotalSize;
				csCurrentKernel.ParamTotalSize += size;
				csCurrentKernel.Parameters.push_back(param);
				if(csCurrentKernel.ParamTotalSize > 256)
					throw 1008;		
			}
			return;
		}

		if(csCurrentDirective.Parts.size()>3)
			throw 1002; //incorrect no. of arguments
		unsigned int size = currentArg->ToImmediate32FromInt32();
		if(size%4 !=0 )
			throw 1007; //size of parameter must be multiple of 4; issue: may not be necessary
		if(size>256)
			throw 1008; //size of parameter cannot be larger than 256

		unsigned int count = 1;
		if(csCurrentDirective.Parts.size()==3)
		{
			currentArg++;
			if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
				count = currentArg->ToImmediate32FromHexConstant(false);
			else
				count = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		}

		if(count>256)
			throw 1009; //prevent overflow

		for(int i =0; i<count; i++)
		{
			KernelParameter param;
			param.Size = size;
			param.Offset = csCurrentKernel.ParamTotalSize;
			csCurrentKernel.ParamTotalSize += size;
			csCurrentKernel.Parameters.push_back(param);
		}
		if(csCurrentKernel.ParamTotalSize > 256)
			throw 1008;		
	}
}DRParam;

//Shared
struct DirectiveRuleShared: DirectiveRule 
{
	DirectiveRuleShared()
	{
		Name = "Shared";
	}
	virtual void Process()//!Local Size 
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		csCurrentKernel.SharedSize = size;
	}
}DRShared;

//Local
struct DirectiveRuleLocal: DirectiveRule 
{
	DirectiveRuleLocal()
	{
		Name = "Local";
	}
	virtual void Process()//!Local Size 
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		csCurrentKernel.LocalSize = size;
	}
}DRLocal;

//MinStack
struct DirectiveRuleMinStack: DirectiveRule 
{
	DirectiveRuleMinStack()
	{
		Name = "MinStack";
	}
	virtual void Process()//!MinStack Size 
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		csCurrentKernel.MinStackSize = size;
	}
}DRMinStack;

//MinFrame
struct DirectiveRuleMinFrame: DirectiveRule 
{
	DirectiveRuleMinFrame()
	{
		Name = "MinFrame";
	}
	virtual void Process()//!MinFrame Size 
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		csCurrentKernel.MinFrameSize = size;
	}
}DRMinFrame;

//Constant2
struct DirectiveRuleConstant2: DirectiveRule //!Constant2 size
{
	DirectiveRuleConstant2()
	{
		Name = "Constant2";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments
		if(cubinConstant2Size)
			throw 1015; //Constant2 could be declared only once per cubin.

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		if(size>65536)
			throw 1014; //Maximal constant2 size supported is 65536 bytes.
		cubinConstant2Size = size;
		cubinSectionConstant2.SectionSize = size;
		cubinSectionConstant2.SectionContent = new unsigned char[size];
		memset(cubinSectionConstant2.SectionContent, 0, size);
	}
}DRConstant2;


struct DirectiveRuleConstant: DirectiveRule //!Constant type offset
{
	DirectiveRuleConstant()
	{
		Name = "Constant";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=3)
			throw 1002; //incorrect no. of arguments
		if(cubinConstant2Size==0)
			throw 1012; //constant2 size must be declared to non-zero before constant could be declared

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin();

		//type
		currentArg++;
		if(currentArg->Compare("int"))
			cubinCurrentConstant2Parser = &Constant2ParseInt;
		else if(currentArg->Compare("long"))
			cubinCurrentConstant2Parser = &Constant2ParseLong;
		else if(currentArg->Compare("float"))
			cubinCurrentConstant2Parser = &Constant2ParseFloat;
		else if(currentArg->Compare("double"))
			cubinCurrentConstant2Parser = &Constant2ParseDouble;
		else if(currentArg->Compare("mixed"))
			cubinCurrentConstant2Parser = &Constant2ParseMixed;
		else
			throw 1016; //Unsupported constant type


		//offset
		currentArg++;
		int offset;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			offset = currentArg->ToImmediate32FromHexConstant(false);
		else
			offset = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?

		if(offset>cubinConstant2Size)
			throw 1013; //Offset is larger than constant2

		cubinCurrentConstant2Offset = offset;
		csLineParserStack.push(csLineParser);
		csLineParser = (LineParser*)&LPConstant2;
	}
}DRConstant;


struct DirectiveRuleEndConstant: DirectiveRule //!Constant type offset
{
	DirectiveRuleEndConstant()
	{
		Name = "EndConstant";
	}
	virtual void Process()
	{
		csLineParser = csLineParserStack.top();
		csLineParserStack.pop();
	}
}DREndConstant;


struct DirectiveRuleRegCount: DirectiveRule 
{
	DirectiveRuleRegCount()
	{
		Name = "RegCount";
	}
	virtual void Process()//!RegCount count
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on count
		int count;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			count = currentArg->ToImmediate32FromHexConstant(false);
		else
			count = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		if(count>63)
			throw 1019;//no larger than 63
		csCurrentKernel.RegCount = count;
	}
}DRRegCount;


struct DirectiveRuleBarCount: DirectiveRule 
{
	DirectiveRuleBarCount()
	{
		Name = "BarCount";
	}
	virtual void Process()//!BarCount count
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on count
		int count;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			count = currentArg->ToImmediate32FromHexConstant(false);
		else
			count = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		if(count>127)
			throw 1020;//no larger than 127
		if(count>16)
			hpWarning(13); //warn for large count
		csCurrentKernel.BarCount = count;
	}
}DRBarCount;

struct DirectiveRuleRawInstruction: DirectiveRule
{
	DirectiveRuleRawInstruction()
	{
		Name = "RawInstruction";
	}
	virtual void Process()//!RawInstruction 0xabcd (0xabcd)
	{
		if(csCurrentDirective.Parts.size()<2 || csCurrentDirective.Parts.size()>3)
			throw 1002; //incorrect no. of arguments
		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on first hex
		Instruction inst;
		inst.Reset(csCurrentDirective.DirectiveString, csInstructionOffset, csLineNumber); //issue: csLineNumber is not being updated anywhere
		inst.Is8 = csCurrentDirective.Parts.size()==3;
		inst.OpcodeWord0 = currentArg->ToImmediate32FromHexConstant(false);
		csInstructionOffset+=4;
		if(inst.Is8)
		{
			currentArg++;
			inst.OpcodeWord1 = currentArg->ToImmediate32FromHexConstant(false);
			csInstructionOffset+=4;
		}
		csInstructions.push_back(inst);
	}
}DRRawInstruction;


struct DirectiveRuleArch: DirectiveRule
{
	DirectiveRuleArch()
	{
		Name = "Arch";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002;
		list<SubString>::iterator part = csCurrentDirective.Parts.begin();
		part++;
		char zeroSaver = part->Start[part->Length];
		part->Start[part->Length] = 0;
		if(strcmp("sm_20", part->Start)==0)
		{
			cubinArchitecture = sm_20;
		}
		else if(strcmp("sm_21", part->Start)==0)
		{
			cubinArchitecture = sm_21;
		}
		else if(strcmp("sm_30", part->Start)==0)
		{
			cubinArchitecture = sm_30;
		}
		else
			throw 1021;// unsupported argument
	}
}DRArch;

struct DirectiveRuleMachine: DirectiveRule
{
	DirectiveRuleMachine()
	{
		Name = "Machine";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002;
		list<SubString>::iterator part = csCurrentDirective.Parts.begin();
		part++;
		char zeroSaver = part->Start[part->Length];
		part->Start[part->Length] = 0;
		if(strcmp("64", part->Start)==0)
		{
			hpCubinSet64(true);
		}
		else if(strcmp("32", part->Start)==0)
		{
			hpCubinSet64(false);
		}
		else
			throw 1021;// unsupported argument
	}
}DRMachine;



struct DirectiveRuleAlign: DirectiveRule
{
	Instruction nop;
	DirectiveRuleAlign()
	{
		nop.Is8 = true;
		nop.OpcodeWord0=0x00001de4;
		nop.OpcodeWord1=0x40000000;
		Name = "Align";
	}
	virtual void Process()
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on count
		SubString state = *currentArg;
		state.RemoveBlankAtEnd();
		bool eight = false;
		if(state.Compare("0"))
			eight = false;
		else if(state.Compare("8"))
			eight = true;
		else
			throw 1025; //unsupported argument
		bool aligned8 = (csInstructionOffset%16==8);
		if(eight^aligned8)
		{
			csInstructions.push_back(nop);
			csInstructionOffset+=8;
		}

	}
}DRAlign;


struct DirectiveRuleSelfDebug: DirectiveRule
{
	DirectiveRuleSelfDebug()
	{
		Name = "SelfDebug";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002;
		list<SubString>::iterator part = csCurrentDirective.Parts.begin();
		part++;
		char zeroSaver = part->Start[part->Length];
		part->Start[part->Length] = 0;
		if(strcmp("On", part->Start)==0)
		{
			csSelfDebug = true;
		}
		else if(strcmp("Off", part->Start)==0)
		{
			csSelfDebug = false;
		}
		else
			throw 1022;// unsupported argument
	}
}DRSelfDebug;
