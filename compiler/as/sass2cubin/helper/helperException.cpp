#include "../DataTypes.h"
#include "../GlobalVariables.h"
#include "helperException.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include <exception>


extern void hpUsage();
void hpExceptionHandler(int e)		//===
{
	if(csSelfDebug)
		throw exception();
	char *message;
	switch(e)
	{
	case 0:		message = "Invalid arguments.";
		break;
	case 1:		message = "Unable to open input file";
		break;
	case 3:		message = "Incorrect kernel offset.";
		break;
	case 4:		message = "Cannot open output file";
		break;
	case 5:		message = "Cannot find specified kernel";
		break;
	case 6:		message = "File to be modified is invalid.";
		break;
	case 7:		message = "Failed to read cubin";
		break;
	case 8:		message = "Cannot find the specified section";
		break;
	case 9:		message = "Specific section not large enough to contain all the assembled opcodes.";
		break;
	case 10:	message = "Single-line mode only supported in Replace Mode.";
		break;
	case 20:	message = "Insufficient number of arguments";
		break;
	case 50:	message = "Initialization error. Repeating rule indices. Please debug to find out the exact repeating entry.";
		break;
	case 97:	message = "Operation mode not supported.";
		break;
	case 98:	message = "Cannot proceed due to errors.";
		break;
	case 99:	message = "Command-line argument used is not supported.";
		break;
	case 100:	message = "Last kernel is not ended with EndKernel directive.";
		break;
	case 101:	message = "No valid kernel is found in file.";
		break;
	default:	message = "No error message";
		break;
	};
	cout<<message<<endl;
	if(csExceptionPrintUsage)
		hpUsage();
#ifdef DebugMode
	getchar();
#endif
}
void hpInstructionErrorHandler(int e)
{
	if(csSelfDebug)
		throw exception();
	csErrorPresent = true;
	char *message;
	switch(e)
	{
	case 100:	message = "Instruction name is absent following the predicate";
		break;
	case 101:	message = "Unsupported modifier.";
		break;
	case 102:	message = "Too many operands.";
		break;
	case 103:	message = "Insufficient number of operands.";
		break;
	case 104:	message = "Incorrect register format.";
		break;
	case 105:	message = "Register number too large.";
		break;
	case 106:	message = "Incorrect hex value.";
		break;
	case 107:	message = "Incorrect memory operand format.";
		break;
	case 108:	message = "Instruction not supported.";
		break;
	case 109:	message = "Incorrect predicate.";
		break;
	case 110:	message = "Incorrect constant memory format.";
		break;
	case 111:	message = "Memory address for constant memory too large.";
		break;
	case 112:	message = "Register cannot be used in composite constant memory operand.";
		break;
	case 113:	message = "The immediate value is limited to 20-bit.";
		break;
	case 114:	message = "Constant memory bank number too large.";
		break;
	case 115:	message = "Immediate value is limited to 20-bit.";
		break;
	case 116:	message = "Invalid operand.";
		break;
	case 117:	message = "Incorrect floating number.";
		break;
	case 118:	message = "20-bit immediate value cannot contain 64-bit number.";
		break;
	case 119:	message = "Register cannot be used in FADD-style constant address.";
		break;
	case 120:	message = "Only constants can be negative for MOV-style operand.";
		break;
	case 121:	message = "Incorrect floating point number format.";
		break;
	case 122:	message = "Too many modifiers present.";
		break;
	case 123:	message = "Empty modifier.";
		break;
	case 124:	message = "Instruction name not present.";
		break;
	case 125:	message = "Empty operand.";
		break;
	case 126:	message = "Incorrect predicate.";
		break;
	case 127:	message = "Insufficient number of modifiers";
		break;
	case 128:	message = "Incorrect special register name.";
		break;
	case 129:	message = "Negative sign cannot be used here.";
		break;
	case 130:	message = "Shared memory address immediate value cannot be longer than 20 bits.";
		break;
	case 131:	message = "Offset is limited to 24-bit long.";
		break;
	case 132:	message = "Empty operand.";
		break;
	case 133:	message = "Shift cannot be larger than 31";
		break;
	case 134:	message = "Negative sign not allowed here.";
		break;
	case 135:	message = "Incorrect NOP operand";
		break;
	case 136:	message = "Value limited to 16 bits";
		break;
	case 137:	message = "This operand does not accept constant memory address with a register.";
		break;
	case 138:	message = "The address must be a multiple of 4.";
		break;
	case 139:	message = "barrier identifier greater than 127 is not supported. Barrier identifier should normally be less than 16.";
		break;
	case 140:	message = "Thread count should be no larger than 4095.";
		break;
	case 141:	message = "The memory offset is a 20-bit signed integer.";
		break;
	case 142:	message = "Unrecognised operand present at the end.";
		break;
	case 143:	message = "The resulting address of the instruction is negative. If this is the intended behaviour, please use relative addressing mode";
		break;
	case 144:	message = "Only numerical expression can be used as barrier identifier here.";
		break;
	case 145:	message = "Empty label name.";
		break;
	case 146:	message = "Constant memory cannot be used with this instruction.";
		break;
	case 147:	message = "The register number is too large to be used with .64/128.";
		break;
	case 148:	message = "Constant memory address should be a multiple of 4. Multiples of 2 are allowed for the same of experimentation.";
		break;
	case 149:	message = "The register number is too large to be used with .E";
		break;
	case 150:	message = "This immediate offset of this memory operand is limited to 24 bits.";
		break;
	case 151:	message = "Register number should be less than 62.";
		break;
	case 152:	message = "Unrecognised sub-word selector modifier.";
		break;
	case 153:	message = "Operand's sub-word selector is incompatible with its corresponding OpType modifier";
		break;
	case 154:	message = "VADD's immediate operand cannot be longer than 16 bits";
		break;
	case 155:	message = "Schedule number cannot exceed 0xff, or 11111111";
		break;
	case 156:	message = "Incorrect binary number";
		break;
	default:	message = "Unknown Error.";
		break;
	};
	char *line = csCurrentInstruction.InstructionString.ToCharArrayStopOnCR();
	cout<<"Error Line "<<csCurrentInstruction.LineNumber<<": "<<line<<endl<<"	"<<message<<endl;
	delete[] line;
}
void hpDirectiveErrorHandler(int e)
{
	if(csSelfDebug)
		throw exception();
	csErrorPresent = true;
	char *message;
	switch(e)
	{
	case 1000:	message = "Empty Instruction.";
		break;
	case 1001:	message = "Unsupported directive.";
		break;
	case 1002:	message = "Incorrect number of directive arguments.";
		break;
	case 1003:	message = "Kernel directive can only be used in DirectOutput Mode.";
		break;
	case 1004:	message = "Without an EndKernel directive, instructions in the previous kernel will be ignored.";
		break;
	case 1005:	message = "Corresponding Kernel directive not found.";
		break;
	case 1006:	message = "Parameters can only be defined within kernel sections.";
		break;
	case 1007:	message = "Size of parameter must be multiples of 4.";
		break;
	case 1008:	message = "Size of parameters cannot exceed 256 bytes.";
		break;
	case 1009:	message = "Parameter count too large.";
		break;
	case 1010:	message = "Incorrect number of kernel arguments.";
		break;
	case 1011:	message = "Unsupported architecture. Please use only sm_20, sm_21 or sm_30"; //issue: architecture support may increase in future
		break;
	case 1012:	message = "Constant2 size must be declared to non-zero first before constant could be declared.";
		break;
	case 1013:	message = "Offset is larger than constant2.";
		break;
	case 1014:	message = "Maximal constant2 size supported is 65536 bytes.";
		break;
	case 1015:	message = "Constant2 could be declared only once per cubin.";
		break;
	case 1016:	message = "Unsupported constant type.";
		break;
	case 1017:	message = "Constant object too large.";
		break;
	case 1018:	message = "Next directive can only be EndConstant.";
		break;
	case 1019:	message = "RegCount should not be larger than 63.";
		break;
	case 1020:	message = "BarCount should not be larger than 127.";
		break;
	case 1021:	message = "Unsupported mode.";
		break;
	case 1022:	message = "Mode can only be On or Off.";
		break;
	case 1023:	message = "Repeating label name.";
		break;
	case 1024:	message = "Label not found.";
		break;
	case 1025:	message = "Unsupported argument.";
		break;
	default:	message = "Unknown error.";
		break;
	};
	
	char *line = csCurrentDirective.DirectiveString.ToCharArrayStopOnCR();
	cout<<"Error Line "<<csCurrentDirective.LineNumber<<": "<<line<<endl<<"	"<<message<<endl;
	delete[] line;
}

void hpWarning(int e)
{
	if(csSelfDebug)
		throw exception();
	char* message;
	switch(e)
	{
	case 10:	message = "Evaluation of constant returned zero. Consider using RZ if value of zero is intended.";
		break;
	case 11:	message = "Evaluation of constant overflowed.";
		break;
	case 12:	message = "Some instructions before the kernel section are not included in any kernel.";
		break;
	case 13:	message = "BarCount usually should be limited to 16. Values larger than 16 and lower than 128 is allowed only for the sake of experimentation.";
		break;
	case 14:	message = "Constant memory address should be a multiple of 4.";
		break;
	default:	message = "No warning message available.";
		break;
	}
	char *line = csCurrentLine.LineString.ToCharArrayStopOnCR();
	cout<<"Warning Line "<<csCurrentInstruction.LineNumber<<": "<<line<<endl<<"	"<<message<<endl;
	delete[] line;
}
