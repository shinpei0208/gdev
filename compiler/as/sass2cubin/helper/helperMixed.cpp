
#include <sstream>
#include "../GlobalVariables.h"
#include "helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark


//	1
//-----Main helper functions

void hpCleanUp() //parsers are created with the new keyword,        =====
{
	delete[] csInstructionRules;
	delete[] csInstructionRuleIndices;

	for(list<Kernel>::iterator i = csKernelList.begin(); i != csKernelList.end(); i++)
	{
		if(i->TextSection.SectionContent)
			delete i->TextSection.SectionContent;
		if(i->Constant0Section.SectionContent)
			delete i->Constant0Section.SectionContent;
		if(i->InfoSection.SectionContent)
			delete i->InfoSection.SectionContent;
		if(i->SharedSection.SectionContent)
			delete i->SharedSection.SectionContent;
		if(i->LocalSection.SectionContent)
			delete i->LocalSection.SectionContent;
	}

	delete[] csSource;
	if(csInput.is_open())
		csInput.close();
	if(csOutput.is_open())
		csOutput.close();
}
//-----End of main helper functions









//	2
//----- Command-line stage helper functions
void hpUsage()
{
	puts("sass2cubin");
	puts("based on asfermi Version 0.3.0. Updated: 10 August 2011");
	puts("");
	puts("Usage:");
	puts("sass2cubin sourcefile [Options [option arguments]]");
	puts("Source file must be the first command-line option. However, it could be repla-");
	puts("ced by a -I option specified below.");
	puts("Options:");
	puts("  -I \"instruction\": This can be used to replace the inputfile.");
	puts("     A single line of instruction, surrounded by double quotation marks, will");
	puts("     be prrocessed as source input. Note that comment is not supported in thi-");
	puts("     s mode.");
	puts("  -o outputfile: Output cubin to the specified file.");
	puts("  -r target_cubin kernel_name offset: Replace the opcodes in specified locati-");
	puts("     on of a kernel in a specified cubin file with the assembled opcodes.");
	puts("  -sm_20: output cubin for architecture sm_20.");
	puts("  -sm_21: output cubin for architecture sm_21.");
	puts("  -sm_30: output cubin for architecture sm_30. This is the default architectu-");
	puts("     re assumed by sass2cubin.");
	puts("  -32: output 32-bit cubin.");
	puts("  -64: output 64-bit cubin. This is the default behaviour.");
	puts("  -SelfDebug: throw unhandled exception when things go wrong. For debugging o-");
	puts("     f sass2cubin only.");
}



int hpHexCharToInt(char* str)
{
	int result;
	//the string must start with "0x" or "0X"
	if(	str[0]!='0' ||
		!(str[1]=='x'|| str[1]=='X') ||
		(int)str[2]==0)
		throw 3; //incorrect kernel offset
	int numberlength = 0;
	stringstream ss;
	ss<<std::hex<<str;
	ss>>result;
	if(result==0) //when result = 0, str must be 0x00000..
	{
		for(int i =2; i<20; i++)
		{
			if(str[i]==0) //stop if end is reached.
				break;
			if(str[i]!='0') //when result is zero, all digitis ought to be '0' as well.
				throw 3;    //When this is not true, throws exception 3
		}
	}
	return result;
}


int hpFileSizeAndSetBegin(fstream &file)		//===
{
	file.seekg(0, fstream::end);
	int fileSize = file.tellg();
	file.seekg(0, fstream::beg);
	return fileSize;
}
int hpFindInSource(char target, int startPos, int &length)
{
	int currentPos;
	for(currentPos = startPos; currentPos < csSourceSize; currentPos++)
	{
		if(target == csSource[currentPos])
		{
			length = currentPos - startPos;
			return currentPos;
		}
	}
	length = currentPos - startPos;
	return -1;
}
void hpReadSource(char* path)				//===
{
	//Open file and check for validity of file
	csInput.open(path, fstream::in | fstream::binary);
	if(!csInput.is_open() | !csInput.good())
		throw 1; //Unable to open input file

	//Read all of file and close stream
	csSourceSize = hpFileSizeAndSetBegin(csInput);
	csSource = new char[csSourceSize+1];  //+1 to make space for ToCharArray or the like
	csInput.read(csSource, csSourceSize);		//read all into csSource
	csInput.close();
	
	
	int lineNumber = 0;
	bool inBlockComment = false;

	int startPos = 0;
	int length = 0;
	int lastLineFeedPos = 0;


//Add lines
	do
	{
		lastLineFeedPos = ::hpFindInSource(10, startPos, length);
		//comment check
		for(int i =startPos + 1; i< startPos + length; i++)
		{
			if(inBlockComment)
			{
				if(csSource[i]=='/' && csSource[i-1]=='*')
				{
					inBlockComment = false;
					startPos = i + 1;
					length = lastLineFeedPos - startPos;
					i=startPos;
					continue;
				}
			}
			else
			{
				if(csSource[i]=='/' && csSource[i-1]=='/')
				{
					length = i - 1 - startPos;
					break;
				}
				if(csSource[i]=='*' && csSource[i-1]=='/')
				{
					inBlockComment = true;
					csLines.push_back(Line(SubString(startPos, i - 1 - startPos), lineNumber));
					i++; //jump over a character
					continue;
				}
			}
		}
		//comment check end
		if(!inBlockComment)
			csLines.push_back(Line(SubString(startPos, length), lineNumber));
		startPos = lastLineFeedPos + 1;
		lineNumber++;
	}
	while(lastLineFeedPos!=-1);
}

void hpCheckOutputForReplace(char* path, char* kernelname, char* replacepoint)		//===
{
	//open and check file
	csOutput.open(path, fstream::in | fstream::out | fstream::binary);
	if( !csOutput.is_open() || !csOutput.good() )
		throw 4; //can't open file

	//find file length and read entire file into buffer
	int fileSize = ::hpFileSizeAndSetBegin(csOutput);
	char* buffer = new char[fileSize];
	csOutput.read(buffer, fileSize);
	if(csOutput.bad() || csOutput.fail())
		throw 7; //failed to read cubin

	//Start checking the cubin and looking for sections
	if ( fileSize < 100 || (int)buffer[0] != 0x7f || (int)buffer[1] != 0x45 || (int)buffer[2] != 0x4c || (int)buffer[3] != 0x46) //ELF file identifier
		throw 6; //file to be modified is invalid
	//Reading the ELF header
	unsigned int SHTOffset =	*((unsigned int*)  (buffer+0x20));	//section header table offset, stored at 0x20, length 4
	unsigned int SHsize =		*((unsigned short*)(buffer+0x2e));	//section header size, stored at 0x2e, length 2
	unsigned int SHcount =		*((unsigned short*)(buffer+0x30));	//section header count
	unsigned int SHStrIdx =		*((unsigned short*)(buffer+0x32));	//section header string table index
	unsigned int SHStrOffset =	*((unsigned int*)  (buffer+SHTOffset+SHsize*SHStrIdx+0x10)); //offset in file of the section header string table, 0x10 is the offset in a header for section offset
	int fileoffset = SHTOffset;
	//Going through the section headers to look for the named section
	bool found = false;
	char *sectionname;
	for(unsigned int i =0; i<SHcount; i++, fileoffset+=SHsize)
	{
		unsigned int SHNameIdx =	*((unsigned int*)  (buffer+fileoffset)); //first 4-byte word in a section header is the name index
		sectionname = buffer + SHStrOffset + SHNameIdx;
		if(strcmp(sectionname, kernelname)==0)
		{
			found = true;
			csOutputSectionOffset = *((unsigned int*)  (buffer+fileoffset+0x10));
			csOutputSectionSize   = *((unsigned int*)  (buffer+fileoffset+0x14));
			csOutputInstructionOffset = hpHexCharToInt(replacepoint);
			delete[] buffer;
			return;
		}
	}
	delete[] buffer;
	throw 8;	//section not found
}
//-----End of command-line helper functions
















//9
//-----Debugging functions
void hpPrintLines()
{
	cout<<"Lines:"<<endl;
	for(vector<Line>::iterator i = csLines.begin(); i!=csLines.end(); i++)
	{
		char * result = i->LineString.ToCharArray();
		cout<<result<<endl;
		delete[] result;
	}
}
void hpPrintInstructions()
{
	cout<<"Instructions"<<endl;
	for(vector<Instruction>::iterator i = csInstructions.begin(); i!=csInstructions.end(); i++)
	{
		char * result = i->InstructionString.ToCharArray();
		cout<<result<<endl;
		delete[] result;
	}
}
void hpPrintDirectives()
{
	cout<<"Directives"<<endl;
	for(list<Directive>::iterator i = csDirectives.begin(); i!=csDirectives.end(); i++)
	{
		char * result = i->DirectiveString.ToCharArray();
		cout<<result<<endl;
		delete[] result;
	}
}
	
void hpPrintComponents(Instruction &instruction)
{
	char* line = instruction.InstructionString.ToCharArrayStopOnCR();
	cout<<"-------- Line "<<instruction.LineNumber<<"--------"<<endl;
	cout<<"Content: "<<line<<endl;
	delete[] line;
	int count = 0;
	for(list<SubString>::iterator i = instruction.Components.begin(); i != instruction.Components.end(); i++, count++)
	{
		line = i->ToCharArrayStopOnCR();
		cout<<"Component "<<count<<":"<<line<<"||";
		delete[]line;
		cout<<endl;
	}
}

void hpPrintBinary8(unsigned int word0, unsigned int word1)
{
	char *thisbyte = (char*) &word0;
	for(int i =0; i<4; i++)
	{
		unsigned char index = *(thisbyte++);
		cout<<binaryRef[index %16]<<binaryRef[index/16];
	}
	thisbyte = (char*) &word1;
	for(int i =0; i<4; i++)
	{
		unsigned char index = *(thisbyte++);
		cout<<binaryRef[index %16]<<binaryRef[index/16];
	}
	cout<<endl;
}
//-----End of debugging functions





//Convert binary string often seen on asfermi's site into an unsigned int
void hpBinaryStringToOpcode4(char* string, unsigned int &word0, int &i) //little endian
{
	word0 = 0;
	int counted = 0;
	i=0;
	for(; i<200; i++)
	{
		if(string[i]=='1')
		{
			word0 |=  1<<counted;
			counted++;
			if(counted==32)
				break;
		}
		else if(string[i]=='0')
		{
			counted++;
			if(counted==32)
				break;
		}
	}
	if(i==200)
		throw exception(); //error in binary string
}
void hpBinaryStringToOpcode4(char* string, unsigned int &word0)
{
	int i=0;
	hpBinaryStringToOpcode4(string, word0, i);
}

void hpBinaryStringToOpcode8(char* string, unsigned int &word0, unsigned int &word1)
{
	word0 = 0;
	int i =0;
	hpBinaryStringToOpcode4(string, word0, i);
	hpBinaryStringToOpcode4(string+i+1, word1);
		
}
