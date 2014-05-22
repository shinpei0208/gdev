
#include "Cubin.h"
#include "SubString.h"

#include "stdafx.h" //Mark


const unsigned int ELFFlagsForsm_20 = 0x00140114;
const unsigned int ELFFlagsForsm_21 = 0x00140115;
const unsigned int ELFFlagsForsm_30 = 0x001E011E;


//default is 64-bit
int ELFHeaderSize = 0x40;
int ELFSectionHeaderSize=0x40;
int ELFSegmentHeaderSize=0x38;
int ELFSymbolEntrySize=0x18;

//ELFHeader32
ELFHeader32::ELFHeader32()
{
	//0x00
	Byte0 = 0x7f;
	Byte1 = 'E';
	Byte2 = 'L';
	Byte3 = 'F';
	//0x04
	FileClass = 1; //1 is 32-bit
	Encoding = 1; //LSB
	FileVersion = 1; //1 for current				
	//0x07
	memset(&Padding, 0, 9);
	Padding[0] = 0x33; //issue: same for all? any 
	Padding[1] = 0x04;

	//0x10
	FileType = 0x0002;
	Machine = 0x00BE;								
		
	//0x14
	Version = 1;
	EntryPoint = 0;
	//PHTOffset not set
	SHTOffset = 0x34;

	//0x24
	Flags = 0x00140114; //default sm_20
	//00aabbcc
	//aa: ptx target architecture
	//bb: ptx address size 01: 32-bit, 05:64-bit
	//cc: architecture: 14: sm_20, 15: sm_21, 1E: sm_30
		
	//0x28
	HeaderSize = 0x34;
	PHSize = 0x20;
	//PHCount not set
	SHSize = 0x28;
	//SHCount not set;
	SHStrIdx = 1;									//0x34
}
ELFHeader32 ELFH32;


//ELFSection
ELFSection::ELFSection()
{
	SectionContent = 0;
}

//ELFSymbolEntry
void ELFSymbolEntry::Reset()
{
	Name = 0;
	Value = 0;
	Size = 0;
	Info = 0;
	Other = 0;
	SHIndex = 0;
}


//Kernel
void Kernel::Reset()
{
	KernelName.Start = 0;
	KernelName.Length = 0;

	TextSize = 0;
	SharedSize = 0;
	LocalSize = 0;
	BarCount = 0;
	RegCount = 0;

	StrTabOffset = 0;
	MinStackSize = 0;
	MinFrameSize = 0;
	GlobalSymbolIndex;

	ParamTotalSize = 0;
	Parameters.clear(); //issue: would this affect the saved kernel?
	KernelInstructions.clear();
}
