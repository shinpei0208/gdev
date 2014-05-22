/*
This file contains various helper functions used during cubin output (the post-processin stage)
1: Stage1 functions
2: Stage2 functions
3: Stage3 functions
4: Stage4 and later functions

all functions are prefixed with 'hpCubin'
*/

#ifndef helperCubinDefined //prevent multiple inclusion
#define helperCubinDefined
//---code starts ---
#include "../Cubin.h"

void hpCubinSet64(bool is64);

//	1
//-----Stage1 functions
//SectionIndex, SectionSize, OffsetFromFirst, SHStrTabOffset 
void hpCubinStage1SetSection(ELFSection &section, SectionType nvType, unsigned int kernelNameLength);
void hpCubinStage1();

//	2
//-----Stage2 functions
inline void hpCubinAddSectionName1(unsigned char* sectionContent, int &offset, char* sectionPrefix, SubString &kernelName);
inline void hpCubinAddSectionName2(unsigned char* sectionContent, int &offset, char* sectionName);
inline void hpCubinAddSectionName3(unsigned char* sectionContent, int &offset, SubString &kernelName);
inline void hpCubinStage2SetSHStrTabSectionContent();
inline void hpCubinStage2SetStrTabSectionContent();
inline void hpCubinStage2AddSectionSymbol(ELFSection &section, ELFSymbolEntry &entry, int &offset, unsigned int size);
inline void hpCubinStage2SetSymTabSectionContent();
void hpCubinStage2();

//	3
//-----Stage3
void hpCubinStage3();

//	4
//-----Stage4 and later functions
void hpCubinSetELFSectionHeader1(ELFSection &section, unsigned int type, unsigned int alignment, unsigned int &offset);

//Stage4: Setup all section headers
void hpCubinStage4();

//Stage5: Setup all program segments
void hpCubinStage5();

//Stage6: Setup ELF header
void hpCubinStage6();

//Stage7: Write to cubin
void hpCubinStage7();
//-----End of cubin helper functions
#else
#endif