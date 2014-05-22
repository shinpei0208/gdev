#include "SubString.h"
extern char* csSource;
extern void hpWarning(int e);

#include "stdafx.h" //Mark


char str_zerosaver;



SubString::SubString(int offset, int length)
{
	Length = length;
	Start = csSource + offset;
#ifdef DebugMode
	if(length<0)
		throw exception();
#endif
}

SubString::SubString(char* target)
{
	Start = target; //In this case Start can point to a location outside of csSource
	Length = strlen(target);
}

char SubString::operator[] (int position)
{
	return Start[position]; //Individual characters of the SubString can be directly accessed using the square bracket notation
}






//	1 ==================================
//-----Start of functions used in parsing



//syntax checking are done in primary substring functions as well as in composite operand processors
//but not in 20-bit functions
	
	
unsigned int SubString::ToImmediate32FromBinary()
{
	unsigned result = 0;
	if(Length<1)
		throw 156; //incorrect binary
	unsigned startPos = 0;
	// binary can start with 'b', or not
	if(Start[0]=='b')
		startPos = 1;
	
	unsigned adder = 1;
	unsigned i;
	for(i=startPos; i<Length; i++, adder<<=1)
	{
		if(Start[i]=='0')
		{}// do nothing
		else if(Start[i]=='1')
			result|=adder;
		else break; //issue: warning?
	}
	// binary digits absent, or longer than 32 digits
	if(adder==1 || i-startPos>32)
		throw 156;
	return result;
}

unsigned int SubString::ToImmediate20FromHexConstant(bool acceptNegative)
{
	unsigned int result = ToImmediate32FromHexConstant(acceptNegative);
	bool negative = (result&0x80000000);
	if(negative)
	{
		if(result<0xFFF80000)
			throw 113;
		result&=0x000FFFFF;
	}
	else if(result>0xFFFFF)
		throw 113;
	return result;
}

unsigned int SubString::ToImmediate20FromIntConstant()
{
	if(Start[0] == 'H' || Start[0] == 'L')
		throw 118; //20-bit cannot contain 64-bit
	unsigned int result = ToImmediate32FromInt32();
	if(result>0xFFFFF)
		throw 113;
	return result;
}

unsigned int SubString::ToImmediate20FromFloatConstant()
{
	if(Length<2 || Start[0] != 'F') //need to check this to ensure access to Start[1] doesn't yield error
		throw 117; //Incorrect floating number
	if(Start[1] == 'H' || Start[1] == 'L')
		throw 118; //20-bit cannot contain 64-bit
	else
		return ToImmediate32FromFloat32() >> 12; //issue: no warning regarding loss of precision
}
	




//Parse the SubString as a hexadecimal expression in the form 0xabcd
//static const unsigned int op_HexRef[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
unsigned int SubString::ToImmediate32FromHexConstant(bool acceptNegative)
{
	if(Length<=2)
		throw 106; //incorrect hex, at least 0x0 need to be present
	int startPos = 0;
	bool negative = false; //if the number starts with '-', needs to do a two's complement later on
	if(acceptNegative && Start[0]=='-') //not all hexadecimal expressions allow the negative sign
	{
		if(Length<=3)
			throw 106;
		negative = true;
		startPos = 1;
	}
	//Check for presence of 0x or 0X
	if(Start[startPos]!='0' || (Start[startPos+1]!='x'&& Start[startPos+1] != 'X'))
		throw 106;
	unsigned int  result = 0;

	int digit;
	int i;
	for(i = startPos + 2; i< Length && i < startPos + 10; i++)
	{
		digit = (int) Start[i];
		
		if(digit<58)
		{
			if(digit<48)
				break; //issue: should issue a warning instead of silently returning
			digit-=48;
		}
		else if(digit<71)
		{
			if(digit<65)
				break;
			digit-=55;
		}
		else if(digit<103)
		{
			if(digit<97)
				break;
			digit -= 87;
		}
		else
			break;
		result <<=4;
		result |= digit;
		//result |= op_HexRef[digit];
	}
	//i has not been incremented, meaning the first character encountered is not hexadecimal digit
	if(i==startPos + 2) 
		throw 106;
	//negative could only be true when acceptNegative is already true
	if(negative) 
	{
		if(result > 0x80000000)
			throw 106;
		result ^= 0xFFFFFFFF;
		result += 1;
	}
	return result;
}

//Parse the SubString as a floating number expression in the form F1234.1234
float ti32_PINF = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000.0;
float ti32_NINF = -1000000000000000000000000000000000000000000000000000000000000000000000000000000000.0;
unsigned int SubString::ToImmediate32FromFloat32()
{
	//At least F0
	if(Length<2 || Start[0]!='F') 
		throw 121; //incorrect floating point number format
	//pad the null at the end
	int zeroPos = Length;
	char zeroSaver = Start[zeroPos];
	Start[zeroPos] = (char)0;

	float fresult = atof(Start+1);
	Start[zeroPos] = zeroSaver;
	if(fresult == 0.0)
		hpWarning(10); //evaluation of constant returned zero
	else if(fresult == ti32_PINF || fresult == ti32_NINF)
		hpWarning(11); //overflow
	return *(unsigned int *)&fresult;
}

//Parse the SubString as a floating number expression in the form FH1234.1234 or FL1234.1234
unsigned int SubString::ToImmediate32FromFloat64()
{
	if(Length<3 || Start[0]!='F' || (Start[1]!='H' && Start[1]!='L') ) //At least FH0/FL0
		throw 121; //incorrect floating point number format

	int zeroPos = Length;
	char zeroSaver = Start[zeroPos];
	Start[zeroPos] = (char)0;

	double dresult = atof(Start+2);
	Start[zeroPos] = zeroSaver;

	if(dresult == 0.0)
		hpWarning(10); //evaluation of constant returned zero. Issue: suggestion of using RZ maybe incorrect.
	else if(dresult == HUGE_VAL || dresult == -HUGE_VAL) //defined in math.h
		hpWarning(11);//overflow
	unsigned int *resultbits = (unsigned int *)&dresult;
	if(Start[1]=='H')
		resultbits++;
	return *resultbits;
}

//Parse as int
unsigned int SubString::ToImmediate32FromInt32()
{
	char zeroSaver = Start[Length];
	Start[Length] = 0;
	int result = atoi(Start);
	Start[Length] = zeroSaver;

	if(result == 0)
		hpWarning(10);
	else if(result == INT_MAX || result == INT_MIN)
		hpWarning(11);
	return *(unsigned int*)&result; //issue: can't deal with large unsigned integers
}
//Parse as integer expression in the form of H1234 or L1234
unsigned int SubString::ToImmediate32FromInt64()
{
	char zeroSaver = Start[Length];
	Start[Length] = 0;
	long result = atol(Start+1); //first character is H or L
	Start[Length] = zeroSaver;

	if(result == 0)
		hpWarning(10);
	else if(result == LONG_MAX || result == LONG_MIN)
		hpWarning(11);

	unsigned int *resultbits = (unsigned int *)&result;
	if(Start[0]=='H')
		resultbits++;
	return *resultbits;
}

//Parse as integer expression. May be in the form 1234 or H1234/L1234
unsigned int SubString::ToImmediate32FromIntConstant()
{
	if(Start[0]=='H' || Start[0]=='L') //long
	{
		return ToImmediate32FromInt64();
	}
	else //Int
	{
		return ToImmediate32FromInt32();
	}
}
//Parse as float expression. may be in the form F1234 or FH1234/FL1234
unsigned int SubString::ToImmediate32FromFloatConstant()
{
	if(Length<2) //F0, ToImmediate32FromDouble assumes length 2 or above
		throw 117; //Incorrect floating number
	if(Start[1] == 'H' || Start[1] == 'L')
		return ToImmediate32FromFloat64();
	else
		return ToImmediate32FromFloat32();
}


//Parse as a simple register expression
int SubString:: ToRegister()
{
	int result;
	if(Length<2 || Start[0]!='R')
		throw 104; //Incorrect register

	//RZ is has a register number of 63
	if(Start[1] == 'Z')
		return 63;
	//first digit of the register number
	result = (int)Start[1] - 48;
	if(result<0 || result>9)
		throw 104;

	//Only R0 to R9
	if(Length==2)
		return result;

	//above R9
	int n2 = (int)Start[2] - 48;
	if(n2<0 || n2>9)
		return result;
	result *= 10;
	result += n2;
	
	//register number too large
	if(result>=63)
		throw 105;
	return result;
}


//Parse SubString as global memory operand in the form [Rxx + 0xabcd]
void SubString::ToGlobalMemory(int &register1, unsigned int&memory)
{
	//incorrect global mem. Shortest expression will be [0]
	if(Length < 3 || Start[0]!='[') 
		throw 107;
	register1 = 63; //default RZ
	memory = 0; //default 0
	
	//skip blank characters after [ first
	int startPos = 1;
	while(startPos<Length)
	{
		if(Start[startPos] > 32)
			break;
		startPos++;
	}
	//Look for '+' to determine if both register and hex value will be present
	int plusPos = Find('+', startPos);
	//Not present, only register expression or hex is used
	if(plusPos==-1)
	{
		if(Start[startPos]=='R')
			register1 = (SubStr(startPos, Length -startPos)).ToRegister();
		else //Issue: what about integer values?
			memory = SubStr(startPos, Length -startPos ).ToImmediate32FromHexConstant(true);
	}
	//both register expression and hex are present
	else
	{
		register1 = SubStr(startPos, Length -startPos).ToRegister();
		startPos = plusPos+1;
		while(startPos<Length)
		{
			if(Start[startPos] > 32)
				break;
			startPos++;
		}
		memory = SubStr(startPos, Length - startPos).ToImmediate32FromHexConstant(true);
	}
}

//parse SubString as a constant memory expression in the form of c[0xa][Rxx+0xb]
void SubString::ToConstantMemory(unsigned int &bank, int &register1, unsigned int &memory, int maxBank)
{
	//shortest expression: c[0x0][0x0]
	if(Length<11|| Start[0]!='c') 
		throw 110; //incorrect constant memory format

	//skip the blanks after the 'c'
	int startPos;
	for(startPos = 1; startPos<Length; startPos++)
	{
		if(Start[startPos]>32)
			break;
	}
	if(startPos==Length || Start[startPos]!='[')
		throw 110;
	startPos++;
	int endPos = Find(']', startPos);
	if(endPos == -1)
		throw 110;
	bank = SubStr(startPos, endPos - startPos).ToImmediate32FromHexConstant(false);
	if(bank > maxBank)
		throw 114; //bank number too large
	startPos = endPos + 1;
	for(; startPos<Length; startPos++)
	{
		if(Start[startPos]>32)
			break;
	}
	if(startPos>=Length || Start[startPos]!='[')
		throw 110;
	SubStr(startPos, Length - startPos).ToGlobalMemory(register1, memory); //issue: negative address offset will not be correctly processed
	if(memory>0xFFFF) 
		throw 111; //too large memory address
}






//	2  =============================
//-----Miscellaneous functions for SubString


//Find
//Look for 'target' in the SubString, starting at startPos
//Returns the index of the first character found
//If target is not found, returns -1
int SubString::Find(char target, int startPos) //startPos is the position in this string
{
	for(int currentPos = startPos; currentPos < Length; currentPos++)
	{
		if(target == Start[currentPos])
			return currentPos;
	}
	return -1;
}


//FindBlank
//Similar to Find, but looks for all characters with ASCII number below 33 (all taken as blank)
int SubString::FindBlank(int startPos)
{
	for(int currentPos = startPos; currentPos < Length; currentPos++)
	{
		if(Start[currentPos] < 33)
			return currentPos;
	}
	return -1;
}

//SubStr
//Extract a SubString from the current SubString
SubString SubString::SubStr(int startPos, int length)
{
	SubString result(startPos + Start - csSource , length);
	return result;
}

//RemoveBlankAtBeginning
//Remove all blanks of the SubSting until the first non-blank character is found
//Note that this does not affect csSource. It only changes the starting position of the SubString and its length
void SubString::RemoveBlankAtBeginning()
{
	int i =0;
	for(; i<Length; i++)
	{
		if(Start[i]>32)
			break;
	}
	Start += i;
	Length -= i;
}
void SubString::RemoveBlankAtEnd()
{
	int i = Length - 1;
	for(; i>0; i--)
	{
		if(Start[i]>32)
			break;
	}
	Length = i + 1;
}


//Compare
//Compare with another SubString. Return true only when both the length as well as all the characters match
bool SubString::Compare(SubString subString)
{
	if(subString.Length != Length)
		return false;
	for(int i =0; i<Length; i++)
	{
		if(Start[i]!= subString[i])
			return false;
	}
	return true;
}

bool SubString::CompareIgnoreEndingBlank(SubString subString)
{
	//using subString as the gold standard
	//so the string in Start could be longer than subString
	if(Length<subString.Length)
		return false;
	int i=0;
	for(; i<subString.Length;i++)
	{
		if(Start[i]!= subString[i])
			return false;
	}
	for(; i<Length; i++)
	{
		if(Start[i]>32)
			return false;
	}
	return true;
}

//CompareWithCharArray
//Compare with a char array
//Do not use this.
bool SubString::CompareWithCharArray(char* target, int length) //length is the length of the char
{
	if(length < Length)
		return false;
	for(int i =0; i<length; i++)
	{
		if(Start[i]!=target[i])
			return false;
	}
	return true;
}


//Auxiliary. Not to be used
char* SubString::ToCharArray()
{
	char *result = new char[Length + 1];
	for(int i =0 ; i<Length; i++)
		result[i] = Start[i];
	result[Length] = (char)0;
	return result;
}


//SubEndWithNull
//asfermi now uses atoi, atol and atof to convert numerical constant expressions to 
//the corresponding number types. atoi and alike recognise null-terminated char arrays
//So this function replaces the end of the SubString with null
void SubString::SubEndWithNull()
{
	str_zerosaver = Start[Length];
	Start[Length] = 0;
}
//And this function recovers the null to its original value
void SubString::RecoverEndWithNull()
{
	Start[Length] = str_zerosaver;
}



char* SubString::ToCharArrayStopOnCR()
{
		char *result = new char[Length + 1];
		for(int i =0 ; i<Length; i++)
		{
			if((int)Start[i]==13) //carriage return
			{
				result[i] = (char)0;
				return result;
			}
			result[i] = Start[i];
		}
		result[Length] = (char)0;
		return result;
}


//======================



	SortElement::SortElement(void *extraInfo, SubString name)
	{
		ExtraInfo = extraInfo;
		Name = name;
	}

unsigned int SortComputeIndex(SubString subString)
{
	unsigned int result=0;
	if(subString.Length>=2)
	{
		result=((unsigned int)subString[0]+(unsigned int)subString[1])<<24;
		result += (unsigned int)subString[subString.Length-2];
		if(subString.Length>=4)
		{
			result+=((unsigned int)subString[2]+(unsigned int)subString[3])<<16;
			if(subString.Length>=8)
			{
				result+=((unsigned int)subString[6]+(unsigned int)subString[7])<<8;
			}
		}
	}
	result+=(unsigned int)subString[subString.Length-1];
	return result;

}
void SortInitialize(list<SortElement>elementList, SortElement* &sortedList, unsigned int* &indicesList)
{
	int count = elementList.size();
	sortedList = new SortElement[count];
	indicesList = new unsigned int[count];
	int i =0;
	//copy
	for(list<SortElement>::iterator element = elementList.begin(); element!= elementList.end(); element++, i++)
	{
		sortedList[i] = *element;
		indicesList[i] = SortComputeIndex(element->Name);
	}
	//sort
	SortElement typeSaver;
	unsigned int indexsaver;
	for(int i = count - 1; i >  0; i--)
	{
		for(int j =0; j< i; j++)
		{
			//larger one behind
			if(indicesList[j] > indicesList[j+1])
			{
				typeSaver = sortedList[j];
				indexsaver = indicesList[j];
				sortedList[j] = sortedList[j+1];
				indicesList[j] = indicesList[j+1];
				sortedList[j+1] = typeSaver;
				indicesList[j+1] = indexsaver;
			}
		}
	}	
}

SortElement SortNotFound((void*)0xFFFFFFFF,"");
SortElement SortFind(SortElement* sortedList, unsigned int* indicesList, unsigned int count, SubString target)
{
	unsigned int index = SortComputeIndex(target);
	int start = 0; //inclusive
	int end = count;//exclusive
	int mid;
	while(start<end) //still got unchecked numbers
	{
		mid = (start+end)/2;
		if(index > indicesList[mid])
			start = mid + 1;
		else if(index < indicesList[mid])
			end = mid;
		else
		{
			//backtrack
			while(mid!=0&&indicesList[mid-1]==index)
			{
				mid--;
			}
			//forward checking
			do
			{
				if(target.Compare(sortedList[mid].Name))
					return sortedList[mid];
				mid++;
			}
			while(mid!=count&&indicesList[mid]==index);
			return SortNotFound;
		}
	}
	return SortNotFound;
}
