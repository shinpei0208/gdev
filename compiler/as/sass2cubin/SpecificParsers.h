#ifndef SpecificParsersDefined
#define SpecificParsersDefined

#include "SubString.h"
#include "DataTypes.h"

//	1
//-----Declaration of default parsers: DefaultMasterParser, DefaultLineParser, DefaultInstructionParser, DefaultDirectiveParser
struct MasterParserDefault;
extern MasterParserDefault MPDefault;

struct LineParserDefault;
extern LineParserDefault LPDefault;

struct InstructionParserDefault;
extern InstructionParserDefault IPDefault;

struct DirectiveParserDefault;
extern DirectiveParserDefault DPDefault;
//-----End of default parser declarations


struct LineParserConstant2;
extern LineParserConstant2 LPConstant2;
void Constant2ParseInt(SubString &content);
void Constant2ParseLong(SubString &content);
void Constant2ParseFloat(SubString &content);
void Constant2ParseDouble(SubString &content);
void Constant2ParseMixed(SubString &content);

#else
#endif