/*! \file ptx1_4grammar.ypp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Monday June 15, 2009
	\brief The YACC grammar file for PTX
*/

%{
	#include <iostream>
	#include <ocelot/parser/interface/PTXParser.h>
	#include <ocelot/parser/interface/PTXLexer.h>
	#include <hydrazine/interface/debug.h>
	#include <cassert>
	#include <cstring>

	#define YYERROR_VERBOSE 1

	#ifdef REPORT_BASE
	#undef REPORT_BASE
	#endif

	#define REPORT_BASE 0

	namespace ptx
	{
	
	int yylex( YYSTYPE* token, YYLTYPE* location, parser::PTXLexer& lexer, 
		parser::PTXParser::State& state );
	void yyerror( YYLTYPE* location, parser::PTXLexer& lexer, 
		parser::PTXParser::State& state, char const* message );
	
	std::string yyTypeToString( int );
	
%}

%union
{
	char text[1024];
	long long int value;
	long long unsigned int uvalue;

	double doubleFloat;
	float singleFloat;
}

%parse-param {parser::PTXLexer& lexer}
%parse-param {parser::PTXParser::State& state}
%lex-param   {parser::PTXLexer& lexer}
%lex-param   {parser::PTXParser::State& state}
%pure-parser

%token<text> TOKEN_LABEL TOKEN_IDENTIFIER TOKEN_STRING TOKEN_METADATA
%token<text> TOKEN_INV_PREDICATE_IDENTIFIER TOKEN_PREDICATE_IDENTIFIER

%token<text> OPCODE_COPYSIGN OPCODE_COS OPCODE_SQRT OPCODE_ADD OPCODE_RSQRT
%token<text> OPCODE_MUL OPCODE_SAD OPCODE_SUB OPCODE_EX2 OPCODE_LG2 OPCODE_ADDC
%token<text> OPCODE_RCP OPCODE_SIN OPCODE_REM OPCODE_MUL24 OPCODE_MAD24
%token<text> OPCODE_DIV OPCODE_ABS OPCODE_NEG OPCODE_MIN OPCODE_MAX
%token<text> OPCODE_MAD OPCODE_MADC OPCODE_SET OPCODE_SETP OPCODE_SELP 
%token<text> OPCODE_SLCT OPCODE_MOV OPCODE_ST OPCODE_CVT OPCODE_AND OPCODE_XOR 
%token<text> OPCODE_OR OPCODE_CVTA OPCODE_ISSPACEP OPCODE_LDU
%token<text> OPCODE_SULD OPCODE_TXQ OPCODE_SUST OPCODE_SURED OPCODE_SUQ
%token<text> OPCODE_BRA OPCODE_CALL OPCODE_RET OPCODE_EXIT OPCODE_TRAP 
%token<text> OPCODE_BRKPT OPCODE_SUBC OPCODE_TEX OPCODE_LD OPCODE_BARSYNC
%token<text> OPCODE_ATOM OPCODE_RED OPCODE_NOT OPCODE_CNOT OPCODE_VOTE
%token<text> OPCODE_SHR OPCODE_SHL OPCODE_FMA OPCODE_MEMBAR OPCODE_PMEVENT
%token<text> OPCODE_POPC OPCODE_PRMT OPCODE_CLZ OPCODE_BFIND OPCODE_BREV 
%token<text> OPCODE_BFI OPCODE_BFE OPCODE_TESTP OPCODE_TLD4 OPCODE_BAR
%token<text> OPCODE_PREFETCH OPCODE_PREFETCHU OPCODE_SHFL

%token<value> PREPROCESSOR_INCLUDE PREPROCESSOR_DEFINE PREPROCESSOR_IF 
%token<value> PREPROCESSOR_IFDEF PREPROCESSOR_ELSE PREPROCESSOR_ENDIF 
%token<value> PREPROCESSOR_LINE PREPROCESSOR_FILE

%token<value> TOKEN_ENTRY TOKEN_EXTERN TOKEN_FILE TOKEN_VISIBLE TOKEN_LOC
%token<value> TOKEN_FUNCTION TOKEN_STRUCT TOKEN_UNION TOKEN_TARGET TOKEN_VERSION
%token<value> TOKEN_SECTION TOKEN_ADDRESS_SIZE TOKEN_WEAK

%token<value> TOKEN_MAXNREG TOKEN_MAXNTID TOKEN_MAXNCTAPERSM TOKEN_MINNCTAPERSM 
%token<value> TOKEN_SM11 TOKEN_SM12 TOKEN_SM13 TOKEN_SM20 TOKEN_MAP_F64_TO_F32
%token<value> TOKEN_SM21 TOKEN_SM10 TOKEN_SM30 TOKEN_SM35
%token<value> TOKEN_TEXMODE_INDEPENDENT TOKEN_TEXMODE_UNIFIED

%token<value> TOKEN_CONST TOKEN_GLOBAL TOKEN_LOCAL TOKEN_PARAM TOKEN_PRAGMA TOKEN_PTR
%token<value> TOKEN_REG TOKEN_SHARED TOKEN_TEXREF TOKEN_CTA TOKEN_SURFREF 
%token<value> TOKEN_GL TOKEN_SYS TOKEN_SAMPLERREF

%token<value> TOKEN_U32 TOKEN_S32 TOKEN_S8 TOKEN_S16 TOKEN_S64 TOKEN_U8 
%token<value> TOKEN_U16 TOKEN_U64 TOKEN_B8 TOKEN_B16 TOKEN_B32 TOKEN_B64 
%token<value> TOKEN_F16 TOKEN_F64 TOKEN_F32 TOKEN_PRED

%token<value> TOKEN_EQ TOKEN_NE TOKEN_LT TOKEN_LE TOKEN_GT TOKEN_GE
%token<value> TOKEN_LS TOKEN_HS TOKEN_EQU TOKEN_NEU TOKEN_LTU TOKEN_LEU
%token<value> TOKEN_GTU TOKEN_GEU TOKEN_NUM TOKEN_NAN

%token<value> TOKEN_HI TOKEN_LO TOKEN_AND TOKEN_OR TOKEN_XOR
%token<value> TOKEN_RN TOKEN_RM TOKEN_RZ TOKEN_RP TOKEN_SAT TOKEN_VOLATILE
%token<value> TOKEN_TAIL TOKEN_UNI TOKEN_ALIGN TOKEN_BYTE TOKEN_WIDE TOKEN_CARRY
%token<value> TOKEN_RNI TOKEN_RMI TOKEN_RZI TOKEN_RPI
%token<value> TOKEN_FTZ TOKEN_APPROX TOKEN_FULL TOKEN_SHIFT_AMOUNT
%token<value> TOKEN_R TOKEN_G TOKEN_B TOKEN_A

%token<value> TOKEN_TO

%token<value> TOKEN_CALL_PROTOTYPE TOKEN_CALL_TARGETS

%token<value> TOKEN_V2 TOKEN_V4
%token<value> TOKEN_X TOKEN_Y TOKEN_Z TOKEN_W

%token<value> TOKEN_ANY TOKEN_ALL
%token<value> TOKEN_UP TOKEN_DOWN TOKEN_BFLY TOKEN_IDX

%token<value> TOKEN_MIN TOKEN_MAX TOKEN_DEC TOKEN_INC TOKEN_ADD TOKEN_CAS
%token<value> TOKEN_EXCH

%token<value> TOKEN_1D TOKEN_2D TOKEN_3D TOKEN_A1D TOKEN_A2D TOKEN_CUBE TOKEN_ACUBE

%token<value> TOKEN_CA TOKEN_WB TOKEN_CG TOKEN_CS TOKEN_LU TOKEN_CV TOKEN_WT TOKEN_NC

%token<value> TOKEN_L1 TOKEN_L2

%token<value> TOKEN_P

%token<value> TOKEN_WIDTH TOKEN_DEPTH TOKEN_HEIGHT TOKEN_NORMALIZED_COORDS
%token<value> TOKEN_FILTER_MODE TOKEN_ADDR_MODE_0 TOKEN_ADDR_MODE_1
%token<value> TOKEN_ADDR_MODE_2
%token<value> TOKEN_CHANNEL_DATA_TYPE TOKEN_CHANNEL_ORDER

%token<value> TOKEN_TRAP TOKEN_CLAMP TOKEN_ZERO

%token<value> TOKEN_ARRIVE TOKEN_RED TOKEN_POPC TOKEN_SYNC

%token<value> TOKEN_BALLOT

%token<value> TOKEN_F4E TOKEN_B4E TOKEN_RC8 TOKEN_ECL TOKEN_ECR TOKEN_RC16

%token<value> TOKEN_FINITE TOKEN_INFINITE TOKEN_NUMBER TOKEN_NOT_A_NUMBER
%token<value> TOKEN_NORMAL TOKEN_SUBNORMAL

%token<value> TOKEN_DECIMAL_CONSTANT

%token<uvalue> TOKEN_UNSIGNED_DECIMAL_CONSTANT

%token<singleFloat> TOKEN_SINGLE_CONSTANT

%token<doubleFloat> TOKEN_DOUBLE_CONSTANT

%start statements

%%

nonEntryStatements : version | target | registerDeclaration | fileDeclaration 
	| preprocessor | samplerDeclaration | surfaceDeclaration | textureDeclaration 
	| globalSharedDeclaration | globalLocalDeclaration | addressSize;

nonEntryStatement : nonEntryStatements
{
	state.statementEnd( @1 );
};

statement : initializableDeclaration | nonEntryStatement | entry | functionBody
	| functionDeclaration;

statements : statement | statements statement;

preprocessorCommand : PREPROCESSOR_INCLUDE | PREPROCESSOR_DEFINE 
	| PREPROCESSOR_IF | PREPROCESSOR_IFDEF | PREPROCESSOR_ELSE 
	| PREPROCESSOR_ENDIF | PREPROCESSOR_LINE | PREPROCESSOR_FILE;

preprocessor : preprocessorCommand
{
	state.preprocessor( $<value>1 );
};

version : TOKEN_VERSION TOKEN_DOUBLE_CONSTANT 
{ 
	state.version( $<doubleFloat>2, @2 );
};

identifier : '_' | TOKEN_IDENTIFIER | opcode;
optionalIdentifier : /* empty string */ | identifier;

identifierList : identifier
{
	state.identifierList( $<text>1 );
};

identifierList : identifierList ',' identifier
{
	state.identifierList2( $<text>3 );
};

decimalListSingle : identifier
{
	state.decimalListSingle( 0 );
	state.symbolListSingle( $<text>1 );
};

decimalListSingle : decimalListSingle ',' identifier
{
	state.decimalListSingle2( 0 );
	state.symbolListSingle2( $<text>3 );
};

decimalListSingle : TOKEN_DECIMAL_CONSTANT
{
	state.decimalListSingle( $<value>1 );
};

optionalMetadata : /* empty string */
{
    state.metadata("");
};

optionalMetadata : TOKEN_METADATA
{
    state.metadata( $<text>1 );
};

decimalListSingle : decimalListSingle ',' TOKEN_DECIMAL_CONSTANT
{
	state.decimalListSingle2( $<value>3 );
};

decimalList : '{' decimalList '}' ',' '{' decimalList '}';
decimalList : '{' decimalListSingle '}' ',' '{' decimalListSingle '}';

decimalInitializer : decimalList | '{' decimalList '}' | 
	'{' decimalListSingle '}' | decimalListSingle;

floatListSingle : TOKEN_DOUBLE_CONSTANT
{
	state.floatList( $<doubleFloat>1 );
};

floatListSingle : floatListSingle ',' TOKEN_DOUBLE_CONSTANT
{
	state.floatList1( $<doubleFloat>3 );
};

floatList : '{' floatList '}' ',' '{' floatList '}';
floatList : '{' floatListSingle '}' ',' '{' floatListSingle '}';

floatInitializer : floatList |  '{' floatList '}' | '{' floatListSingle '}' 
	| floatListSingle;

singleListSingle : TOKEN_SINGLE_CONSTANT
{
	state.singleList( $<singleFloat>1 );
};

singleListSingle : singleListSingle ',' TOKEN_SINGLE_CONSTANT
{
	state.singleList1( $<singleFloat>3 );
};

singleList : '{' singleList '}' ',' '{' singleList '}';
singleList : '{' singleListSingle '}' ',' '{' singleListSingle '}';

singleInitializer : singleList |  '{' singleList '}' | '{' singleListSingle '}' 
	| singleListSingle;

shaderModel : TOKEN_SM10 | TOKEN_SM11 | TOKEN_SM12 | TOKEN_SM13 | TOKEN_SM20
	| TOKEN_SM21 | TOKEN_SM30 | TOKEN_SM35;
	
floatingPointOption : TOKEN_MAP_F64_TO_F32;
textureOption: TOKEN_TEXMODE_INDEPENDENT | TOKEN_TEXMODE_UNIFIED;

targetOption : shaderModel | floatingPointOption | textureOption;
targetElement : targetOption
{
	state.targetElement( $<value>1 );
};

targetElementList : /* empty string */ | targetElement 
	| targetElementList ',' targetElement;

target : TOKEN_TARGET targetElementList
{
	state.target();
};

addressSize : TOKEN_ADDRESS_SIZE TOKEN_DECIMAL_CONSTANT
{
	state.addressSize( $<value>2 );
};

addressSpaceIdentifier : TOKEN_CONST | TOKEN_GLOBAL | TOKEN_LOCAL
	| TOKEN_PARAM | TOKEN_SHARED;

addressSpace : addressSpaceIdentifier
{
	state.addressSpace( $<value>1 );
};

optionalAddressSpace : addressSpace;
optionalAddressSpace : /* empty string */
{
	state.noAddressSpace();
};

pointerDataTypeId: TOKEN_U64 | TOKEN_U32;

dataTypeId : TOKEN_U8 | TOKEN_U16 | TOKEN_U32 | TOKEN_U64 | TOKEN_S8 
	| TOKEN_S16 | TOKEN_S32 | TOKEN_S64 | TOKEN_B8 | TOKEN_B16 | TOKEN_B32 
	| TOKEN_B64 | TOKEN_F16 | TOKEN_F32 | TOKEN_F64 | TOKEN_PRED;

dataType : dataTypeId
{
	state.dataType( $<value>1 );
};

pointerDataType : pointerDataTypeId
{
	state.dataType( $<value>1 );
};

vectorToken : TOKEN_V2 | TOKEN_V4;

statementVectorType : vectorToken
{
	state.statementVectorType( $<value>1 );
};

instructionVectorType : vectorToken
{
	state.instructionVectorType( $<value>1 );
};

optionalInstructionVectorType : instructionVectorType;
optionalInstructionVectorType : /* empty string */;

alignment : TOKEN_ALIGN TOKEN_DECIMAL_CONSTANT { state.alignment = $<value>2; };

kernelParameterPtrSpace: TOKEN_GLOBAL | TOKEN_CONST | TOKEN_LOCAL | TOKEN_SHARED;
parameterAttribute: TOKEN_PTR kernelParameterPtrSpace
{
	state.paramArgumentDeclaration($<value>2);
}

addressableVariablePrefix : dataType statementVectorType {state.alignment = 1;};
addressableVariablePrefix : statementVectorType dataType {state.alignment = 1;};
addressableVariablePrefix : dataType { state.alignment = 1; };
addressableVariablePrefix : alignment dataType statementVectorType;
addressableVariablePrefix : alignment statementVectorType dataType;
addressableVariablePrefix : dataType alignment statementVectorType;
addressableVariablePrefix : dataType statementVectorType alignment;
addressableVariablePrefix : statementVectorType dataType alignment;
addressableVariablePrefix : statementVectorType alignment dataType;
addressableVariablePrefix : dataType alignment;
addressableVariablePrefix : alignment dataType;
addressableVariablePrefix : dataType parameterAttribute alignment;

arrayDimensionSet : '[' TOKEN_DECIMAL_CONSTANT ']'
{
	state.arrayDimensionSet( $<value>2, @2, false );
};

arrayDimensionSet : arrayDimensionSet '[' TOKEN_DECIMAL_CONSTANT ']'
{
	state.arrayDimensionSet( $<value>3, @3, true );
};

arrayDimensionSet : '[' ']'
{
	state.arrayDimensionSet( );
};

arrayDimensions : /* empty string */
{
	state.arrayDimensions();
};

arrayDimensions : arrayDimensionSet;

initializer : /* empty string */;

assignment : '='
{
	state.assignment();
};

initializer : assignment decimalInitializer | assignment floatInitializer 
	| assignment singleInitializer;

registerIdentifierList : identifier
{
	state.registerDeclaration( $<text>1, @1 );
};

registerIdentifierList : identifier '<' TOKEN_DECIMAL_CONSTANT '>'
{
	state.registerDeclaration( $<text>1, @1, $<value>3 );
};

registerSeperator : ','
{
	state.registerSeperator( @1 );
};

registerIdentifierList : registerIdentifierList registerSeperator identifier
{
	state.registerDeclaration( $<text>3, @3 );
};

registerPrefix : statementVectorType dataType;
registerPrefix : dataType;

registerDeclaration : TOKEN_REG registerPrefix registerIdentifierList ';';

optionalTimestampAndSize : /* empty string */;
optionalTimestampAndSize : ',' TOKEN_DECIMAL_CONSTANT
	',' TOKEN_DECIMAL_CONSTANT;

fileDeclaration : TOKEN_FILE TOKEN_DECIMAL_CONSTANT TOKEN_STRING
	optionalTimestampAndSize
{
	state.fileDeclaration( $<value>2, $<text>3 );
};

globalSharedDeclaration : externOrVisible TOKEN_SHARED 
	addressableVariablePrefix identifier arrayDimensions ';'
{
	state.locationAddress( $<value>2 );
	state.initializableDeclaration( $<text>4, @4, @6 );
};

initializableDeclaration : initializable addressableVariablePrefix 
	identifier arrayDimensions initializer ';'
{
	state.initializableDeclaration( $<text>3, @3, @5 );
	state.statementEnd( @3 );
};

globalLocalDeclaration:  externOrVisible TOKEN_LOCAL 
	addressableVariablePrefix identifier arrayDimensions ';'
{
	state.locationAddress( $<value>2 );
	state.initializableDeclaration( $<text>4, @4, @6 );
}

textureSpace : TOKEN_PARAM | TOKEN_GLOBAL;

textureDeclaration : externOrVisible textureSpace TOKEN_TEXREF identifier ';'
{
	state.textureDeclaration( $<value>2, $<text>4, @1 );
};

samplerDeclaration : externOrVisible textureSpace TOKEN_SAMPLERREF identifier ';'
{
	state.samplerDeclaration( $<value>2, $<text>4, @1 );
};

surfaceDeclaration : externOrVisible textureSpace TOKEN_SURFREF identifier ';'
{
	state.surfaceDeclaration( $<value>2, $<text>4, @1 );
};

parameter : TOKEN_PARAM
{
	state.locationAddress( $<value>1 );
};

parameter : TOKEN_REG
{
	state.locationAddress( $<value>1 );
};

argumentDeclaration : parameter addressableVariablePrefix identifier 
	arrayDimensions
{
	state.attribute( false, false, false );
	state.argumentDeclaration( $<text>3, @1 );
};

returnArgumentListBegin : '('
{
	state.returnArgumentListBegin( @1 );
};

returnArgumentListEnd : ')'
{
	state.returnArgumentListEnd( @1 );
};

argumentListBegin : '('
{
	state.argumentListBegin( @1 );
};

argumentListEnd : ')'
{
	state.argumentListEnd( @1 );
};

openBrace : '{'
{
	state.openBrace( @1 );
};

closeBrace : '}' optionalMetadata
{
	state.closeBrace( @1 );
};

argumentListBody : argumentDeclaration;
argumentListBody : /* empty string */;
argumentListBody : argumentListBody ',' argumentDeclaration;

returnArgumentList : returnArgumentListBegin argumentListBody 
	returnArgumentListEnd;
argumentList : argumentListBegin argumentListBody argumentListEnd;

optionalReturnArgumentList : returnArgumentList | /* empty string */;

functionBegin : TOKEN_FUNCTION
{
	state.functionBegin( @1 );
};

functionName : identifier
{
	state.functionName( $<text>1, @1 );
};

optionalSemicolon: ';';
optionalSemicolon: /* empty string */;

functionDeclaration : externOrVisible functionBegin optionalReturnArgumentList 
	functionName argumentList optionalSemicolon
{
	state.functionDeclaration( @4, false );
};

functionBodyDefinition : externOrVisible functionBegin 
	optionalReturnArgumentList functionName argumentList
{
	state.functionDeclaration( @4, true );
};

functionBody : functionBodyDefinition openBrace entryStatements closeBrace;

entryName : externOrVisible TOKEN_ENTRY identifier
{
	state.entry( $<text>3, @1 );
};

optionalArgumentList : argumentList;
optionalArgumentList : /* empty string */;

entryDeclaration : entryName optionalArgumentList performanceDirectives
{
	state.entryDeclaration( @1 );
};

entry : entryDeclaration openBrace entryStatements closeBrace;

entry : entryDeclaration openBrace closeBrace;

entry : entryDeclaration ';'
{
	state.entryPrototype( @1 );
};

entryStatement : registerDeclaration | location | label | pragma 
	| callprototype | calltargets;

completeEntryStatement : uninitializableDeclaration;

completeEntryStatement : entryStatement
{
	state.statementEnd( @1 );
};

completeEntryStatement : guard instruction optionalMetadata
{
	state.entryStatement( @2 );
	state.instruction();
};

completeEntryStatement : openBrace entryStatements closeBrace;

entryStatements : completeEntryStatement;
entryStatements : entryStatements completeEntryStatement;

maxnreg : TOKEN_MAXNREG TOKEN_DECIMAL_CONSTANT
{
	state.maxnreg( $<value>2 );
};

maxntid : TOKEN_MAXNTID TOKEN_DECIMAL_CONSTANT
{
	state.maxntid( $<value>2 );
};

maxntid : TOKEN_MAXNTID TOKEN_DECIMAL_CONSTANT ',' TOKEN_DECIMAL_CONSTANT
{
	state.maxntid( $<value>2, $<value>4 );
};

maxntid : TOKEN_MAXNTID TOKEN_DECIMAL_CONSTANT ',' TOKEN_DECIMAL_CONSTANT ',' 
	TOKEN_DECIMAL_CONSTANT
{
	state.maxntid( $<value>2, $<value>4, $<value>6 );
};

ctapersm : shaderModel ':' TOKEN_DECIMAL_CONSTANT
{
	state.ctapersm( $<value>1, $<value>3 );
};

ctapersmList : ctapersm | ctapersmList ',' ctapersm;

minnctapersm : TOKEN_MINNCTAPERSM TOKEN_DECIMAL_CONSTANT
{
	state.minnctapersm( $<value>2 );
};

minnctapersm : TOKEN_MINNCTAPERSM ctapersmList
{
	state.minnctapersm( );
};

maxnctapersm : TOKEN_MAXNCTAPERSM TOKEN_DECIMAL_CONSTANT
{
	state.maxnctapersm( $<value>2 );
};

maxnctapersm : TOKEN_MAXNCTAPERSM ctapersmList
{
	state.maxnctapersm();
};

performanceDirective : maxnreg | maxntid | maxnctapersm | minnctapersm;
performanceDirectiveList : performanceDirective
	| performanceDirectiveList performanceDirective;
performanceDirectives : /* empty string */ | performanceDirectiveList;

externOrVisible : TOKEN_WEAK
{
	state.attribute( false, false, true );
};

externOrVisible : TOKEN_EXTERN
{
	state.attribute( false, true, false );
};

externOrVisible : TOKEN_VISIBLE
{
	state.attribute( true, false, false );
};

externOrVisible : /* empty string */
{
	state.attribute( false, false, false );
};

uninitializableAddress : TOKEN_LOCAL | TOKEN_SHARED | TOKEN_PARAM;
initializableAddress : TOKEN_CONST | TOKEN_GLOBAL;

uninitializable : externOrVisible uninitializableAddress
{
	state.locationAddress( $<value>2 );
};

initializable : externOrVisible initializableAddress
{
	state.locationAddress( $<value>2 );
};

opcode : OPCODE_COS | OPCODE_SQRT | OPCODE_ADD | OPCODE_RSQRT | OPCODE_ADDC
	| OPCODE_MUL | OPCODE_SAD | OPCODE_SUB | OPCODE_EX2 | OPCODE_LG2
	| OPCODE_RCP | OPCODE_SIN | OPCODE_REM | OPCODE_MUL24 | OPCODE_MAD24
	| OPCODE_DIV | OPCODE_ABS | OPCODE_NEG | OPCODE_MIN | OPCODE_MAX
	| OPCODE_MAD | OPCODE_MADC | OPCODE_SET | OPCODE_SETP | OPCODE_SELP
	| OPCODE_SLCT | OPCODE_MOV | OPCODE_ST | OPCODE_COPYSIGN | OPCODE_SHFL
	| OPCODE_CVT | OPCODE_CVTA | OPCODE_ISSPACEP 
	| OPCODE_AND | OPCODE_XOR | OPCODE_OR
	| OPCODE_BRA | OPCODE_CALL | OPCODE_RET | OPCODE_EXIT | OPCODE_TRAP 
	| OPCODE_BRKPT | OPCODE_SUBC | OPCODE_TEX | OPCODE_LD | OPCODE_LDU
	| OPCODE_BARSYNC | OPCODE_SULD | OPCODE_TXQ | OPCODE_SUST | OPCODE_SURED 
	| OPCODE_SUQ | OPCODE_ATOM | OPCODE_RED | OPCODE_NOT | OPCODE_CNOT
	| OPCODE_VOTE | OPCODE_SHR | OPCODE_SHL | OPCODE_MEMBAR | OPCODE_FMA
	| OPCODE_PMEVENT | OPCODE_POPC | OPCODE_CLZ | OPCODE_BFIND | OPCODE_BREV
	| OPCODE_BFI | OPCODE_TESTP | OPCODE_TLD4
	| OPCODE_PREFETCH | OPCODE_PREFETCHU;

uninitializableDeclaration : uninitializable addressableVariablePrefix 
	identifier arrayDimensions ';'
{
	state.uninitializableDeclaration( $<text>3 );
	state.statementEnd( @2 );
};

location : TOKEN_LOC TOKEN_DECIMAL_CONSTANT TOKEN_DECIMAL_CONSTANT 
	TOKEN_DECIMAL_CONSTANT
{
	state.location( $<value>2, $<value>3, $<value>4 );
};

label : TOKEN_LABEL optionalMetadata
{
	state.label( $<text>1 );
};

labelOperand : identifier
{
	state.labelOperand( $<text>1 );
};

returnType : parameter dataTypeId optionalIdentifier
{
	state.returnType( $<value>2 );
};

returnTypeListBody : returnType;
returnTypeListBody : returnTypeListBody ',' returnType;
returnTypeList : '(' returnTypeListBody ')' | '(' ')' | /* empty string */;

optionalAlignment : /* empty string */ | alignment;

argumentType : parameter optionalAlignment dataTypeId
	optionalIdentifier arrayDimensions
{
	state.argumentType( $<value>3 );
};

argumentTypeListBody : argumentType;
argumentTypeListBody : argumentTypeListBody ',' argumentType;
argumentTypeList : '(' argumentTypeListBody ')' | '(' ')';

callprototype : TOKEN_LABEL TOKEN_CALL_PROTOTYPE returnTypeList identifier 
	argumentTypeList ';'
{
	state.callPrototype( $<text>1, $<text>4, @1 );
};

calltargets : TOKEN_LABEL TOKEN_CALL_TARGETS identifierList ';'
{
	state.callTargets( $<text>1, @1 );
};

pragma : TOKEN_PRAGMA TOKEN_STRING
{
	state.pragma( $<text>2 );
};

pragma : TOKEN_PRAGMA TOKEN_STRING ';'
{
	state.pragma( $<text>2 );
};

vectorIndex : TOKEN_X | TOKEN_Y | TOKEN_Z | TOKEN_W;

optionalVectorIndex : vectorIndex
{
	state.vectorIndex( $<value>1 );
};

optionalVectorIndex : /* empty string */
{

};

nonLabelOperand : identifier optionalVectorIndex
{
	state.nonLabelOperand( $<text>1, @1, false );
};

nonLabelOperand : '!' identifier
{
	state.nonLabelOperand( $<text>2, @1, true );
};

constantOperand : TOKEN_DECIMAL_CONSTANT
{
	state.constantOperand( $<value>1 );
};

constantOperand : TOKEN_UNSIGNED_DECIMAL_CONSTANT
{
	state.constantOperand( $<uvalue>1 );
};

constantOperand : TOKEN_DOUBLE_CONSTANT
{
	state.constantOperand( $<doubleFloat>1 );
};

constantOperand : TOKEN_SINGLE_CONSTANT
{
	state.constantOperand( $<singleFloat>1 );
};

addressableOperand : identifier
{
	state.addressableOperand( $<text>1, 0, @1, false );
};

offsetAddressableOperand : identifier '+' TOKEN_DECIMAL_CONSTANT
{
	state.addressableOperand( $<text>1, $<value>3, @1, false );
};

offsetAddressableOperand : identifier '-' TOKEN_DECIMAL_CONSTANT
{
	state.addressableOperand( $<text>1, $<value>3, @1, true );
};

callOperand : operand
operand : constantOperand | nonLabelOperand;

memoryOperand : constantOperand | addressableOperand | offsetAddressableOperand;

branchOperand : labelOperand;

arrayOperand : operand;
arrayOperand : '{' identifierList '}'
{
	state.arrayOperand( @1 );
};

guard : TOKEN_PREDICATE_IDENTIFIER
{
	state.guard( $<text>1, @1, false );
};

guard : TOKEN_INV_PREDICATE_IDENTIFIER
{
	state.guard( $<text>1, @1, true );
};

guard : /* empty string */
{
	state.guard();
}

floatRoundingToken : TOKEN_RN  | TOKEN_RM  | TOKEN_RP  | TOKEN_RZ;
intRoundingToken   : TOKEN_RNI | TOKEN_RMI | TOKEN_RPI | TOKEN_RZI;

floatRounding : floatRoundingToken
{
	state.modifier( $<value>1 );
};

intRounding : intRoundingToken
{
	state.modifier( $<value>1 );
};

optionalFloatRounding : floatRounding | /* empty string */;

instruction : ftzInstruction2 | ftzInstruction3 | approxInstruction2 
	| basicInstruction3 | bfe | bfi | bfind | brev | branch | addOrSub
	| addCOrSubC | atom | bar | brkpt | clz | cvt | cvta | isspacep | div | exit
	| ld | ldu | mad | mad24 | madc | membar | mov | mul24 | mul | notInstruction
	| pmevent | popc | prefetch | prefetchu | prmt | rcpSqrtInstruction | red
	| ret | sad | selp | set | setp | slct | st | suld | suq | sured | sust
	| testp | tex | tld4 | trap | txq | vote | shfl;

basicInstruction3Opcode : OPCODE_AND | OPCODE_OR 
	| OPCODE_REM | OPCODE_SHL | OPCODE_SHR | OPCODE_XOR | OPCODE_COPYSIGN;

basicInstruction3 : basicInstruction3Opcode dataType operand ',' operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

approxInstruction2Opcode : OPCODE_RSQRT | OPCODE_SIN | OPCODE_COS | OPCODE_LG2 
	| OPCODE_EX2;

approximate : TOKEN_APPROX
{
	state.modifier( $<value>1 );
};

approxInstruction2 : approxInstruction2Opcode approximate optionalFtz dataType 
	operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>4 );
};

ftz : TOKEN_FTZ
{
	state.modifier( $<value>1 );
};

optionalFtz : ftz | /* empty string */;

sat : TOKEN_SAT
{
	state.modifier( $<value>1 );
};

optionalSaturate : sat | /* empty string */;

ftzInstruction2Opcode : OPCODE_ABS | OPCODE_NEG;

ftzInstruction2 : ftzInstruction2Opcode optionalFtz dataType operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

ftzInstruction3Opcode : OPCODE_MAX | OPCODE_MIN;

ftzInstruction3 : ftzInstruction3Opcode optionalFtz dataType operand ',' 
	operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

optionalUni : /* empty string */
{
	state.uni( false );
};

optionalUni : TOKEN_UNI
{
	state.uni( true );
};

branch : OPCODE_BRA optionalUni branchOperand ';'
{
	state.instruction( $<text>1 );
};

returnOperand : callOperand
{
	state.returnOperand();
};

returnOperandList : returnOperand;
returnOperandList : returnOperandList ',' returnOperand;

optionalReturnOperandList : '(' returnOperandList ')' ',' | /* empty string */;

callArgumentList : callOperand;
callArgumentList : callArgumentList ',' callOperand;

optionalPrototypeName : ',' '(' callArgumentList ')' ',' identifier
{
	state.callPrototypeName( $<text>6 );
};

optionalPrototypeName : ',' '(' ')' ',' identifier
{
	state.callPrototypeName( $<text>5 );
};

optionalPrototypeName : ',' identifier
{
	state.callPrototypeName( $<text>2 );
};

optionalPrototypeName : ',' '(' callArgumentList ')'
	| ',' '(' ')' | /* empty string */;

optionalUniOrTail : TOKEN_TAIL
{
	state.tail( true );
};

optionalUniOrTail : optionalUni
{
	state.tail( false );
};

branch : call;

call : OPCODE_CALL optionalUniOrTail optionalReturnOperandList identifier 
	optionalPrototypeName ';'
{
	state.call( $<text>4, @1 );
};

optionalCarry : TOKEN_CARRY
{
	state.carry( true );
};

optionalCarry : /* empty string */
{
	state.carry( false );
};

addModifier : TOKEN_CARRY
{
	state.carry( true );
};

addModifier : optionalFloatRounding optionalFtz optionalSaturate
{
	state.carry( false );
};

addOrSubOpcode : OPCODE_ADD | OPCODE_SUB;

addOrSub : addOrSubOpcode addModifier dataType operand ',' operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

addCOrSubCOpcode : OPCODE_ADDC | OPCODE_SUBC;

addCModifier : optionalCarry;

addCOrSubC : addCOrSubCOpcode addCModifier dataType operand ',' operand 
	',' operand ';'
{
	state.carryIn();
	state.instruction( $<text>1, $<value>3 );
};

atomicOperationId : TOKEN_AND | TOKEN_OR | TOKEN_XOR | TOKEN_CAS | TOKEN_EXCH 
	| TOKEN_ADD | TOKEN_INC | TOKEN_DEC | TOKEN_MIN | TOKEN_MAX;

atomicOperation : atomicOperationId
{
	state.atomic( $<value>1 );
};

atomModifier: addressSpace;
atomModifier: /* empty string */
{
	state.addressSpace(TOKEN_GLOBAL);
}

atom : OPCODE_ATOM atomModifier atomicOperation dataType operand ',' '[' 
	memoryOperand ']' ',' operand ';'
{
	state.instruction( $<text>1, $<value>4 );
};

atom : OPCODE_ATOM atomModifier atomicOperation dataType operand ',' '[' 
	memoryOperand ']' ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>4 );
};

shiftAmount : TOKEN_SHIFT_AMOUNT
{
	state.shiftAmount( true );
};

shiftAmount : /* empty string */
{
	state.shiftAmount( false );
};

bfe : OPCODE_BFE dataType operand ',' operand ',' operand 
	',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

bfi : OPCODE_BFI dataType operand ',' operand ',' operand 
	',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

bfind : OPCODE_BFIND shiftAmount dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

barrierOperation : TOKEN_ARRIVE | TOKEN_RED | TOKEN_SYNC
{
	state.barrierOperation( $<value>1, @1 );
};

optionalBarrierOperator : reductionOperation dataType | /* or nothing */ ;

operandSequence: operand operandSequence | /* empty */ ;

bar : OPCODE_BAR barrierOperation optionalBarrierOperator operandSequence ';'
{
	state.instruction( $<text>1 );
};

brev : OPCODE_BREV dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

brkpt : OPCODE_BRKPT ';'
{
	state.instruction( $<text>1 );
};

clz : OPCODE_CLZ dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

floatRoundingModifier : floatRounding
{
	state.modifier( $<value>1 );
};

intRoundingModifier : intRounding
{
	state.modifier( $<value>1 );
};

cvtRoundingModifier : intRoundingModifier | floatRoundingModifier;

cvtModifier : cvtRoundingModifier optionalFtz sat;
cvtModifier : cvtRoundingModifier optionalFtz;
cvtModifier : optionalFtz sat;
cvtModifier : optionalFtz;

cvt : OPCODE_CVT cvtModifier dataType dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
	state.relaxedConvert( $<value>4, @1 );
};

cvtaOperand : operand | offsetAddressableOperand;

cvta : OPCODE_CVTA addressSpace pointerDataType operand ',' cvtaOperand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

cvta : OPCODE_CVTA TOKEN_TO addressSpace pointerDataType operand
	',' cvtaOperand ';'
{
	state.instruction( $<text>1, $<value>4 );
	state.cvtaTo();
};

/*
	multiple orderings of modifiers for OptiX support
*/

divFullModifier : TOKEN_FULL optionalFtz
{
	state.full();
}

divApproxModifier : TOKEN_APPROX optionalFtz
{
	state.modifier($<value>1);
};

divRnModifier : TOKEN_RN optionalFtz
{
	state.modifier($<value>1);
};

divRnModifier : /* empty string */;

divModifier : divFullModifier | divApproxModifier 
	| divRnModifier;

div : OPCODE_DIV divModifier dataType operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

exit : OPCODE_EXIT ';'
{
	state.instruction( $<text>1 );
};

isspacep : OPCODE_ISSPACEP addressSpace operand ',' operand ';'
{
	state.instruction( $<text>1, TOKEN_U32 );
}

volatileModifier : TOKEN_VOLATILE
{
	state.volatileFlag( true );
};

optionalVolatile : volatileModifier;

optionalVolatile : /* empty string */
{
	state.volatileFlag( false );
};

ldModifier : optionalVolatile optionalAddressSpace optionalCacheOperation
	optionalInstructionVectorType;

ld : OPCODE_LD ldModifier dataType arrayOperand ',' '[' memoryOperand ']' ';'
{
	state.instruction( $<text>1, $<value>3 );
};

ldu : OPCODE_LDU ldModifier dataType arrayOperand ',' '[' memoryOperand ']' ';'
{
	state.instruction( $<text>1, $<value>3 );
}

hiOrLo : TOKEN_HI | TOKEN_LO;

roundHiLoWide : floatRounding | hiOrLo | TOKEN_WIDE;

mulModifier : roundHiLoWide optionalFtz optionalSaturate
{
	state.modifier( $<value>1 );
	state.carry( false );
};

mulModifier : hiOrLo TOKEN_CARRY
{
	state.carry( true );
};

mulModifier : optionalFtz optionalSaturate
{
	state.carry( false );
};

madOpcode : OPCODE_MAD | OPCODE_FMA;

mad : madOpcode mulModifier dataType operand ',' operand 
	',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

mad24Modifier : optionalSaturate;

mad24Modifier : hiOrLo optionalSaturate
{
	state.modifier( $<value>1 );
};

mad24 : OPCODE_MAD24 mad24Modifier dataType operand ',' operand 
	',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

madCModifier : hiOrLo optionalCarry
{
	state.modifier( $<value>1 );
};

madc : OPCODE_MADC madCModifier dataType operand ',' operand 
	',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

membarSpaceType : TOKEN_GL | TOKEN_CTA | TOKEN_SYS;

membarSpace : membarSpaceType
{
	state.level( $<value>1 );
};

membar : OPCODE_MEMBAR membarSpace ';'
{
	state.instruction( $<text>1 );
};

movIndexedOperand : identifier '[' TOKEN_DECIMAL_CONSTANT ']'
{
	state.indexedOperand( $<text>1, @1, $<value>3 );
};

movSourceOperand : arrayOperand | offsetAddressableOperand | movIndexedOperand;

mov : OPCODE_MOV dataType arrayOperand ',' movSourceOperand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

mul24Modifier : /* empty string */;

mul24Modifier : hiOrLo
{
	state.modifier( $<value>1 );
};

mul24 : OPCODE_MUL24 mul24Modifier dataType operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

mul : OPCODE_MUL mulModifier dataType operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

notOpcode : OPCODE_CNOT | OPCODE_NOT;

notInstruction : notOpcode dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

pmevent : OPCODE_PMEVENT operand
{
	state.instruction( $<text>1 );
};

popc : OPCODE_POPC dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

permuteModeType : TOKEN_F4E | TOKEN_B4E | TOKEN_RC8 
	| TOKEN_ECL | TOKEN_ECR | TOKEN_RC16;

permuteMode : permuteModeType
{
	state.permute( $<value>1 );
};

permuteMode : /* empty string */
{
	state.defaultPermute();
};

cacheLevel : TOKEN_L1 | TOKEN_L2
{
	state.cacheLevel( $<value>1 );
};

prefetch : OPCODE_PREFETCH addressSpace cacheLevel '[' memoryOperand ']' ';'
{
	state.instruction( $<text>1 );
};

prefetchu : OPCODE_PREFETCHU cacheLevel '[' memoryOperand ']' ';'
{
	state.instruction( $<text>1 );
};

prmt : OPCODE_PRMT dataType permuteMode operand ',' operand 
	',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

rcpSqrtModifier : TOKEN_APPROX optionalFtz
{
	state.modifier( $<value>1 );
};

rcpSqrtModifier : /* empty string */;
rcpSqrtModifier : TOKEN_RN optionalFtz
{
	state.modifier( $<value>1 );
};

rcpSqrtOpcode : OPCODE_RCP | OPCODE_SQRT;

rcpSqrtInstruction : rcpSqrtOpcode rcpSqrtModifier dataType operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

reductionOperationId : TOKEN_AND | TOKEN_XOR | TOKEN_OR | TOKEN_ADD | TOKEN_INC
	| TOKEN_DEC | TOKEN_MIN | TOKEN_MAX;
	
reductionOperation : reductionOperationId
{
	state.reduction( $<value>1 );
};

red : OPCODE_RED addressSpace reductionOperation dataType operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>4 );
};

ret : OPCODE_RET optionalUni ';'
{
	state.instruction( $<text>1 );
};

ret : OPCODE_RET optionalUni operand ';'
{
	state.instruction( $<text>1 );
};

comparisonId : TOKEN_EQ | TOKEN_NE | TOKEN_LT | TOKEN_LE | TOKEN_GT | TOKEN_GE
	| TOKEN_LS | TOKEN_HS | TOKEN_EQU | TOKEN_NEU | TOKEN_LTU | TOKEN_LEU
	| TOKEN_GTU | TOKEN_GEU | TOKEN_NUM | TOKEN_NAN | TOKEN_LO | TOKEN_HI;

comparison : comparisonId
{
	state.comparison( $<value>1 );
};

boolOperatorId : TOKEN_AND | TOKEN_OR | TOKEN_XOR;

boolOperator : boolOperatorId
{
	state.boolean( $<value>1 );
};

sad : OPCODE_SAD dataType operand ',' operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

selp : OPCODE_SELP dataType operand ',' operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>2 );
	state.operandCIsAPredicate();
};


/* multiple orderings of modifiers for OptiX support */
setModifier : comparison optionalFtz;
setModifier : ftz comparison;

set : OPCODE_SET setModifier dataType dataType operand ',' 
	operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
	state.convert( $<value>4, @1 );
};

set : OPCODE_SET comparison boolOperator optionalFtz dataType dataType operand 
	',' operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>5 );
	state.convert( $<value>6, @1 );
};

/* multiple orderings of modifiers for OptiX support */
setpModifier : comparison optionalFtz;
setpModifier : ftz comparison;

predicatePair : operand '|' operand | operand;

setp : OPCODE_SETP setpModifier dataType predicatePair ',' operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

setp : OPCODE_SETP dataType setpModifier predicatePair ',' operand ',' 
	operand ';'
{
	state.instruction( $<text>1, $<value>2 );
};

setp : OPCODE_SETP comparison boolOperator optionalFtz dataType predicatePair 
	',' operand ',' operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>5 );
};

shuffleModifierId : TOKEN_UP | TOKEN_DOWN | TOKEN_BFLY | TOKEN_IDX;

shuffleModifier : shuffleModifierId
{
	state.shuffle( $<value>1 );
};

shfl : OPCODE_SHFL shuffleModifier dataType predicatePair ',' operand ','
	operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

slct : OPCODE_SLCT optionalFtz dataType dataType operand ',' operand ',' 
	operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
	state.convertC( $<value>4, @1 );
};

st : OPCODE_ST ldModifier dataType '[' memoryOperand ']' ',' arrayOperand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

geometryId : TOKEN_1D | TOKEN_2D | TOKEN_3D | TOKEN_A1D | TOKEN_A2D |
	TOKEN_CUBE | TOKEN_ACUBE;

geometry : geometryId
{
	state.geometry( $<value>1 );
};

floatingPointModeType : TOKEN_FINITE | TOKEN_INFINITE | TOKEN_NUMBER 
	| TOKEN_NOT_A_NUMBER | TOKEN_NORMAL | TOKEN_SUBNORMAL;

floatingPointMode : floatingPointModeType
{
	state.floatingPointMode( $<value>1 );
}; 

testp : OPCODE_TESTP floatingPointMode dataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

tex : OPCODE_TEX geometry TOKEN_V4 dataType dataType arrayOperand ',' '[' 
	operand ',' arrayOperand ']' ';'
{
	state.tex( $<value>5 );
	state.convertD( $<value>4, @1 );
};

colorComponentId : TOKEN_A | TOKEN_B | TOKEN_R | TOKEN_G;

colorComponent : colorComponentId
{
	state.colorComponent( $<value>1 );
};

tld4 : OPCODE_TLD4 colorComponent TOKEN_2D TOKEN_V4 dataType dataType
	arrayOperand ',' '[' operand ',' arrayOperand ']' ';'
{
	state.tld4( $<value>5 );
};

//
// Surface sampling 
// 

surfaceQuery : TOKEN_WIDTH | TOKEN_HEIGHT | TOKEN_DEPTH
	| TOKEN_CHANNEL_DATA_TYPE | TOKEN_CHANNEL_ORDER | TOKEN_NORMALIZED_COORDS
	| TOKEN_FILTER_MODE | TOKEN_ADDR_MODE_0 | TOKEN_ADDR_MODE_1
	| TOKEN_ADDR_MODE_2
{
	state.surfaceQuery( $<value>1 );
};

txq : OPCODE_TXQ surfaceQuery dataType operand ',' '[' operand ']' ';'
{
	state.surfaceQuery( $<value>2 );
	state.instruction( $<text>1, $<value>3 );
};

suq : OPCODE_SUQ surfaceQuery dataType operand ',' '[' operand ']' ';'
{
	state.instruction( $<text>1, $<value>3 );
	state.surfaceQuery( $<value>2 );
};

cacheOperation : TOKEN_CA | TOKEN_CG | TOKEN_CS | TOKEN_CV | TOKEN_NC
{
	state.cacheOperation( $<value>1 );
};

optionalCacheOperation : cacheOperation | /* empty */;

clampOperation : TOKEN_CLAMP | TOKEN_ZERO | TOKEN_TRAP
{
	state.clampOperation( $<value>1 );
};

formatMode : TOKEN_B | TOKEN_P
{
	state.formatMode( $<value>1 );
};

suld : OPCODE_SULD formatMode geometry optionalCacheOperation 
	instructionVectorType dataType clampOperation arrayOperand ',' 
	'[' operand ',' arrayOperand ']' ';'
{
	state.instruction( $<text>1, $<value>6 );
	state.formatMode( $<value>2 );
	state.clampOperation( $<value>7 );
};

sust : OPCODE_SUST formatMode geometry optionalCacheOperation 
	instructionVectorType dataType clampOperation '[' operand ','
	arrayOperand ']' ',' arrayOperand ';'
{
	state.instruction( $<text>1, $<value>6 );
	state.formatMode( $<value>2 );
	state.clampOperation( $<value>7 );
};

sured : OPCODE_SURED formatMode reductionOperation geometry dataType
	clampOperation '[' operand ',' arrayOperand ']' ',' arrayOperand ';'
{
	state.instruction( $<text>1, $<value>5 );
	state.formatMode( $<value>2 );
	state.clampOperation( $<value>6 );
};

//
// End surface sampling 
// 

trap : OPCODE_TRAP ';'
{
	state.instruction( $<text>1 );
};

voteOperationId : TOKEN_ANY | TOKEN_ALL | TOKEN_UNI | TOKEN_BALLOT;

voteOperation : voteOperationId
{
	state.vote( $<value>1 );
};

voteDataType : TOKEN_PRED | TOKEN_B32;

vote : OPCODE_VOTE voteOperation voteDataType operand ',' operand ';'
{
	state.instruction( $<text>1, $<value>3 );
};

%%

int yylex( YYSTYPE* token, YYLTYPE* location, parser::PTXLexer& lexer, 
	parser::PTXParser::State& state )
{
	lexer.yylval = token;
	
	int tokenValue         = lexer.yylexPosition();
	location->first_line   = lexer.lineno();
	location->first_column = lexer.column;
	
	report( " Lexer (" << location->first_line << ","
		<< location->first_column 
		<< "): " << parser::PTXLexer::toString( tokenValue ) << " \"" 
		<< lexer.YYText() << "\"");
	
	return tokenValue;
}

void yyerror( YYLTYPE* location, parser::PTXLexer& lexer, 
	parser::PTXParser::State& state, char const* message )
{
	parser::PTXParser::Exception exception;
	std::stringstream stream;
	stream << parser::PTXParser::toString( *location, state ) 
		<< " " << message;
	exception.message = stream.str();
	exception.error = parser::PTXParser::State::SyntaxError;
	throw exception;
}

}
