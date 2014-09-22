/*! \file ptx.lpp
	\date Wednesday January 14, 2009
	\author Gregory Diamos
	\brief The flex lexical description of the PTX language
*/

/******************************************************************************/
/* DEFINITIONS                                                                */

%option yylineno
%option noyywrap
%option yyclass="parser::PTXLexer"
%option prefix="ptx"
%option c++

%{
	
	#ifndef PTX_LPP_INCLUDED
	#define PTX_LPP_INCLUDED

	// Ocelot Includes
	#include <ocelot/parser/interface/PTXLexer.h>

	// Hydrazine Includes
	#include <hydrazine/interface/macros.h>
	#include <hydrazine/interface/string.h>

	// Standard Library Includes
	#include <cassert>
	#include <sstream>
	#include <cstring>
		
	void sstrcpy( char* destination, const char* source, unsigned int max );
	
	// Convert binary string to uint
	long long unsigned int binaryToUint( const std::string& );
	
%}

BIT [01]
NON_ZERO_DIGIT [1-9]
BINARY_CONSTANT (0[bB][{BIT}]+)
DECIMAL_CONSTANT ("-"?{NON_ZERO_DIGIT}[[:digit:]]*)
HEX_CONSTANT (0[xX][[:xdigit:]]+)
SINGLE_CONSTANT (0[dDfF][[:xdigit:]]{8})
DOUBLE_CONSTANT (0[dDfF][[:xdigit:]]{16})
OCT_CONSTANT (0[01234567]*)

UNSIGNED_BINARY_CONSTANT {BINARY_CONSTANT}U
UNSIGNED_DECIMAL_CONSTANT {DECIMAL_CONSTANT}U
UNSIGNED_HEX_CONSTANT {HEX_CONSTANT}U
UNSIGNED_OCT_CONSTANT {OCT_CONSTANT}U

STRING L?\"(\\.|[^\\"])*\"
COMMENT ("/*"([^*]|"*"[^/])*"*/")|("/"(\\\n)*"/"[^\n]*)
METADATA ("//!"[^\n]*)

DSEQ ([[:digit:]]+)
DSEQ_OPT ([[:digit:]]*)
FRAC (({DSEQ_OPT}"."{DSEQ})|{DSEQ}".")
EXP ([eE][+-]?{DSEQ})
EXP_OPT ({EXP}?)
FSUFF [flFL]
FSUFF_OPT ({FSUFF}?)
HPREF (0[xX])
HDSEQ ([[:xdigit:]]+)
HDSEQ_OPT ([[:xdigit:]]*)
HFRAC (({HDSEQ_OPT}"."{HDSEQ})|({HDSEQ}"."))
BEXP ([pP][+-]?{DSEQ})
DFC (({FRAC}{EXP_OPT}{FSUFF_OPT})|({DSEQ}{EXP}{FSUFF_OPT}))
HFC (({HPREF}{HFRAC}{BEXP}{FSUFF_OPT})|({HPREF}{HDSEQ}{BEXP}{FSUFF_OPT}))
FLOAT_CONSTANT ({DFC}|{HFC})

FOLLOWSYM  [[:alnum:]_$]
IDENTIFIER ([[:alpha:]]{FOLLOWSYM}*|[_$%]{FOLLOWSYM}+)
VECTOR_SUFFIX {""}

NEW_LINE ([\n\r]*)
TAB [\t]*
SPACE [ ]*
WHITESPACE [ \t\r]*
LABEL ({IDENTIFIER}{WHITESPACE}":")

/******************************************************************************/

%%
                                    
"#include"                      { yylval->value = PREPROCESSOR_INCLUDE; \
                                    return PREPROCESSOR_INCLUDE; }
"#define"                       { yylval->value = PREPROCESSOR_DEFINE; \
                                    return PREPROCESSOR_DEFINE; }
"#if"                           { yylval->value = PREPROCESSOR_IF; \
                                    return PREPROCESSOR_IF; }
"#ifdef"                        { yylval->value = PREPROCESSOR_IFDEF; \
                                    return PREPROCESSOR_IFDEF; }
"#else"                         { yylval->value = PREPROCESSOR_ELSE; \
                                    return PREPROCESSOR_ELSE; }
"#endif"                        { yylval->value = PREPROCESSOR_ENDIF; \
                                    return PREPROCESSOR_ENDIF; }
"#line"                         { yylval->value = PREPROCESSOR_LINE; \
                                    return PREPROCESSOR_LINE; }
"#file"                         { yylval->value = PREPROCESSOR_FILE; \
                                    return PREPROCESSOR_FILE; }

"add"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_ADD; }
"addc"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_ADDC; }
"and"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_AND; }
"atom"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_ATOM; }
"abs"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_ABS; }
"bar"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BAR; }
"bfi"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BFI; }
"bfe"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BFE; }
"bfind"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BFIND; }
"bra"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BRA; }
"brev"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BREV; }
"brkpt"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_BRKPT; }
"call"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_CALL; }
"clz"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_CLZ; }
"cnot"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_CNOT; }
"copysign"                      { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_COPYSIGN; }
"cos"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_COS; }
"cvt"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_CVT; }
"cvta"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_CVTA; }
"div"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_DIV; }
"ex2"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_EX2; }
"exit"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_EXIT; }
"fma"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_FMA; }
"isspacep"                      { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_ISSPACEP; }
"ld"                            { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_LD; }
"ldu"                            { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_LDU; }
"lg2"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_LG2; }
"membar"                        { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MEMBAR; }
"min"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MIN; }
"mad"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MAD; }
"madc"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MADC; }
"mad24"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MAD24; }
"max"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MAX; }
"mov"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MOV; }
"mul"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MUL; }
"mul24"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_MUL24; }
"neg"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_NEG; }
"not"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_NOT; }
"or"                            { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_OR; }
"pmevent"                       { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_PMEVENT; }
"prefetch"                      { sstrcpy( yylval->text, yytext, 1024 ); \
                                  return OPCODE_PREFETCH; }
"prefetchu"                     { sstrcpy( yylval->text, yytext, 1024 ); \
                                  return OPCODE_PREFETCHU; }
"popc"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_POPC; }
"prmt"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_PRMT; }
"rcp"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_RCP; }
"red"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_RED; }
"rem"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_REM; }
"ret"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_RET; }
"rsqrt"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_RSQRT; }
"sad"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SAD; }
"selp"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SELP; }
"set"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SET; }
"setp"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SETP; }
"shfl"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SHFL; }
"shl"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SHL; }
"shr"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SHR; }
"sin"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SIN; }
"slct"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SLCT; }
"sqrt"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SQRT; }
"st"                            { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_ST; }
"sub"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SUB; }
"subc"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SUBC; }
"suld"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SULD; }
"sust"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SUST; }
"sured"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SURED; }
"suq"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_SUQ; }
"testp"                         { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_TESTP; }
"tex"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_TEX; }
"tld4"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_TLD4; }
"trap"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_TRAP; }
"txq"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_TXQ; }
"vote"                          { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_VOTE; }
"xor"                           { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return OPCODE_XOR; }


".align"                        { yylval->value = TOKEN_ALIGN; \
                                    return TOKEN_ALIGN; }
".address_size"                 { yylval->value = TOKEN_ADDRESS_SIZE; \
                                    return TOKEN_ADDRESS_SIZE; }
".callprototype"                { yylval->value = TOKEN_CALL_PROTOTYPE; \
                                    return TOKEN_CALL_PROTOTYPE; }
".calltargets"                  { yylval->value = TOKEN_CALL_TARGETS; \
                                    return TOKEN_CALL_TARGETS; }
".const"                        { yylval->value = TOKEN_CONST; \
                                    return TOKEN_CONST; }
".entry"                        { yylval->value = TOKEN_ENTRY; \
                                    return TOKEN_ENTRY; }
".extern"                       { yylval->value = TOKEN_EXTERN; \
                                    return TOKEN_EXTERN; }
".file"                         { yylval->value = TOKEN_FILE; \
                                    return TOKEN_FILE; }
".func"                         { yylval->value = TOKEN_FUNCTION; \
                                    return TOKEN_FUNCTION; }
".global"                       { yylval->value = TOKEN_GLOBAL; \
                                    return TOKEN_GLOBAL; }
".local"                        { yylval->value = TOKEN_LOCAL; \
                                    return TOKEN_LOCAL; }
".loc"                          { yylval->value = TOKEN_LOC; \
                                    return TOKEN_LOC; }
".maxnctapersm"                 { yylval->value = TOKEN_MAXNCTAPERSM; 
                                    return TOKEN_MAXNCTAPERSM; }
".minnctapersm"                 { yylval->value = TOKEN_MINNCTAPERSM; 
                                    return TOKEN_MINNCTAPERSM; }
".maxnreg"                      { yylval->value = TOKEN_MAXNREG; 
                                    return TOKEN_MAXNREG; }
".maxntid"                      { yylval->value = TOKEN_MAXNTID; 
                                    return TOKEN_MAXNTID; }
".param"                        { yylval->value = TOKEN_PARAM; \
                                    return TOKEN_PARAM; }
".pragma"                       { yylval->value = TOKEN_PRAGMA; \
                                    return TOKEN_PRAGMA; }
".ptr"                          { yylval->value = TOKEN_PTR; \
									return TOKEN_PTR; }
".reg"                          { yylval->value = TOKEN_REG; \
                                    return TOKEN_REG; }
".samplerref"                   { yylval->value = TOKEN_SAMPLERREF; \
                                    return TOKEN_SAMPLERREF; }
".section"                      { yylval->value = TOKEN_SECTION; \
                                    return TOKEN_SECTION; }
".shared"                       { yylval->value = TOKEN_SHARED; \
                                    return TOKEN_SHARED;}
".shiftamt"                     { yylval->value = TOKEN_SHIFT_AMOUNT; \
                                    return TOKEN_SHIFT_AMOUNT;}
".surfref"	                   { yylval->value = TOKEN_SURFREF; \
                                    return TOKEN_SURFREF; }
".target"                       { yylval->value = TOKEN_TARGET; \
                                    return TOKEN_TARGET; }
".texref"                       { yylval->value = TOKEN_TEXREF; \
                                    return TOKEN_TEXREF; }
".version"                      { yylval->value = TOKEN_VERSION; \
                                    return TOKEN_VERSION; }
".visible"                      { yylval->value = TOKEN_VISIBLE; \
                                    return TOKEN_VISIBLE; }
".weak"                         { yylval->value = TOKEN_WEAK; \
                                    return TOKEN_WEAK; }

".cta"                          { yylval->value = TOKEN_CTA; return TOKEN_CTA; }
".gl"                           { yylval->value = TOKEN_GL; return TOKEN_GL; }
".sys"                          { yylval->value = TOKEN_SYS; return TOKEN_SYS; }

"sm_10"                         { yylval->value = TOKEN_SM10; 
                                    return TOKEN_SM10; }
"sm_11"                         { yylval->value = TOKEN_SM11; 
                                    return TOKEN_SM11; }
"sm_12"                         { yylval->value = TOKEN_SM12; 
                                    return TOKEN_SM12; }
"sm_13"                         { yylval->value = TOKEN_SM13; 
                                    return TOKEN_SM13; }
"sm_20"                         { yylval->value = TOKEN_SM20; 
                                    return TOKEN_SM20; }
"sm_21"                         { yylval->value = TOKEN_SM21; 
                                    return TOKEN_SM21; }
"sm_30"                         { yylval->value = TOKEN_SM30; 
                                    return TOKEN_SM30; }
"sm_35"                         { yylval->value = TOKEN_SM35; 
                                    return TOKEN_SM35; }
"map_f64_to_f32"                { yylval->value = TOKEN_MAP_F64_TO_F32; 
                                    return TOKEN_MAP_F64_TO_F32; }
"texmode_independent"           { yylval->value = TOKEN_TEXMODE_INDEPENDENT; 
                                    return TOKEN_TEXMODE_INDEPENDENT; }
"texmode_unified"         		  { yylval->value = TOKEN_TEXMODE_UNIFIED; 
                                    return TOKEN_TEXMODE_UNIFIED; }

".u32"				            { yylval->value = TOKEN_U32; return TOKEN_U32; }
".s32"				            { yylval->value = TOKEN_S32; return TOKEN_S32; }
".s8"				            { yylval->value = TOKEN_S8; return TOKEN_S8; }
".s16"			            	{ yylval->value = TOKEN_S16; return TOKEN_S16; }
".s64"			            	{ yylval->value = TOKEN_S64; return TOKEN_S64; }
".u8"			            	{ yylval->value = TOKEN_U8; return TOKEN_U8; }
".u16"			            	{ yylval->value = TOKEN_U16; return TOKEN_U16; }
".u64"			            	{ yylval->value = TOKEN_U64; return TOKEN_U64; }
".b8"			            	{ yylval->value = TOKEN_B8; return TOKEN_B8; }
".b16"			            	{ yylval->value = TOKEN_B16; return TOKEN_B16; }
".b32"			            	{ yylval->value = TOKEN_B32; return TOKEN_B32; }
".b64"			            	{ yylval->value = TOKEN_B64; return TOKEN_B64; }
".f16"			            	{ yylval->value = TOKEN_F16; return TOKEN_F16; }
".f64"			            	{ yylval->value = TOKEN_F64; return TOKEN_F64; }
".f32"			            	{ yylval->value = TOKEN_F32; return TOKEN_F32; }
".pred"		                    { yylval->value = TOKEN_PRED; \
                                    return TOKEN_PRED; }

".eq"                           { yylval->value = TOKEN_EQ; return TOKEN_EQ; }
".ne"                           { yylval->value = TOKEN_NE; return TOKEN_NE; }
".lt"                           { yylval->value = TOKEN_LT; return TOKEN_LT; }
".le"                           { yylval->value = TOKEN_LE; return TOKEN_LE; }
".gt"                           { yylval->value = TOKEN_GT; return TOKEN_GT; }
".ge"                           { yylval->value = TOKEN_GE; return TOKEN_GE; }
".ls"                           { yylval->value = TOKEN_LS; return TOKEN_LS; }
".hs"                           { yylval->value = TOKEN_HS; return TOKEN_HS; }
".equ"                          { yylval->value = TOKEN_EQU; return TOKEN_EQU; }
".neu"                          { yylval->value = TOKEN_NEU; return TOKEN_NEU; }
".ltu"                          { yylval->value = TOKEN_LTU; return TOKEN_LTU; }
".leu"                          { yylval->value = TOKEN_LEU; return TOKEN_LEU; }
".gtu"                          { yylval->value = TOKEN_GTU; return TOKEN_GTU; }
".geu"                          { yylval->value = TOKEN_GEU; return TOKEN_GEU; }
".num"                          { yylval->value = TOKEN_NUM; return TOKEN_NUM; }
".nan"                          { yylval->value = TOKEN_NAN; return TOKEN_NAN; }

".and"                          { yylval->value = TOKEN_AND; return TOKEN_AND; }
".or"                           { yylval->value = TOKEN_OR; return TOKEN_OR; }
".xor"                          { yylval->value = TOKEN_XOR; return TOKEN_XOR; }

".hi"                           { yylval->value = TOKEN_HI; return TOKEN_HI; }
".lo"                           { yylval->value = TOKEN_LO; return TOKEN_LO; }
".rn"                           { yylval->value = TOKEN_RN; return TOKEN_RN; }
".rm"                           { yylval->value = TOKEN_RM; return TOKEN_RM; }
".rz"                           { yylval->value = TOKEN_RZ; return TOKEN_RZ; }
".rp"                           { yylval->value = TOKEN_RP; return TOKEN_RP; }
".rni"                          { yylval->value = TOKEN_RNI; return TOKEN_RNI; }
".rmi"                          { yylval->value = TOKEN_RMI; return TOKEN_RMI; }
".rzi"                          { yylval->value = TOKEN_RZI; return TOKEN_RZI; }
".rpi"                          { yylval->value = TOKEN_RPI; return TOKEN_RPI; }
".sat"                          { yylval->value = TOKEN_SAT; return TOKEN_SAT; }
".ftz"                          { yylval->value = TOKEN_FTZ; return TOKEN_FTZ; }
".approx"                       { yylval->value = TOKEN_APPROX; \
                                    return TOKEN_APPROX; }

".tail"                         { yylval->value = TOKEN_TAIL; \
                                    return TOKEN_TAIL; }
".uni"                          { yylval->value = TOKEN_UNI; return TOKEN_UNI; }
".byte"                         { yylval->value = TOKEN_BYTE; \
                                    return TOKEN_BYTE; }
".wide"                         { yylval->value = TOKEN_WIDE; \
                                    return TOKEN_WIDE; }
".cc"                           { yylval->value = TOKEN_CARRY; \
                                    return TOKEN_CARRY; }
".volatile"                     { yylval->value = TOKEN_VOLATILE; \
                                    return TOKEN_VOLATILE; }
".full"                         { yylval->value = TOKEN_FULL; \
                                    return TOKEN_FULL; }

".v2"                           { yylval->value = TOKEN_V2; return TOKEN_V2; }
".v4"                           { yylval->value = TOKEN_V4; return TOKEN_V4; }

".x"                            { yylval->value = TOKEN_X; return TOKEN_X; }
".y"                            { yylval->value = TOKEN_Y; return TOKEN_Y; }
".z"                            { yylval->value = TOKEN_Z; return TOKEN_Z; }
".w"                            { yylval->value = TOKEN_W; return TOKEN_W; }
".r"                            { yylval->value = TOKEN_R; return TOKEN_R; }
".g"                            { yylval->value = TOKEN_G; return TOKEN_G; }
".b"                            { yylval->value = TOKEN_B; return TOKEN_B; }
".a"                            { yylval->value = TOKEN_A; return TOKEN_A; }

".any"                          { yylval->value = TOKEN_ANY; return TOKEN_ANY; }
".all"                          { yylval->value = TOKEN_ALL; return TOKEN_ALL; }

".up"                           { yylval->value = TOKEN_UP;   return TOKEN_UP;   }
".down"                         { yylval->value = TOKEN_DOWN; return TOKEN_DOWN; }
".bfly"                         { yylval->value = TOKEN_BFLY; return TOKEN_BFLY; }
".idx"                          { yylval->value = TOKEN_IDX;  return TOKEN_IDX;  }

".min"                          { yylval->value = TOKEN_MIN; return TOKEN_MIN; }
".max"                          { yylval->value = TOKEN_MAX; return TOKEN_MAX; }
".dec"                          { yylval->value = TOKEN_DEC; return TOKEN_DEC; }
".inc"                          { yylval->value = TOKEN_INC; return TOKEN_INC; }
".add"                          { yylval->value = TOKEN_ADD; return TOKEN_ADD; }
".cas"                          { yylval->value = TOKEN_CAS; return TOKEN_CAS; }
".exch"                         { yylval->value = TOKEN_EXCH; return TOKEN_EXCH; }

".1d"                           { yylval->value = TOKEN_1D; return TOKEN_1D; }
".2d"                           { yylval->value = TOKEN_2D; return TOKEN_2D; }
".3d"                           { yylval->value = TOKEN_3D; return TOKEN_3D; }
".a1d"                          { yylval->value = TOKEN_A1D; return TOKEN_A1D; }
".a2d"                          { yylval->value = TOKEN_A2D; return TOKEN_A2D; }
".cube"                         { yylval->value = TOKEN_CUBE; return TOKEN_CUBE; }
".acube"                        { yylval->value = TOKEN_ACUBE; return TOKEN_ACUBE; }

".p"                           	{ yylval->value = TOKEN_P; return TOKEN_P; }

".ca"                           { yylval->value = TOKEN_CA; return TOKEN_CA; }
".wb"                           { yylval->value = TOKEN_WB; return TOKEN_WB; }
".cg"                           { yylval->value = TOKEN_CG; return TOKEN_CG; }
".cs"                           { yylval->value = TOKEN_CS; return TOKEN_CS; }
".lu"                           { yylval->value = TOKEN_LU; return TOKEN_LU; }
".cv"                           { yylval->value = TOKEN_CV; return TOKEN_CV; }
".nc"                           { yylval->value = TOKEN_NC; return TOKEN_NC; }
".wt"                           { yylval->value = TOKEN_WT; return TOKEN_WT; }

".L1"                           { yylval->value = TOKEN_L1; return TOKEN_L1; }
".L2"                           { yylval->value = TOKEN_L2; return TOKEN_L2; }

".width"                        { yylval->value = TOKEN_WIDTH; \
                                    return TOKEN_WIDTH; }
".height"                       { yylval->value = TOKEN_HEIGHT; \
                                    return TOKEN_HEIGHT; }
".depth"                        { yylval->value = TOKEN_DEPTH; \
                                    return TOKEN_DEPTH; }
".normalized_coords"            { yylval->value = TOKEN_NORMALIZED_COORDS; \
                                    return TOKEN_NORMALIZED_COORDS; }
".filter_mode"                  { yylval->value = TOKEN_FILTER_MODE; \
                                    return TOKEN_FILTER_MODE; }
".addr_mode_0"                  { yylval->value = TOKEN_ADDR_MODE_0; \
                                    return TOKEN_ADDR_MODE_0; }
".addr_mode_1"                  { yylval->value = TOKEN_ADDR_MODE_1; \
                                    return TOKEN_ADDR_MODE_1; }
".addr_mode_2"                  { yylval->value = TOKEN_ADDR_MODE_2; \
                                    return TOKEN_ADDR_MODE_2; }
".channel_data_type"            { yylval->value = TOKEN_CHANNEL_DATA_TYPE; \
                                    return TOKEN_CHANNEL_DATA_TYPE; }
".channel_order"                { yylval->value = TOKEN_CHANNEL_ORDER; \
                                    return TOKEN_CHANNEL_ORDER; }
".trap"                         { yylval->value = TOKEN_TRAP; \
                                    return TOKEN_TRAP; }
".to"                           { yylval->value = TOKEN_TO; return TOKEN_TO; }
".clamp"                        { yylval->value = TOKEN_CLAMP; 
                                    return TOKEN_CLAMP; }
".zero"                         { yylval->value = TOKEN_ZERO; \
                                    return TOKEN_ZERO; }
".arrive"                       { yylval->value = TOKEN_ARRIVE; \
                                    return TOKEN_ARRIVE; }
".red"                          { yylval->value = TOKEN_RED; \
                                    return TOKEN_RED; }
".sync"                          { yylval->value = TOKEN_SYNC; \
                                    return TOKEN_SYNC; }
".popc"                         { yylval->value = TOKEN_POPC; \
                                    return TOKEN_POPC; }

".ballot"                       { yylval->value = TOKEN_BALLOT; \
                                    return TOKEN_BALLOT; }
                                    
".f4e"                          { yylval->value = TOKEN_F4E; return TOKEN_F4E; }
".b4e"                          { yylval->value = TOKEN_B4E; return TOKEN_B4E; }
".rc8"                          { yylval->value = TOKEN_RC8; return TOKEN_RC8; }
".ecl"                          { yylval->value = TOKEN_ECL; return TOKEN_ECL; }
".ecr"                          { yylval->value = TOKEN_ECR; return TOKEN_ECR; }
".rc16"                         { yylval->value = TOKEN_RC16; \
                                    return TOKEN_RC16; }
        
".finite"                       { yylval->value = TOKEN_FINITE; \
                                    return TOKEN_FINITE; }
".infinite"                     { yylval->value = TOKEN_INFINITE; \
                                    return TOKEN_INFINITE; }
".number"                       { yylval->value = TOKEN_NUMBER; \
                                    return TOKEN_NUMBER; }
".notanumber"                   { yylval->value = TOKEN_NOT_A_NUMBER; \
                                    return TOKEN_NOT_A_NUMBER; }
".normal"                       { yylval->value = TOKEN_NORMAL; \
                                    return TOKEN_NORMAL; }
".subnormal"                    { yylval->value = TOKEN_SUBNORMAL; \
                                    return TOKEN_SUBNORMAL; }

@{IDENTIFIER}                   { sstrcpy( yylval->text, yytext + 1, 1024 ); \
                                    return TOKEN_PREDICATE_IDENTIFIER; }
@!{IDENTIFIER}                  { sstrcpy( yylval->text, yytext + 2, 1024 ); \
                                    return TOKEN_INV_PREDICATE_IDENTIFIER; }
{IDENTIFIER}                    { sstrcpy( yylval->text, yytext, 1024 ); \
                                    return TOKEN_IDENTIFIER;}
{STRING}                        { sstrcpy( yylval->text, yytext + 1, \
                                    MIN( strlen( yytext ) - 1, 1024 ) ); \
                                    return TOKEN_STRING;}
                                                                       
{DECIMAL_CONSTANT}              { std::stringstream stream; stream << yytext; \
                                    stream >> yylval->value; \
                                    return TOKEN_DECIMAL_CONSTANT; }
{BINARY_CONSTANT}               { yylval->value = binaryToUint( yytext ); \
                                    return TOKEN_DECIMAL_CONSTANT; }
{HEX_CONSTANT}                  { std::stringstream stream; \
                                    stream << std::hex; stream << yytext; \
                                    stream >> yylval->value; \
                                    return TOKEN_DECIMAL_CONSTANT; }
{OCT_CONSTANT}                  { std::stringstream stream; \
                                    stream << std::oct; stream << yytext; \
                                    stream >> yylval->value; \
                                    return TOKEN_DECIMAL_CONSTANT; }

{UNSIGNED_DECIMAL_CONSTANT}     { std::stringstream stream; stream << yytext; \
                                    stream >> yylval->uvalue; \
                                    return TOKEN_DECIMAL_CONSTANT; }
{UNSIGNED_BINARY_CONSTANT}      { yylval->uvalue = binaryToUint( yytext ); \
                                    return TOKEN_DECIMAL_CONSTANT; }
{UNSIGNED_HEX_CONSTANT}         { std::stringstream stream; \
                                    stream << std::hex; stream << yytext; \
                                    stream >> yylval->uvalue; \
                                    return TOKEN_DECIMAL_CONSTANT; }
{UNSIGNED_OCT_CONSTANT}         { std::stringstream stream; \
                                    stream << std::oct; stream << yytext; \
                                    stream >> yylval->uvalue; \
                                    return TOKEN_DECIMAL_CONSTANT; }

{SINGLE_CONSTANT}               { yytext[1] = 'x'; std::stringstream stream; \
                                    stream << std::hex << yytext; \
                                    stream >> yylval->uvalue; \
                                    return TOKEN_SINGLE_CONSTANT; }
{DOUBLE_CONSTANT}               { yytext[1] = 'x'; std::stringstream stream; \
                                    stream << std::hex << yytext; \
                                    stream >> yylval->uvalue; \
                                    return TOKEN_DOUBLE_CONSTANT; }                  
{FLOAT_CONSTANT}                { std::stringstream stream; stream << yytext; \
                                    stream >> yylval->doubleFloat; \
                                    return TOKEN_DOUBLE_CONSTANT; }

{LABEL}                         { const char* position = strchr( yytext, ' ' );\
                                    if( position == 0 ) \
                                    { \
                                        sstrcpy( yylval->text, yytext, \
                                           MIN( strlen( yytext ), 1024 ) ); \
                                    } \
                                    else \
                                    { \
                                        sstrcpy( yylval->text, yytext, \
                                           MIN( position - yytext + 1, \
                                           1024 ) ); \
                                    } \
                                    \
                                    return TOKEN_LABEL; \
                                }


{METADATA}                      { sstrcpy( yylval->text, yytext, 1024 );
                                    return TOKEN_METADATA; }
{COMMENT}                       { nextColumn += strlen( yytext ); }
{TAB}                           { nextColumn += strlen( yytext ) * 4; }
{SPACE}                         { nextColumn += strlen( yytext ); }
{NEW_LINE}                      { nextColumn  = 1; }

","                             { yylval->text[0] = ','; 
                                    yylval->text[1] = '\0'; return (','); }
";"                             { yylval->text[0] = ';'; 
                                    yylval->text[1] = '\0'; return (';'); }
"."                             { yylval->text[0] = '.'; 
                                    yylval->text[1] = '\0'; return ('.'); }
"{"                             { yylval->text[0] = '{'; 
                                    yylval->text[1] = '\0'; return ('{'); }
"}"                             { yylval->text[0] = '}'; 
                                    yylval->text[1] = '\0'; return ('}'); }
"["                             { yylval->text[0] = '['; 
                                    yylval->text[1] = '\0'; return ('['); }
"]"                             { yylval->text[0] = ']'; 
                                    yylval->text[1] = '\0'; return (']'); }
"("                             { yylval->text[0] = '('; 
                                    yylval->text[1] = '\0'; return ('('); }
")"                             { yylval->text[0] = ')'; 
                                    yylval->text[1] = '\0'; return (')'); }
"<"                             { yylval->text[0] = '<'; 
                                    yylval->text[1] = '\0'; return ('<'); }
"+"                             { yylval->text[0] = '+'; 
                                    yylval->text[1] = '\0'; return ('+'); }
">"                             { yylval->text[0] = '>'; 
                                    yylval->text[1] = '\0'; return ('>'); }
"="                             { yylval->text[0] = '='; 
                                    yylval->text[1] = '\0'; return ('='); }
"-"                             { yylval->text[0] = '-'; 
                                    yylval->text[1] = '\0'; return ('-'); }
"!"                             { yylval->text[0] = '!'; 
                                    yylval->text[1] = '\0'; return ('!'); }
"|"                             { yylval->text[0] = '|'; 
                                    yylval->text[1] = '\0'; return ('|'); }
"_"                             { yylval->text[0] = '_'; 
                                    yylval->text[1] = '\0'; return ('_'); }

%%

/******************************************************************************/
/* USER CODE                                                                  */

long long unsigned int binaryToUint( const std::string& string )
{
	return hydrazine::binaryToUint( string );
}

void sstrcpy( char* destination, const char* source, unsigned int max )
{
	return hydrazine::strlcpy( destination, source, max );
}

#endif

/******************************************************************************/

