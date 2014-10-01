/*!
	\file PTXLexer.h
	\date Monday January 19, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the PTXLexer class.
*/

#ifndef PTX_LEXER_H_INCLUDED
#define PTX_LEXER_H_INCLUDED

// FIXME: From the some version of bison, generated header contains yyparse
// function prototype. Because it uses undeclared nested class and it contains
// circular reference, we cannot build it with this prototype.
// To avoid this compile error, we replace `yyparse(...)` to
// `dummy_yyparse_to_avoid_compile_error(void)` by using C macro.
#define yyparse(...) dummy_yyparse_to_avoid_compile_error(void)
#include <ptxgrammar.hpp>
#undef yyparse

namespace parser
{
	/*!	\brief A wrapper around yyFlexLexer to allow for a local variable */
	class PTXLexer : public ptxFlexLexer
	{
		public:
			YYSTYPE*     yylval;
			int          column;
			int          nextColumn;

		public:
			PTXLexer( std::istream* arg_yyin = 0, 
				std::ostream* arg_yyout = 0 );
	
			int yylex();
			int yylexPosition();
			
		public:
			static std::string toString( int token );
	
	};

}

#endif

