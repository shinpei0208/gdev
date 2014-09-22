/*!

	\file XmlLexer.h
	
	\date Saturday September 13, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The header file for the XmlLexer class.

*/

#ifndef XML_LEXER_H_INCLUDED
#define XML_LEXER_H_INCLUDED

#include <fstream>
#include <string>

namespace hydrazine
{

	class XmlLexer
	{
	
		public:
					
			class Token
			{
				
				public:

					enum Type
					{
			
						CARET_OPEN = '<',
						CARET_CLOSE = '>',
						BACKSLASH = '/',
						IDENTIFIER,
						END_OF_FILE = -1,
						INVALID
			
					};					

					Type type;
					unsigned int column;
					unsigned int line;
					std::string lineText;
					std::string string;
			
			};

		private:
				
			Token _token;
			std::ifstream file;
			std::string _fileName;

		private:
		
			bool devourComment();
			void devourWhiteSpace();
			void tokenizeIdentifier();
			void peekline( std::string& string );
			
		public:
		
			XmlLexer( const std::string& fileName );
			~XmlLexer();
			
			bool next();
			const Token& token() const;
			const std::string& fileName() const;
	
	};

}

#endif

