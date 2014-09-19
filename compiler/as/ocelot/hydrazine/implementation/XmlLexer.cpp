/*!

	\file XmlLexer.cpp
	
	\date Saturday September 13, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The source file for the XmlLexer class.

*/

#ifndef XML_LEXER_CPP_INCLUDED
#define XML_LEXER_CPP_INCLUDED

#include <hydrazine/interface/XmlLexer.h>
#include <hydrazine/interface/Exception.h>
#include <cstring>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace hydrazine
{

	bool XmlLexer::devourComment()
	{

		report( "Checking for comments." );
	
		devourWhiteSpace();
		
		char commentBegin[5];
		commentBegin[4] = 0;
		std::ios::streampos position = file.tellg();
		file.readsome( commentBegin, 4 );
		
		if( file.gcount() == 4 )
		{
		
			if( !strcmp( "<!--", commentBegin ) )
			{
			
				unsigned int endCount = 0;
				
				while( endCount < 3 )
				{
				
					char get = file.get();
				
					if( !file.good() )
					{
					
						file.seekg( position );
						return false;							
						
					}
				
					switch( endCount )
					{
						
						case 0:
						
						case 1:
						{
						
							if( get == '-' )
							{
							
								++endCount;
							
							}
							break;						
						}
						
						case 2:
						{

							if( get == '>' )
							{
							
								++endCount;
							
							}
							break;
						
						}
						
						default:
						{
						
							assert( "Invalid count." == 0 );
						
						}
						
					}
				
				}
				
				devourWhiteSpace();
						
				return true;
			
			}
			else
			{
			
				file.seekg( position );
				return false;
			
			}
		
		}
		else
		{
		
			file.seekg( position );
			return false;
			
		}
	
	}


	void XmlLexer::devourWhiteSpace()
	{
	
		bool foundWhiteSpace = true;
	
		report( " Consuming white space." );
	
		while( foundWhiteSpace )
		{
		
			int peek = file.peek();

			if( !file.good() )
			{
			
				break;
			
			}
			
			switch( peek )
			{
			
				case ' ':
				case '\t':
				{
					file.get();
					break;
				}
				
				case '\n':
				{
				
					_token.column = 0;
					++_token.line;
					file.get();
					peekline( _token.lineText );
					break;
				}
				
				default:
				{
				
					foundWhiteSpace = false;
					break;
				
				}
			
			}
		
		}
		
	}

	void XmlLexer::tokenizeIdentifier()
	{
	
		int peek = file.peek();
		bool string = false;
	
		_token.string.clear();
		_token.type = Token::IDENTIFIER;
		
		do
		{
		
			if( peek == '"' )
			{
				string = !string;
				file.get();
			}
			else if( string )
			{
				_token.string.push_back( file.get() );
				++_token.column;
			}
			else if( peek == '\n' || peek == ' ' || peek == '\t' )
			{
			
				std::ios::streampos checkpoint = file.tellg();
				bool scanning = true;
				while( scanning )
				{
				
					int get = file.peek();
										
					switch( get )
					{
					
						case Token::END_OF_FILE:
						case Token::BACKSLASH:
						case Token::CARET_OPEN:
						case Token::CARET_CLOSE:
						{

							scanning = false;
							++_token.column;
							break;
						
						}
						
						case '\n':
						{
						
							++_token.line;
							_token.column = 0;			
						
						}
						case ' ':
						case '\t':
						{
							file.get();
							break;
						}
						
						default:
						{

							file.seekg( checkpoint );						
							_token.string.push_back( file.get() );
							++_token.column;
							scanning = false;
							break;
							
						}
					
					}
				
				}
			
			}
			else
			{
			
				_token.string.push_back( file.get() );
				++_token.column;
			
			}

			peek = file.peek();		
					
		}
		while
		(
		
			peek != Token::END_OF_FILE &&
			( ( peek != Token::BACKSLASH &&
			peek != Token::CARET_OPEN &&
			peek != Token::CARET_CLOSE ) || string )
		
		);
	
	}
	
	void XmlLexer::peekline( std::string& string )
	{

		report( " Peeking at the next line." );
	
		string.clear();
		std::ios::streampos position = file.tellg();
		
		while( file.good() )
		{
		
			if( file.peek() == '\n' )
			{
			
				break;
			
			}
			
			string.push_back( file.get() );
		
		}
		
		file.seekg( position );
		report( " The next line was:\n  " << string );
	
	}
	
	XmlLexer::XmlLexer( const std::string& fileName )
	{
	
		file.open( fileName.c_str() );
		_fileName = fileName;
		
		if( !file.is_open() )
		{
			throw Exception ("Could not open XML file " + fileName);
		}
		
		_token.column = 0;
		_token.line = 0;
		peekline( _token.lineText );		
		_token.string = "";
		_token.type = Token::INVALID;
		while( devourComment() );
	
	}
	
	XmlLexer::~XmlLexer()
	{
	
		file.close();
	
	}
	
	bool XmlLexer::next()
	{

		int nextChar = file.peek();
		bool nextValid = true;
		
		if( !file.good() )
		{
		
			nextChar = Token::END_OF_FILE;
		
		}
		
		switch( nextChar )
		{
		
			case Token::CARET_OPEN:
			{
			
				_token.type = Token::CARET_OPEN;
				_token.string = Token::CARET_OPEN;
				++_token.column;
				file.get();
				break;
				
			}
			
			case Token::CARET_CLOSE:
			{
			
				_token.type = Token::CARET_CLOSE;
				_token.string = Token::CARET_CLOSE;
				++_token.column;
				file.get();
				break;
					
			}
			
			case Token::BACKSLASH:
			{
			
				_token.type = Token::BACKSLASH;
				_token.string = Token::BACKSLASH;
				++_token.column;
				file.get();
				break;
			
			}
			
			case Token::END_OF_FILE:
			{
			
				_token.type = Token::END_OF_FILE;
				_token.string = Token::END_OF_FILE;
				++_token.column;
				nextValid = false;				
				break;
		
			}
			
			default:
			{
			
				tokenizeIdentifier();
				break;
				
			}
		
		}	
		
		if( nextValid )
		{
		
			while( devourComment() );
	
		}
		
		report( "Got token: " << _token.string );
		
		return nextValid;
	
	}
	
	const XmlLexer::Token& XmlLexer::token() const
	{
	
		return _token;
	
	}
	
	const std::string& XmlLexer::fileName() const
	{
	
		return _fileName;
	
	}

}

#endif

