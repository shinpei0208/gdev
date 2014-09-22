/*!

	\file XmlParser.cpp
	
	\date Sunday September 14, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The source file for the XmlParser class.

*/

#ifndef XML_PARSER_CPP_INCLUDED
#define XML_PARSER_CPP_INCLUDED

#include <hydrazine/interface/XmlParser.h>

#include <hydrazine/interface/Exception.h>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace hydrazine
{
		
	void XmlParser::_beginCaretOpen( )
	{
	
		assert( _state == BeginCaretOpen );
		
		if( _lexer->token().type != XmlLexer::Token::CARET_OPEN )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column << " ) : Expecting '<', but \
				got '" << _lexer->token().string << 
				"' instead.\nThe line was:\n\t" << _lexer->token().lineText;
			throw Exception( error.str() );
			
		}
		
		report( " Hit state: BeginCaretOpen" );
		report( " Advancing state to BeginIdentifier" );
		
		_state = BeginIdentifier;
				
	}
	
	void XmlParser::_beginIdentifier( )
	{
	
		assert( _state == BeginIdentifier );
		assert( _identifierStack.empty() );
		
		if( _lexer->token().type != XmlLexer::Token::IDENTIFIER )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting IDENTIFIER, but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}
		
		_treeIterator = _tree.insert( _lexer->token().string, _treeIterator,
			XmlTree::XmlTreeNode::Intermediate );
		_identifierStack.push( _treeIterator->identifier );

		report( " Hit state: BeginIdentifier" );
		report( " Advancing state to BeginCaretClose" );
		
		_state = BeginCaretClose;
	
	}
	
	void XmlParser::_beginCaretClose( )
	{
	
		assert( _state == BeginCaretClose );
		
		if( _lexer->token().type != XmlLexer::Token::CARET_CLOSE )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting '>', but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}

		report( " Hit state: BeginCaretClose" );
		report( " Advancing state to IntermediateCaretOpenOrIdentifier" );
		
		_state = IntermediateCaretOpenOrIdentifier;
	
	}
	
	void XmlParser::_intermediateCaretOpenOrIdentifier( )
	{
	
		assert( _state == IntermediateCaretOpenOrIdentifier );
		
		if( _lexer->token().type != XmlLexer::Token::CARET_OPEN
			&& _lexer->token().type != XmlLexer::Token::IDENTIFIER )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting '<' or IDENTFIER, but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}

		report( " Hit state: IntermediateCaretOpenOrIdentifier" );
		
		if( _lexer->token().type == XmlLexer::Token::CARET_OPEN )
		{
		
			_state = IntermediateCaretOpenThenIdentifierOrBackslash;
			report( " Advancing state to" <<
				" IntermediateCaretOpenThenIdentifierOrBackslash" );
		
		}
		else
		{

			_tree.insert( _lexer->token().string, _treeIterator,
					XmlTree::XmlTreeNode::Leaf );
			_state = IntermediateCaretOpen;
			report( " Advancing state to IntermediateCaretOpen" );
		
		}
			
	}
	
	void XmlParser::_intermediateCaretOpenThenIdentifierOrBackslash( )
	{
	
		assert( _state == IntermediateCaretOpenThenIdentifierOrBackslash );

		if( _lexer->token().type != XmlLexer::Token::BACKSLASH
			&& _lexer->token().type != XmlLexer::Token::IDENTIFIER )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting '/' or IDENTFIER, but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}
		
		report( " Hit state: IntermediateCaretOpenThenIdentifierOrBackslash" );
		
		if( _lexer->token().type == XmlLexer::Token::BACKSLASH )
		{
		
			report( " Advancing state to " <<
				"IntermediateCaretOpenThenBackslashThenIdentifier" );
			_state = IntermediateCaretOpenThenBackslashThenIdentifier;
		
		}
		else
		{
		
			_treeIterator = _tree.insert( _lexer->token().string, 
				_treeIterator, 
				XmlTree::XmlTreeNode::Intermediate );
			_identifierStack.push( _treeIterator->identifier );
			report( " Advancing state to " << 
				"IntermediateCaretOpenThenIdentifierThenCaretClose" );
			_state = IntermediateCaretOpenThenIdentifierThenCaretClose;
		
		}
	
	}
	
	void XmlParser::_intermediateCaretOpenThenIdentifierThenCaretClose( )
	{
	
		assert( _state == IntermediateCaretOpenThenIdentifierThenCaretClose );

		if( _lexer->token().type != XmlLexer::Token::CARET_CLOSE )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting '<', but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}

		report( " Hit state: " 
			<< "IntermediateCaretOpenThenIdentifierThenCaretClose" );
		report( " Advancing state to IntermediateCaretOpenOrIdentifier" );
		
		_state = IntermediateCaretOpenOrIdentifier;
	
	}
	
	void XmlParser::_intermediateCaretOpenThenBackslashThenIdentifier( )
	{
	
		assert( _state == IntermediateCaretOpenThenBackslashThenIdentifier );
	
		if( _lexer->token().type != XmlLexer::Token::IDENTIFIER )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting IDENTIFIER, but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}
		
		if( _lexer->token().string != _identifierStack.top() )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Tag mismatch - original was '" 
				<< _identifierStack.top() 
				<< "'.\n It did not match : " << _lexer->token().string 
				<< ". The line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
		
		}

		report( " Hit state: " 
			<< "IntermediateCaretOpenThenBackslashThenIdentifier" );
		report( " Advancing state to " << 
			"IntermediateCaretOpenThenBackslashThenIdentifierThenCaretClose" );
		_treeIterator.ascend();
		_identifierStack.pop();
		_state = IntermediateCaretOpenThenBackslashThenIdentifierThenCaretClose;
	
	}
	
	void 
	XmlParser::_intermediateCaretOpenThenBackslashThenIdentifierThenCaretClose( )
	{
	
		assert( _state == 
			IntermediateCaretOpenThenBackslashThenIdentifierThenCaretClose );
	
		if( _lexer->token().type != XmlLexer::Token::CARET_CLOSE )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting '>', but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}
	
		report( " Hit state: " <<
			"IntermediateCaretOpenThenBackslashThenIdentifierThenCaretClose" );
		
		if( _identifierStack.empty() )
		{
		
			_state = BeginCaretOpen;
			report( " Advancing state to BeginCaretOpen" );
		
		}
		else
		{
		
			_state = IntermediateCaretOpenOrIdentifier;
			report( " Advancing state to IntermediateCaretOpenOrIdentifier" );
		
		}
	
	}
	
	void XmlParser::_intermediateCaretOpen( )
	{
	
		assert( _state == IntermediateCaretOpen );
		
		if( _lexer->token().type != XmlLexer::Token::CARET_OPEN )
		{
		
			std::stringstream error;
			error << _lexer->fileName() << "( " << _lexer->token().line 
				<< ", " << _lexer->token().column 
				<< " ) : Expecting '<' or IDENTFIER, but got '" 
				<< _lexer->token().string 
				<< "' instead.\nThe line was:\n\t" 
				<< _lexer->token().lineText;
			throw Exception( error.str() );
			
		}

		report( " Hit state: " <<
			"IntermediateCaretOpen" );
	
		_state = IntermediateCaretOpenThenIdentifierOrBackslash;
		report( " Advancing state to " << 
			"IntermediateCaretOpenThenIdentifierOrBackslash" );
	
	}

	void XmlParser::_handleToken()
	{
	
		switch( _state )
		{
		
			case BeginCaretOpen:
			{
			
				_beginCaretOpen();
				break;
				
			}
				
			case BeginIdentifier:
			{
			
				_beginIdentifier();
				break;
				
			}
				
			case BeginCaretClose:
			{
			
				_beginCaretClose();
				break;
				
			}
				
			case IntermediateCaretOpenOrIdentifier:
			{
			
				_intermediateCaretOpenOrIdentifier();
				break;
				
			}
				
			case IntermediateCaretOpenThenIdentifierOrBackslash:
			{
			
				_intermediateCaretOpenThenIdentifierOrBackslash();
				break;
				
			}
				
			case IntermediateCaretOpenThenIdentifierThenCaretClose:
			{
			
				_intermediateCaretOpenThenIdentifierThenCaretClose();
				break;
				
			}
				
			case IntermediateCaretOpenThenBackslashThenIdentifier:
			{
			
				_intermediateCaretOpenThenBackslashThenIdentifier();
				break;
				
			}
				
			case IntermediateCaretOpenThenBackslashThenIdentifierThenCaretClose:
			{
			
				_intermediateCaretOpenThenBackslashThenIdentifierThenCaretClose();
				break;
				
			}
				
			case IntermediateCaretOpen:
			{
			
				_intermediateCaretOpen();
				break;
				
			}
			
			default:
			{
			
				assert( "XmlParser: Unknown state." == 0 );
				break;
			
			}
		
		}
	
	}
	
	XmlParser::XmlParser( const std::string& fileName )
	{
		report( "Creating new XML parser on file " << fileName );
		_lexer = new XmlLexer( fileName );
		_state = BeginCaretOpen;
		_treeIterator = _tree.begin();
		_treeIterator->identifier = fileName;
		
		report( "Parsing file:" );
		
		try
		{
			while( _lexer->next() )
			{
				_handleToken();
			}
		}
		catch( const Exception& e )
		{
			delete _lexer;
			_lexer = 0;
			throw e;		
		}
		
		delete _lexer;
		_lexer = 0;
	}
	
	XmlParser::~XmlParser()
	{
	
		assert( _lexer == 0 );
	
	}
	
	const XmlTree& XmlParser::tree() const
	{
	
		return _tree;
	
	}

}

#endif

