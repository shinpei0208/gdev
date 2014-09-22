/*!

	\file XmlParser.h
	
	\date Sunday September 14, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The header file for the XmlParser class.

*/

#ifndef XML_PARSER_H_INCLUDED
#define XML_PARSER_H_INCLUDED

#include "XmlTree.h"
#include "XmlLexer.h"

namespace hydrazine
{

	class XmlParser
	{

		private:
		
			enum State
			{
			
				BeginCaretOpen = 0,
				BeginIdentifier = 1,
				BeginCaretClose = 2,
				
				IntermediateCaretOpenOrIdentifier = 3,
				IntermediateCaretOpenThenIdentifierOrBackslash = 4,
				IntermediateCaretOpenThenIdentifierThenCaretClose = 5,
				IntermediateCaretOpenThenBackslashThenIdentifier = 6,
				IntermediateCaretOpenThenBackslashThenIdentifierThenCaretClose = 7,
				
				IntermediateCaretOpen = 8,
				
				Finished
			
			};
			
			State _state;
			std::stack< std::string > _identifierStack;
			
		private:
		
			void _beginCaretOpen( );
			void _beginIdentifier( );
			void _beginCaretClose( );
			void _intermediateCaretOpenOrIdentifier( );
			void _intermediateCaretOpenThenIdentifierOrBackslash( );
			void _intermediateCaretOpenThenIdentifierThenCaretClose( );
			void _intermediateCaretOpenThenBackslashThenIdentifier( );
			void _intermediateCaretOpenThenBackslashThenIdentifierThenCaretClose( );
			void _intermediateIdentifier( );
			void _intermediateCaretOpen( );
	
			void _handleToken();
	
		private:
		
			XmlLexer* _lexer;
			XmlTree _tree;
			XmlTree::iterator _treeIterator;
		
		public:
		
			XmlParser( const std::string& fileName );
			~XmlParser();
			
			const XmlTree& tree() const;
	
	};

}

#endif

