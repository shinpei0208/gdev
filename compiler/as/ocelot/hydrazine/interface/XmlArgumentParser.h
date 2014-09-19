/*!

	\file XmlArgumentParser.h
	
	\date Monday September 15, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The header file for the XmlArgumentParser class.

*/

#ifndef XML_ARGUMENT_PARSER_H_INCLUDED
#define XML_ARGUMENT_PARSER_H_INCLUDED

#include "XmlParser.h"
#include <sstream>

namespace hydrazine
{

	class XmlArgumentParser
	{
	
		private:
		
			XmlTree _tree;
			XmlTree::iterator _treeIterator;
		
		public:
		
			XmlArgumentParser( const std::string& fileName );
			~XmlArgumentParser();
			
			void reset();
			void ascend( );
			void descend( const std::string& tag );
			const std::string& tag() const;
			const XmlTree& tree() const;
			
			template< class T >
			void parse( T& i ) const;
	
						
	};

	template< class T >
	void XmlArgumentParser::parse( T& i ) const
	{
	
		std::stringstream stream( _treeIterator.leaf() );
		stream >> i;
	
	}

}

#endif

