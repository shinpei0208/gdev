/*!

	\file XmlArgumentParser.cpp
	
	\date Monday September 15, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The source file for the XmlArgumentParser class.

*/

#ifndef XML_ARGUMENT_PARSER_CPP_INCLUDED
#define XML_ARGUMENT_PARSER_CPP_INCLUDED

#include <hydrazine/interface/XmlArgumentParser.h>

namespace hydrazine
{

	XmlArgumentParser::XmlArgumentParser( const std::string& fileName )
	{
	
		XmlParser parser( fileName );
		_tree = parser.tree();
		_treeIterator = _tree.begin();
	
	}
		
	XmlArgumentParser::~XmlArgumentParser()
	{
	
	
	}
	
	void XmlArgumentParser::ascend( )
	{
	
		_treeIterator.ascend( );
		
	}
	
	void XmlArgumentParser::reset( )
	{
	
		_treeIterator = _tree.begin();
	
	} 
	
	void XmlArgumentParser::descend( const std::string& tag )
	{
	
		_treeIterator.descend( tag );
	
	}
	
	const std::string& XmlArgumentParser::tag() const
	{
	
		return _treeIterator->identifier;
	
	}

	const XmlTree& XmlArgumentParser::tree() const
	{
	
		return _tree;
	
	}
			
}

#endif

