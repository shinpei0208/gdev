/*!
	\file TestXmlArgumentParser.cpp
	\date Wednesday September 17, 2008
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the TestXmlArgumentParser class.
*/

#ifndef TEST_XML_ARGUMENT_PARSER_CPP_INCLUDED
#define TEST_XML_ARGUMENT_PARSER_CPP_INCLUDED

#include "TestXmlArgumentParser.h"
#include <fstream>
#include <hydrazine/implementation/XmlArgumentParser.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE 
#endif

#define REPORT_BASE 0

namespace test
{

	void TestXmlArgumentParser::initializeTree( hydrazine::XmlTree& tree, 
		std::list< TestXmlArgumentParser::Path >& paths )
	{
		unsigned int n = 0;
		hydrazine::XmlTree::iterator fi;
		unsigned int totalLevels = 0;
		Path path;
				
		while( n < nodes )
		{
			unsigned int levels = rand() % maxLevels;
			fi = tree.begin();
			
			for( unsigned int l = 0; l < levels; ++l )
			{
				std::stringstream name;
				name << "level_" << totalLevels << "_" << l;
				fi = tree.insert( name.str(), fi, 
					hydrazine::XmlTree::XmlTreeNode::Intermediate );
				path.path.push_back( name.str() );
			
				if( rand() & 0x1 )
				{				
					std::stringstream nodeName;
					nodeName << rand();
					tree.insert( nodeName.str(), fi, 
						hydrazine::XmlTree::XmlTreeNode::Leaf );
					++n;
					path.value = nodeName.str();
					paths.push_back( path );
				}				
					
				if( n >= nodes )
				{
					break;
				}
			}
			
			path.path.clear();
			++totalLevels;
		}
	}
		
	bool TestXmlArgumentParser::parseTree( 
		hydrazine::XmlTree& reference, 
		std::list< TestXmlArgumentParser::Path >& paths )
	{
		std::ofstream file( tempFile.c_str() );
		
		assert( file.is_open() );
		reference.toStream( file );
		file.close();
		
		hydrazine::XmlArgumentParser parser( tempFile );
		
		if( !dontDelete )
		{
			std::remove( tempFile.c_str() );
		}		

		for( std::list< TestXmlArgumentParser::Path >::iterator 
			fi = paths.begin(); fi != paths.end(); ++fi )
		{
			for( std::list< std::string >::iterator si = fi->path.begin();
				si != fi->path.end(); ++si )
			{
				report( "Searching for node " << *si );
				parser.descend( *si );
			}
			
			unsigned int parsedValue;
			parser.parse( parsedValue );
			std::stringstream stream;
			stream << parsedValue;
			
			if( stream.str() != fi->value )
			{
				status << "At path ";
				
				for( std::list< std::string >::iterator si = fi->path.begin();
					si != fi->path.end(); ++si )
				{
					status << *si << "->";
				}
				
				status << fi->value << ", parsed incorrect value " 
					<< stream.str();
				return false;
			}
			
			parser.reset();
		}
		
		status << "Test Passed.";
		return true;
	}

	bool TestXmlArgumentParser::doTest()
	{
		hydrazine::XmlTree tree;
		std::list< Path > paths;
		
		initializeTree( tree, paths );
		return parseTree( tree, paths );
	}

	TestXmlArgumentParser::TestXmlArgumentParser()
	{
		name = "TestXmlArgumentParser";
		
		description = "Creates a random XML tree, writes to a temp file,";
		description += " parses it with the XmlArgumentParser, then makes sure";
		description += " that all parsed values match their recorded values.";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestXmlArgumentParser test;
	
	parser.description( test.testDescription() );
	parser.parse( "-v", test.verbose, false, 
		"Print out information after the test is over." );
	parser.parse( "-s", test.seed, 0, "Random seed for repeatability." );
	parser.parse( "-n", test.nodes, 100, 
		"The total number of values to specify in the XML file." );
	parser.parse( "-m", test.maxLevels, 10, 
		"The maximum levels to nest values in the XML tree." );
	
	{
		std::stringstream stream;
		stream << "tmp_" << rand() << ".txt";
		parser.parse( "-t", test.tempFile, stream.str().c_str(), 
			"The name of a file to use a temp for the XML parser." );
	}
	
	parser.parse( "-d", test.dontDelete, false, 
		"Don't delete the generated temp XML file." );
	parser.parse();

	test.test();
	return test.passed();
}

#endif

