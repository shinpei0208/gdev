/*!

	\file TestXmlParser.cpp
	
	\date Monday September 15, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The source file for the TestXmlParser class.

*/

#ifndef TEST_XML_PARSER_CPP_INCLUDED
#define TEST_XML_PARSER_CPP_INCLUDED

#include "TestXmlParser.h"
#include <hydrazine/implementation/XmlParser.h>
#include <hydrazine/implementation/ArgumentParser.h>

#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace test
{

	void TestXmlParser::initializeTree( hydrazine::XmlTree& tree ) const
	{
		unsigned int n = 0;
		hydrazine::XmlTree::iterator fi;
		unsigned int totalLevels = 0;
				
		while( n < nodes )
		{
			unsigned int levels = rand() % maxLevels;
			fi = tree.begin();
			
			for( unsigned int l = 0; l < levels; ++l )
			{
				unsigned int nodesPerLevel = maxNodesPerLevel;
				std::stringstream name;
				name << "level_" << totalLevels << "_" << l;
				fi = tree.insert( name.str(), fi, 
					hydrazine::XmlTree::XmlTreeNode::Intermediate );
			
				for( unsigned int npl = 0; npl < nodesPerLevel; ++npl )
				{
					std::stringstream nodeName;
					nodeName << "node_" << n;
					tree.insert( nodeName.str(), fi, 
						hydrazine::XmlTree::XmlTreeNode::Leaf );
					++n;
					
					if( n >= nodes )
					{
						break;
					}
				}
					
				if( n >= nodes )
				{
					break;
				}
			}
			
			++totalLevels;
		}
		
		tree.begin()->identifier = tempFile;
	}
	
	void TestXmlParser::copyTree( 
		hydrazine::XmlTree& reference, 
		hydrazine::XmlTree& parsed ) const
	{
		std::ofstream file( tempFile.c_str() );
		
		assert( file.is_open() );
		reference.toStream( file );
		file.close();
		
		hydrazine::XmlParser parser( tempFile.c_str() );

		parsed = parser.tree();
		
		if( !dontDelete )
		{
			std::remove( tempFile.c_str() );
		}
	}
	
	
	bool TestXmlParser::compareTrees( 
		hydrazine::XmlTree& reference, 
		hydrazine::XmlTree& parsed )
	{
	
		hydrazine::XmlTree::iterator referenceIterator;
		hydrazine::XmlTree::iterator parsedIterator = parsed.begin();
				
		if( reference.size() != parsed.size() )
		{
			status << "Reference XmlTree size " << reference.size() << 
				" does not equal parsed size " << parsed.size();
			return false;
		}
		
		for( referenceIterator = reference.begin(); 
			referenceIterator != reference.end(); 
			++referenceIterator )
		{
			if( parsedIterator == parsed.end() )
			{
				status << "Parsed XML tree hit end early at label " 
					<< parsedIterator->identifier;
				return false;
			}
		
			if( parsedIterator->identifier != referenceIterator->identifier )
			{
				status << "Reference XmlTree label '" 
					<< referenceIterator->identifier << 
					"' does not equal parsed label '" 
					<< parsedIterator->identifier
					<< "'";
				return false;
			}
			
			++parsedIterator;
		}
		
		if( parsedIterator != parsed.end() )
		{
			status << "Parsed XML tree hit end late at label " 
				<< parsedIterator->identifier;
			return false;
		}
		
		status << "Test passed.";
		return true;		
	}

	bool TestXmlParser::doTest()
	{
		hydrazine::XmlTree reference;
		hydrazine::XmlTree parsed;
		
		initializeTree( reference );
		copyTree( reference, parsed );
		
		return compareTrees( reference, parsed );
	}
	
	TestXmlParser::TestXmlParser()
	{
		name = "TestXmlParser";
		
		description = "Tests the XmlParser by generating a random XML tree, ";
		description += "writing it out to a file, parsing the file, and ";
		description += "comparing the two XML trees.";
		
		maxNodesPerLevel = 1;
	}
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );

	test::TestXmlParser test;
	
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
	
	test.test();

	return test.passed();	
}

#endif

