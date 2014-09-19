/*!
	\file TestXmlParser.h
	\date Monday September 15, 2008
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the TestXmlParser class.
*/

#ifndef TEST_XML_PARSER_H_INCLUDED
#define TEST_XML_PARSER_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <hydrazine/implementation/XmlTree.h>

namespace test
{
	/*!
		\brief Generate a random XML tree, write it out to a temp file,
		then parse it and compare to make sure that the trees match.
	*/
	class TestXmlParser : public Test
	{
		private:
			void initializeTree( hydrazine::XmlTree& tree ) const;
			void copyTree( 
				hydrazine::XmlTree& reference, 
				hydrazine::XmlTree& parsed ) const;
			bool compareTrees( 
				hydrazine::XmlTree& reference, 
				hydrazine::XmlTree& parsed );
		
			bool doTest( );
		
		public:
			TestXmlParser();
			
			unsigned int nodes;
			unsigned int maxLevels;
			unsigned int maxNodesPerLevel;	
			std::string tempFile;
			bool dontDelete;
	
	};

}

int main( int argc, char** argv );

#endif

