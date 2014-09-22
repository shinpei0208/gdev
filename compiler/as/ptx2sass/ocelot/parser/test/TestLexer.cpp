/*!

	\file TestLexer.cpp
	
	\date Saturday January 17, 2009
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>

	\brief The source file for the TestLexer class

*/

#ifndef TEST_LEXER_CPP_INCLUDED
#define TEST_LEXER_CPP_INCLUDED

#include "TestLexer.h"
#include <boost/filesystem.hpp>
#include <queue>
#include <fstream>

#undef yyFlexLexer
#define yyFlexLexer ptxFlexLexer
#include <FlexLexer.h>
#include <ocelot/parser/interface/PTXLexer.h>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/macros.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace fs = boost::filesystem;

namespace test
{

	TestLexer::StringVector TestLexer::_getFileNames() const
	{
	
		StringVector names;
		
		fs::path path = input;
		
		if( fs::is_directory( path ) )
		{
		
			std::queue< fs::path > directories;
			directories.push( path );
			
			fs::directory_iterator end;
			
			while( !directories.empty() )
			{
			
				for( fs::directory_iterator 
					file( directories.front() );
					file != end; ++file )
				{
				
					if( fs::is_directory( file->status() ) && recursive )
					{
					
						directories.push( file->path() );
					
					}
					else if( fs::is_regular_file( file->status() ) )
					{
					
						if( file->path().extension() == ".ptx" )
						{
			
							names.push_back( file->path().string() );
			
						}
					
					}
				
				}
				
				directories.pop();
			
			}
		
		}
		else if( fs::is_regular_file( path ) )
		{
		
			if( path.extension() == ".ptx" )
			{
			
				names.push_back( path.string() );
			
			}
		
		}
		
		return names;
	
	}


	bool TestLexer::_testScan()
	{
	
		StringVector first;
		StringVector second;
		
		report( " Scanning file " << ptxFile );
		
		std::ifstream file( ptxFile.c_str() );
		std::stringstream temp;
		std::stringstream stream;
		
		assert( file.is_open() );
		
		YYSTYPE token;
		
		parser::PTXLexer lexer( &file, &temp );
		lexer.yylval = &token;
		
		int tokenValue = lexer.yylex();
		
		while( tokenValue != 0 )
		{
		
			report( "  Line (" << lexer.lineno() << "): Scanned token " 
				<< parser::PTXLexer::toString( tokenValue ) << " \"" 
				<< lexer.YYText() << "\"" );
				
			if( parser::PTXLexer::toString( tokenValue ) == "INVALID" )
			{
			
				status << "First pass line (" << lexer.lineno() 
					<< "): Hit invalid token \"" 
					<< lexer.YYText() << "\"\n";
				return false;
			
			}
			
			first.push_back( lexer.YYText() );
			stream << lexer.YYText() << "\n";
			tokenValue = lexer.yylex();
		
		}
		
		report( " Scanning internal stream." );
		
		lexer.switch_streams( &stream, 0 );
		
		if( temp.str().size() != 0 )
		{
		
			status << "First pass did not consume all of the file, the " 
				<< "following was rejected:\n'" << temp.str() << "'\n";
			status << " remaining characters:" << temp.str().size() << "\n";
			return false;
		
		}
		
		tokenValue = lexer.yylex();
		
		while( tokenValue != 0 )
		{
		
			report( "  Line (" << lexer.lineno() << "): Scanned token " 
				<< parser::PTXLexer::toString( tokenValue ) << " \"" 
				<< lexer.YYText() << "\"" );
			
			if( parser::PTXLexer::toString( tokenValue ) == "INVALID" )
			{
			
				status << "Second pass line (" << lexer.lineno() 
					<< "): Hit invalid token \"" 
					<< lexer.YYText() << "\"\n";
				return false;
				
			}
				
			second.push_back( lexer.YYText() );
			tokenValue = lexer.yylex();
		
		}
		
		if( first.size() != second.size() )
		{
		
			status << "First pass scanned " << first.size() 
				<< " tokens while second scanned " << second.size() << "\n";
			return false;
		
		}
		
		for( StringVector::iterator fi = first.begin(), si = second.begin(); 
			fi != first.end() && si != second.end(); ++fi, ++si )
		{
		
			if( *fi != *si )
			{
			
				status << "At index " << ( fi - first.begin() ) 
					<< ", first token scanned \"" << *fi 
					<< "\" did not match second \"" << *si <<  "\"\n";
				return false;
			
			}
		
		}
		
		report( " Scanned " << first.size() << " tokens." );
		
		return true;
	
	}

	bool TestLexer::doTest( )
	{

		StringVector files = _getFileNames();
		
		report( "Scanning the following files:\n " 
			<< hydrazine::toString( files.begin(), files.end(), "\n " )  );
		
		for( StringVector::iterator fi = files.begin(); 
			fi != files.end(); ++fi )
		{
		
			ptxFile = *fi;
		
			if(  !_testScan( ) )
			{
		
				status << "For file " << ptxFile 
					<< ", Test Point 1 (Scan): Failed\n";
				return false;
		
			}
		
			status << "For file " << ptxFile 
				<< ", Test Point 1 (Scan): Passed\n";
			
		}
			
		return true;
	
	}

	TestLexer::TestLexer()
	{
	
		name = "TestLexer";
		
		description = "Tests for the PTX lexer. Test Point 1: Scan a PTX file";
		description += " and write out a temp stream, scan the stream again ";
		description += "and make sure that the two sets of tokens match";
	
	}

}

int main( int argc, char** argv )
{

	hydrazine::ArgumentParser parser( argc, argv );
	test::TestLexer test;
	parser.description( test.testDescription() );

	parser.parse( "-i", "--input-file", test.input, "../tests/ptx",
		"Input directory to search for ptx files." );
	parser.parse( "-r", "--not-recursive", test.recursive, false,
		"Dont recursively search directories.");
	parser.parse( "-s", test.seed, 0,
		"Set the random seed, 0 implies seed with time." );
	parser.parse( "-v", test.verbose, false, "Print out info after the test." );
	parser.parse();

	test.recursive = !test.recursive;

	test.test();

	return test.passed();

}

#endif

