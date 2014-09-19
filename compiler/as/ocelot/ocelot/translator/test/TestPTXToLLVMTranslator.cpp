/*!
	\file TestPTXToLLVMTranslator.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Sunday August 16, 2009
	\brief A test for the PTXToLLVMTranslator class.
*/

#ifndef TEST_PTX_TO_LLVM_TRANSLATOR_CPP_INCLUDED
#define TEST_PTX_TO_LLVM_TRANSLATOR_CPP_INCLUDED

#include "boost/filesystem.hpp"
#include <queue>
#include <fstream>

#include <ocelot/translator/interface/PTXToLLVMTranslator.h>
#include <ocelot/translator/test/TestPTXToLLVMTranslator.h>

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/LLVMKernel.h>

#include <ocelot/transforms/interface/RemoveBarrierPass.h>
#include <ocelot/transforms/interface/ConvertPredicationToSelectPass.h>
#include <ocelot/transforms/interface/PassManager.h>

#include <ocelot/parser/interface/PTXParser.h>

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
	TestPTXToLLVMTranslator::StringVector 
		TestPTXToLLVMTranslator::_getFileNames() const
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

	bool TestPTXToLLVMTranslator::_testTranslate()
	{
		report( " Loading file " << ptxFile );
		
		ir::Module module;
		
		try 
		{
			module.load( ptxFile );
		}
		catch(parser::PTXParser::Exception& e)
		{
			if(e.error == parser::PTXParser::State::NotVersion2_1)
			{
				status << "Skipping file with incompatible ptx version." 
					<< std::endl;
				return true;
			}
			status << "Load module failed with exception: " 
				<< e.what() << std::endl;
			return false;
		}
		
		report( " Translating file " << ptxFile );
		ir::Module::KernelMap::const_iterator 
			k_it = module.kernels().begin();

		for (; k_it != module.kernels().end(); ++k_it) 
		{
			ir::IRKernel* kernel = module.getKernel( k_it->first );

			transforms::PassManager manager(&module);

			transforms::ConvertPredicationToSelectPass pass1;
			transforms::RemoveBarrierPass pass2;
			translator::PTXToLLVMTranslator translator;

			manager.addPass(&pass1);
			manager.addPass(&pass2);

			manager.runOnKernel(*kernel);
			manager.releasePasses();
			
			manager.addPass(&translator);
			manager.runOnKernel(*kernel);
			manager.releasePasses();

			ir::LLVMKernel* translatedKernel = dynamic_cast< ir::LLVMKernel* >( 
				translator.translatedKernel() );
			translatedKernel->assemble();
			
			std::string outputFile = ptxFile + "." + kernel->name + ".ll";
		
			if( output )
			{
				std::ofstream outFile( outputFile.c_str() );
				outFile << translatedKernel->code();
				outFile << "\n";
				outFile.close();
			}
			
			delete translatedKernel;

		}
		
		return true;	
	}
	
	bool TestPTXToLLVMTranslator::doTest()
	{
		StringVector files = _getFileNames();
		
		hydrazine::Timer timer;
		timer.start();
	
		unsigned int count = 0;
	
		report( "Translating the following files:\n " 
			<< hydrazine::toString( files.begin(), files.end(), "\n " )  );
		
		for( unsigned int i = 0, e = files.size(); i != e; ++i )
		{	
			if( timer.seconds() > timeLimit ) break;

			unsigned int index = random() % files.size();
	
			ptxFile = files[ index ];
		
			if(  !_testTranslate( ) )
			{
				status << "For file " << ptxFile 
					<< ", Test Point 1 (Translate): Failed\n";
				return false;
			}
		
			status << "For file " << ptxFile 
				<< ", Test Point 1 (Translate): Passed\n";
				
			++count;
		}
	
		status << "Finished running " << count << " tests...\n";
			
		return true;	
	}

	TestPTXToLLVMTranslator::TestPTXToLLVMTranslator()
	{
		name = "TestPTXToLLVMTranslator";
	
		description = "This is a basic test that just tries to get through a";
		description += " translation successfully of as many PTX programs as";
		description += " possible Test Points: 1) Scan for all PTX files in a";
		description += " directory, try to translate them.";
	}
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestPTXToLLVMTranslator test;
	parser.description( test.testDescription() );

	parser.parse( "-i", test.input, "../tests/ptx",
		"Input directory to search for ptx files." );
	parser.parse( "-r", test.recursive, true, 
		"Recursively search directories.");
	parser.parse( "-o", test.output, false,
		"Print out the internal representation of each parsed file." );
	parser.parse("-l", "--time-limit", test.timeLimit, 60, 
		"How many seconds to run tests.");
	parser.parse( "-s", test.seed, 0,
		"Set the random seed, 0 implies seed with time." );
	parser.parse( "-v", test.verbose, false, "Print out info after the test." );
	parser.parse();
	
	test.test();

	return test.passed();
}

#endif

