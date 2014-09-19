/*!
	\file TestDataflowGraph.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday July 7, 2009
	\brief The source file for the TestDataflowGraph class.
*/

#ifndef TEST_DATAFLOW_GRAPH_CPP_INCLUDED
#define TEST_DATAFLOW_GRAPH_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/test/TestDataflowGraph.h>

#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/Module.h>

#include <ocelot/parser/interface/PTXParser.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// Boost Includes
#include <boost/filesystem.hpp>

// Standard Library Includes
#include <queue>


#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace fs = boost::filesystem;

namespace test
{
	class Double
	{
		public:
			unsigned int operator()( 
				analysis::DataflowGraph::RegisterPointerVector::const_iterator 
				it )
			{
				return *it->pointer;
			}
	};

	class ToId
	{
		public:
			unsigned int operator()( 
				analysis::DataflowGraph::RegisterVector::const_iterator it )
			{
				return it->id;
			}
	};

	bool TestDataflowGraph::_verify( const analysis::DataflowGraph& graph )
	{
		for( analysis::DataflowGraph::const_iterator block = graph.begin(); 
			block != graph.end(); ++block )
		{
			analysis::DataflowGraph::RegisterSet defined;
			
			report( " Alive in registers:" );
			for( analysis::DataflowGraph::RegisterSet::const_iterator 
				reg = block->aliveIn().begin(); 
				reg != block->aliveIn().end(); ++reg )
			{
				report( "  " << reg->id );
				defined.insert( *reg );
			}
				
			if( !block->phis().empty() )
			{
				status << "  Block " << block->label() 
					<< " has " << block->phis().size() 
					<< " phi instructions." << std::endl;
				return false;			
			}
			
			for( analysis::DataflowGraph::InstructionVector::const_iterator 
				instruction = block->instructions().begin(); 
				instruction != block->instructions().end(); ++instruction )
			{
				report( " " << instruction->i->toString() << ":  " 
					<< hydrazine::toFormattedString( instruction->d.begin(), 
					instruction->d.end(), Double() ) << " <- " 
					<< hydrazine::toFormattedString( instruction->s.begin(), 
					instruction->s.end(), Double() ) );
					
				analysis::DataflowGraph::RegisterPointerVector::const_iterator
					reg = instruction->s.begin();
				for( ; reg != instruction->s.end(); ++reg )
				{
					if( !defined.count( *reg ) )
					{
						status << "  Register " << *reg->pointer 
							<< " in instruction " << instruction->i->toString()
							<< " in block " << block->label() 
							<< " used uninitialized." << std::endl;
						return false;
					}
				}
				
				reg = instruction->d.begin();
					
				for( ; reg != instruction->d.end(); ++reg )
				{
					defined.insert( *reg );
				}
			}
			
			for( analysis::DataflowGraph::RegisterSet::const_iterator 
				reg = block->aliveOut().begin(); 
				reg != block->aliveOut().end(); ++reg )
			{
				if( !defined.count( *reg ) )
				{
					status << "  Register " << reg->id 
						<< " out set of block " << block->label() 
						<< " used uninitialized." << std::endl;
					return false;
				}
			}			
		}
		return true;
	}

	bool TestDataflowGraph::_verifySsa( const analysis::DataflowGraph& graph )
	{
		analysis::DataflowGraph::RegisterSet global;
		for( analysis::DataflowGraph::const_iterator block = graph.begin(); 
			block != graph.end(); ++block )
		{
			analysis::DataflowGraph::RegisterSet defined;
			
			report( block->label() );
			report( " Alive in registers:" );
			for( analysis::DataflowGraph::RegisterSet::const_iterator 
				reg = block->aliveIn().begin(); 
				reg != block->aliveIn().end(); ++reg )
			{
				report( "  " << reg->id );
				defined.insert( *reg );
			}
			
			for( analysis::DataflowGraph::PhiInstructionVector::const_iterator 
				phi = block->phis().begin(); 
				phi != block->phis().end(); ++phi )
			{
				report( " phi " << phi->d.id << " <- " 
					<< hydrazine::toFormattedString( phi->s.begin(), 
						phi->s.end(), ToId() ) );
				for( analysis::DataflowGraph::RegisterVector::const_iterator
					reg = phi->s.begin(); reg != phi->s.end(); ++reg )
				{
					if( !defined.count( *reg ) )
					{
						status << "  Register " << reg->id
							<< " in phi instruction " 
							<< std::distance( block->phis().begin(), phi )
							<< " in " << block->label() 
							<< " used uninitialized." << std::endl;
						return false;
					}
				}
				defined.insert( phi->d );
				if( !global.insert( phi->d ).second )
				{
					status << "  In " << block->label() 
						<< ", instruction phi " 
						<< std::distance( block->phis().begin(), phi )
						<< ", reg " << phi->d.id
						<< " already defined globally." << std::endl;
					return false;
				}
			}
			
			for( analysis::DataflowGraph::InstructionVector::const_iterator 
				instruction = block->instructions().begin(); 
				instruction != block->instructions().end(); ++instruction )
			{
				report( " " << instruction->i->toString() << ":  " 
					<< hydrazine::toFormattedString( instruction->d.begin(), 
					instruction->d.end(), Double() ) << " <- " 
					<< hydrazine::toFormattedString( instruction->s.begin(), 
					instruction->s.end(), Double() ) );
					
				analysis::DataflowGraph::RegisterPointerVector::const_iterator
					reg = instruction->s.begin();
				for( ; reg != instruction->s.end(); ++reg )
				{
					if( !defined.count( *reg ) )
					{
						status << "  Register " << *reg->pointer 
							<< " in instruction " << instruction->i->toString() 
							<< " in " << block->label() 
							<< " used uninitialized." << std::endl;
						return false;
					}
				}
				
				for( reg = instruction->d.begin(); 
					reg != instruction->d.end(); ++reg )
				{
					defined.insert( *reg );
					if( !global.insert( *reg ).second )
					{
						status << "  In " << block->label() 
							<< ", instruction " 
							<< instruction->i->toString() << ", reg "
							<< *reg->pointer
							<< " already defined globally." << std::endl;
						return false;
					}
				}
			}
			
			for( analysis::DataflowGraph::RegisterSet::const_iterator 
				reg = block->aliveOut().begin(); 
				reg != block->aliveOut().end(); ++reg )
			{
				if( !defined.count( *reg ) )
				{
					status << "  Register " << reg->id 
						<< " out set of block " << block->label() 
						<< " used uninitialized." << std::endl;
					return false;
				}
			}			
		}
		return true;
	}

	void TestDataflowGraph::_getFileNames()
	{
		fs::path path = base;
		_files.clear();
		
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
					if( fs::is_directory( file->status() ) )
					{
						directories.push( file->path() );
					}
					else if( fs::is_regular_file( file->status() ) )
					{
						if( file->path().extension() == ".ptx" )
						{
							_files.push_back( file->path().string() );
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
				_files.push_back( path.string() );
			}
		}
	}
	
	bool TestDataflowGraph::_testGeneric()
	{
		status << "Testing Generic Dataflow" << std::endl;
		
		hydrazine::Timer timer;
		timer.start();
	
		StringVector usedFiles;
		
		for( unsigned int i = 0, e = _files.size(); i != e; ++i )
		{	
			if( timer.seconds() > timeLimit ) break;

			unsigned int index = random() % _files.size();
	
			std::string file = _files[ index ];
		
			status << " For File: " << file << std::endl;

			ir::Module module;
			try 
			{
				module.load( file );
			}
			catch(parser::PTXParser::Exception& e)
			{
				if(e.error == parser::PTXParser::State::NotVersion2_1)
				{
					status << "  Skipping file with incompatible ptx version." 
						<< std::endl;
					continue;
				}
				status << "Load module failed with exception: " 
					<< e.what() << std::endl;
				return false;
			}

			for( ir::Module::KernelMap::const_iterator 
				ki = module.kernels().begin(); 
				ki != module.kernels().end(); ++ki )
			{
				ir::PTXKernel& kernel = static_cast< ir::PTXKernel& >( 
					*module.getKernel( ki->first ) );
				status << "  For Kernel: " << kernel.name << std::endl;
				
				analysis::DataflowGraph dfg;
				
				dfg.analyze( kernel );
					
				if( !_verify( dfg ) )
				{
					return false;
				}
			}
			
			usedFiles.push_back( file );
			
		}
		
		_files = usedFiles;
		
		status << " Test Passed" << std::endl;
		return true;
	}
	
	bool TestDataflowGraph::_testSsa()
	{
		status << "Testing SSA Dataflow" << std::endl;
		for( StringVector::const_iterator file = _files.begin(); 
			file != _files.end(); ++file )
		{
			status << " For File: " << *file << std::endl;
			ir::Module module;
			try 
			{
				module.load( *file );
			}
			catch(parser::PTXParser::Exception& e)
			{
				if(e.error == parser::PTXParser::State::NotVersion2_1)
				{
					status << "  Skipping file with incompatible ptx version." 
						<< std::endl;
					continue;
				}
				status << "Load module failed with exception: " 
					<< e.what() << std::endl;
				return false;
			}
			
			for( ir::Module::KernelMap::const_iterator 
				ki = module.kernels().begin(); 
				ki != module.kernels().end(); ++ki )
			{
				ir::PTXKernel& kernel = static_cast< ir::PTXKernel& >( 
					*module.getKernel( ki->first ) );
				status << "  For Kernel: " << kernel.name << std::endl;
				analysis::DataflowGraph dfg;
				
				dfg.analyze( kernel );
				dfg.toSsa();
				
				if( !_verifySsa( dfg ) )
				{
					return false;
				}
			}
		}
		status << " Test Passed" << std::endl;
		return true;
	}
	
	bool TestDataflowGraph::_testReverseSsa()
	{
		status << "Testing SSA then back Dataflow" << std::endl;
		for( StringVector::const_iterator file = _files.begin(); 
			file != _files.end(); ++file )
		{
			status << " For File: " << *file << std::endl;
			ir::Module module;
			try 
			{
				module.load( *file );
			}
			catch(parser::PTXParser::Exception& e)
			{
				if(e.error == parser::PTXParser::State::NotVersion2_1)
				{
					status << "  Skipping file with incompatible ptx version." 
						<< std::endl;
					continue;
				}
				status << "Load module failed with exception: " 
					<< e.what() << std::endl;
				return false;
			}
			
			for( ir::Module::KernelMap::const_iterator 
				ki = module.kernels().begin(); 
				ki != module.kernels().end(); ++ki )
			{
				ir::PTXKernel& kernel = static_cast< ir::PTXKernel& >( 
					*module.getKernel( ki->first ) );
				status << "  For Kernel: " << kernel.name << std::endl;
				analysis::DataflowGraph dfg;
				
				dfg.analyze( kernel );
				dfg.toSsa();
				dfg.fromSsa();

				if( !_verify( dfg ) )
				{
					return false;
				}
			}
		}
		status << " Test Passed" << std::endl;
		return true;	
	}
	
	bool TestDataflowGraph::doTest()
	{
		_getFileNames();
		return _testGeneric() && _testSsa() && _testReverseSsa();
	}
	
	TestDataflowGraph::TestDataflowGraph()
	{
		name  = "TestDataflowGraph";
		
		description = "A test for the DataflowGraph class. Test Points: 1) ";
		description += "Generic: load PTX files, convert them into dataflow";
		description += " graphs, verify that all live ranges spanning blocks";
		description += " are consistent. 2) SSA: convert to ssa form, verify";
		description += " that no register is declared more than once. 3)";
		description += " reverse SSA: convert to ssa then out of ssa, verify";
		description += " that all live ranges spanning blocks are consistent.";
	}
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestDataflowGraph test;
	parser.description( test.testDescription() );

	parser.parse( "-i", "--input", test.base, "../tests/ptx", 
		"Search path for PTX files." );
	parser.parse("-l", "--time-limit", test.timeLimit, 60, 
		"How many seconds to run tests.");
	parser.parse( "-s", test.seed, 0,
		"Set the random seed, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out info after the test." );
	parser.parse();
	
	test.test();
	
	return test.passed();	
}

#endif

