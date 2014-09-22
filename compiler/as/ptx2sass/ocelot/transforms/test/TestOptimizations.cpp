/*! \file TestOptimizations.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday May 1, 2012
	\brief The source file for the TestPTXAssembly class.
*/

// Ocelot Includes
#include <ocelot/transforms/test/TestOptimizations.h>

#include <ocelot/transforms/interface/PassFactory.h>
#include <ocelot/transforms/interface/PassManager.h>

#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/string.h>

// Boost Includes
#include <boost/filesystem.hpp>

// Standard Library Includes
#include <deque>
#include <queue>

namespace fs = boost::filesystem;

namespace test
{

typedef hydrazine::StringVector StringVector;

StringVector getPTXFiles(const std::string& base)
{
	fs::path path = base;
	
	StringVector files;
	
	if(fs::is_directory(path))
	{
		std::queue<fs::path> directories;
		directories.push(path);
		
		fs::directory_iterator end;
		
		while(!directories.empty())
		{
			for(fs::directory_iterator file(directories.front()); 
				file != end; ++file)
			{
				if(fs::is_directory(file->status()))
				{
					directories.push(file->path());
				}
				else if(fs::is_regular_file(file->status()))
				{
					if(file->path().extension() == ".ptx")
					{
						files.push_back(file->path().string());
					}
				}
			}
			
			directories.pop();
		}
	}
	else if(fs::is_regular_file(path))
	{
		if(path.extension() == ".ptx")
		{
			files.push_back(path.string());
		}
	}
	
	return files;
}

static StringVector getOptimizations(const std::string& optimizationList)
{
	StringVector passes = hydrazine::split(optimizationList, ",");
	StringVector splitPasses;
	
	splitPasses.reserve(passes.size());
	
	for(hydrazine::StringVector::iterator pass = passes.begin(); 
		pass != passes.end(); ++pass)
	{
		splitPasses.push_back(hydrazine::strip(*pass, " "));
	}
	
	return splitPasses;
}

static StringVector enumerateTests(const StringVector& files,
	const StringVector& passes)
{
	StringVector tests;
	
	tests.reserve(files.size() * passes.size());
	
	for(auto file = files.begin(); file != files.end(); ++file)
	{
		for(auto pass = passes.begin(); pass != passes.end(); ++pass)
		{
			tests.push_back(*file + "|" + *pass);
		}
	}
	
	return tests;
}

std::string getName(const std::string& test)
{
	return hydrazine::split(test, "|")[0];
}

transforms::Pass* getPass(const std::string& test)
{
	std::string passName = hydrazine::split(test, "|")[1];
	
	return transforms::PassFactory::createPass(passName);
}

TestOptimizations::TestOptimizations()
{
	name = "TestOptimizations";
		
	description =  "A unit test framework for PTX optimizations. Runs random ";
	description += "optimization passes on random PTX files until ";
	description += "a timer expires.";
}

bool TestOptimizations::runTest(const std::string& test)
{
	std::string ptxFileName = getName(test);

	try
	{
		ir::Module module(ptxFileName);
	
		transforms::PassManager manager(&module);
	
		manager.addPass(getPass(test));

		manager.runOnModule();

		manager.releasePasses();
	}
	catch(const std::exception& e)
	{
		status << "Test " << test << " failed: " << e.what() << "\n";
		return false;
	}
	catch(...)
	{
		status << "Test " << test << " failed: Cause unknown\n";
		return false;
	}
	
	return true;
}

bool TestOptimizations::doTest()
{
	typedef std::deque<unsigned int> TestSet;
	
	StringVector testVector = enumerateTests(getPTXFiles(path),
		getOptimizations(optimizations));
	
	status << " Enumerated " << testVector.size() << " tests\n";
	
	TestSet tests;
	
	for(auto test = testVector.begin(); test != testVector.end(); ++test)
	{
		tests.push_back(std::distance(testVector.begin(), test));
	}
	
	hydrazine::Timer timer;
	timer.start();
	
	unsigned int count = 0;
	
	for(unsigned int i = 0, e = tests.size(); i != e; ++i)
	{
		if(timer.seconds() > timeLimit) break;
		
		unsigned int index = random() % tests.size();
	
		TestSet::iterator testPosition = tests.begin() + index;
	
		std::string test = testVector[*testPosition];
	
		status << " Running test '" << test << "'\n";
	
		if(!runTest(test)) return false;
	
		tests.erase(testPosition);
	
		++count;
	}
	
	status << "Finished running " << count << " tests...\n";
	
	return true;
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	test::TestOptimizations test;
	parser.description(test.testDescription());

	parser.parse("-v", "--verbose", test.verbose, false,
		"Print out status info after the test.");
	parser.parse("-V", "--very-verbose", test.veryVerbose, false,
		"Print out information as the test is running.");
	parser.parse("-e", "--enumerate", test.enumerate, false,
		"Only enumerate tests, do not run them.");
	parser.parse("-s", "--seed", test.seed, 0,
		"Random seed for generating input data. 0 implies seed with time.");
	parser.parse("-l", "--time-limit", test.timeLimit, 60, 
		"How many seconds to run tests.");
	parser.parse( "-i", "--input", test.path, "../tests/ptx", 
		"Search path for PTX files." );
	parser.parse( "-O", "--optimizations", test.optimizations, 
		"dead-code-elimination, loop-unrolling, function-inlining, "
		"global-value-numbering, constant-propagation, hoist-parameters,"
		"move-elimination", 
		"Comma separated list of optimizations to run." );
	parser.parse();

	test.test();
	
	return test.passed();
}


