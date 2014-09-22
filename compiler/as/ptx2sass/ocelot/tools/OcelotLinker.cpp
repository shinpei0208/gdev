/*! \file   OcelotLinker.cpp
	\date   Tuesday December 4, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Ocelot PTX linker
*/

// Ocelot Includes
#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/transforms/interface/PassFactory.h>

#include <ocelot/transforms/interface/ModuleLinkerPass.h>

#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/SystemCompatibility.h>

// Standard Library Includes
#include <fstream>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace tools
{

typedef std::set<std::string> StringSet;

static StringSet parseInputList(const std::string& inputList)
{
	report("Checking for input files.");
	
	auto inputs = hydrazine::split(inputList, ",");
	
	if(inputList.find(",") == std::string::npos)
	{
		inputs = hydrazine::split(inputList, " ");
	}
	
	for(auto input = inputs.begin(); input != inputs.end(); ++input)
	{
		*input = hydrazine::strip(*input, " ");
	}
	
	return StringSet(inputs.begin(), inputs.end());
}

int link(const std::string& inputList, const std::string& output,
	bool reportUndefinedSymbols)
{		
	report("Running PTX to PTX Linker.");
	
	auto inputs = parseInputList(inputList);
	
	transforms::ModuleLinkerPass linkerPass;
		
	if(inputs.empty())
	{
		std::cout << "No input file name given.  Bailing out.\n";
		return -1;
	}
	
	for(auto input = inputs.begin(); input != inputs.end(); ++input)
	{
		report(" Loading module '" << *input << "'");
		ir::Module module(*input);

		transforms::PassManager manager(&module);
		
		manager.addPass(&linkerPass);

		try
		{
			manager.runOnModule();
			report("  linked it'");
		}
		catch(const std::exception& e)
		{
			std::cout << "Linker Error: link failed for module '"
				<< *input << "'.\n";
			std::cout << " Message: " << e.what() << ".\n";

			manager.clear();

			return -2;
		}
		
		manager.clear();
	}

	std::ofstream out(output.c_str());
	
	if(!out.is_open())
	{
		std::cout << "Could not open output file '"
			<< output << "' for writing.\n";
		return -3;
	}
	
	linkerPass.linkedModule()->writeIR(out);
	
	if(reportUndefinedSymbols)
	{
		auto symbols = linkerPass.getAllUndefinedSymbols();
		
		for(auto symbol = symbols.begin(); symbol != symbols.end(); ++symbol)
		{
			std::cout << "Undefined symbol: '";
			
			if(hydrazine::isMangledCXXString(*symbol))
			{
				std::cout << hydrazine::demangleCXXString(*symbol); 
			}
			else
			{
				std::cout << *symbol;
			}
			
			std::cout << "'\n";
		}
		
		if(!symbols.empty())
		{
			return -1;
		}
	}
	
	return 0;
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	parser.description("The Ocelot PTX to PTX linker.");
	
	std::string inputs;
	std::string output;
	bool reportUndefinedSymbols = false;
	
	parser.parse( "-i", "--input", inputs, "",
		"The ptx files to be linked." );
	parser.parse( "-r", "--report-undefined-symbols", reportUndefinedSymbols,
		false, "." );
	parser.parse( "-o", "--output", output, 
		"", "The resulting linked file." );
	
	parser.parse();
		
	return tools::link(inputs, output, reportUndefinedSymbols);
}


