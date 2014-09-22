/*! \file PtxToIlTranslator.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date August 9, 2010
 *  \brief The implementation file for the PtxToIlTranslator class.
 */

#include <hydrazine/interface/Version.h>
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/debug.h>
#include <ocelot/translator/test/PtxToIlTranslator.h>
#include <ocelot/translator/interface/PTXToILTranslator.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/ILKernel.h>
#include <ocelot/executive/interface/ATIExecutableKernel.h>
#include <fstream>

void PtxToIlTranslator::translate()
{
	assertM( !input.empty(), "Input PTX file is required for translation." );
	
	translator::PTXToILTranslator translator;
	ir::Module module( input );

	ir::Module::KernelMap::const_iterator 
		k_it = module.kernels().begin();

	for (; k_it != module.kernels().end(); ++k_it) {

		ir::PTXKernel* ptx = dynamic_cast<ir::PTXKernel*>(k_it->second);
		ir::PTXKernel::assignRegisters(*ptx->cfg());

		executive::ATIExecutableKernel* kernel =
			new executive::ATIExecutableKernel(*k_it->second, NULL, NULL, NULL, 
					NULL, NULL, NULL);
		kernel->updateGlobals();

		std::string il = kernel->name + ".il";

		ir::ILKernel* translatedKernel = dynamic_cast< ir::ILKernel* >( 
			translator.translate( kernel ) );
		translatedKernel->assemble();
		std::ofstream ilFile( il.c_str() );
		
		hydrazine::Version version;
		
		ilFile << "; Kernel: " << kernel->name << "\n";
		ilFile << "; Translated from PTX to IL by Ocelot " 
			<< version.toString() << " \n";		
		ilFile << translatedKernel->code();
	}	
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	parser.description( 
		std::string( "A front end for the PTX to IL JIT compiler." ) 
		+ " Translates single .ptx files to .il files." );
	PtxToIlTranslator translator;
	
	parser.parse( "-p", "--ptx-file", translator.input, "", 
		"The input PTX file being translated.");
	parser.parse();

	translator.translate();
}
