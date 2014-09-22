/*! \file PTXOptimzer.h
	\date Thursday December 31, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Ocelot PTX optimizer
*/

#ifndef PTX_OPTIMIZER_H_INCLUDED
#define PTX_OPTIMIZER_H_INCLUDED

// Standard Library Includes
#include <string>
#include <set>

// Forward Declarations
namespace transforms { class Pass; }

namespace tools
{
	/*! \brief Able to run various optimization passes over PTX modules */
	class PTXOptimizer
	{
		public:
			typedef std::set<std::string> StringSet;
	
		public:
			/*! \brief The input file being optimized */
			std::string input;
			
			/*! \brief The output file being generated */
			std::string output;
			
			/*! \brief The type of register allocation to perform */
			std::string registerAllocationType;
			
			/*! \brief The set of passes to run */
			StringSet passes;
			
			/*! \brief The number of registers to allocate */
			unsigned int registerCount;
			
			/*! \brief The target sub-kernel size */
			unsigned int subkernelSize;
			
			/*! \brief The inlining threshold */
			unsigned int inliningThreshold;
			
			/*! \brief The target basic block size */
			unsigned int basicBlockSize;
			
			/*! \brief Print out the CFG of optimized kernels */
			bool cfg;
			
			/*! \brief Print out the CFG of optimized kernels */
			bool dfg;
			
		public:
			/*! \brief The constructor sets the defaults */
			PTXOptimizer();

			/*! \brief Performs the optimizations */
			void optimize();	
			
		public:
			/*! \brief Apply options to the pass */
			void applyOptions( transforms::Pass* pass );
	};
}

int main( int argc, char** argv );

#endif

