/*! \file Translator.h
	\date Wednesday July 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Translator class
*/

#ifndef TRANSLATOR_H_INCLUDED
#define TRANSLATOR_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/Instruction.h>

#include <ocelot/transforms/interface/Pass.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>

namespace ir { class Kernel; }

/*! \brief A namespace for IR to IR translators */
namespace translator
{

/*! \brief An interface for a translator from one ISA to another */
class Translator : public transforms::ImmutableKernelPass
{
public:
	/*! \brief An exception for distinguishing between exceptions */
	class Exception : public hydrazine::Exception
	{
		public:
			Exception( const std::string& message );
	};

	/*! \brief All possible optimization levels */
	enum OptimizationLevel
	{
		NoOptimization = 0x0,
		ReportOptimization,
		DebugOptimization,
		InstrumentOptimization,
		MemoryCheckOptimization,
		BasicOptimization,
		AggressiveOptimization,
		SpaceOptimization,
		FullOptimization
	};

protected:
	/*! \brief The optimization level for all translations */
	OptimizationLevel optimizationLevel;
	
	/*! \brief The source architecture to be translated from */
	ir::Instruction::Architecture sourceArchitecture;
	
	/*! \brief The target architecture to translate to */
	ir::Instruction::Architecture targetArchitecture;

public:
	static std::string toString( OptimizationLevel level );

public:
	/*! \brief The constructor */
	Translator( ir::Instruction::Architecture s, 
		ir::Instruction::Architecture t, 
		OptimizationLevel l = NoOptimization, 
		const StringVector& analyses = StringVector(),
		const std::string& name = "" );
	/*! \brief The destructor is needed for virtual methods */
	virtual ~Translator();
	
public:
	/*! \brief Translate a kernel from the source ISA to the destination
			ISA
	
		\param i The input kernel. Must be in the source ISA.
		\return A newly allocated kernel that is functionally 
			equivalent to i in the target ISA.
	
		Note that this will allocate a new kernel and return it to the
		caller.  It is the caller's resposibility to free it.
	*/
	virtual ir::Kernel* translate( const ir::Kernel* i ) = 0;

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize( const ir::Module& m );
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel( const ir::IRKernel& k );		
	/*! \brief Finalize the pass */
	void finalize();

};

}

#endif

