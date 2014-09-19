/*! 
	\file PTXEmitter.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date September 4, 2012
	\brief Controls how PTX is emitted
*/

#ifndef IR_PTXEMITTER_H_INCLUDED
#define IR_PTXEMITTER_H_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ir {

	/*!
	
	*/
	class PTXEmitter {
	public:
		/*! \brief Controls how PTX modules, kernels, and instructions are emitted */
		enum Target {
			Target_invalid,
			Target_OcelotIR,    //! invertible Ocelot PTX IR (parser <-> IR)
			Target_NVIDIA_PTX30	//! PTX emitter compatible with CUDA driver
		};
	};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

