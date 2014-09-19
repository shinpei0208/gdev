/*! \file MemoryChecker.h
	\date Wednesday March 17, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the MemoryChecker class.
*/

#ifndef MEMORY_CHECKER_H_INCLUDED
#define MEMORY_CHECKER_H_INCLUDED

// Ocelot includes
#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/Dim3.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <iostream>
#include <sstream>

namespace executive
{
	class Device;
	class EmulatedKernel;
}

namespace trace
{
	/*! \brief A trace generator for checking all memory accesses */
	class MemoryChecker : public TraceGenerator
	{
		public:
			/*! \brief A class for a cached memory allocation */
			class Allocation
			{
				public:
					bool valid; // is this a valid allocation?
					ir::PTXU64 base; // base allocation pointer
					ir::PTXU64 extent; // size of the allocation
			
				public:
					Allocation(bool valid = false, 
						ir::PTXU64 base = 0, ir::PTXU64 extent = 0);
			};

			enum Status {
				INVALID,
				NOT_DEFINED,
				PARTIALLY_DEFINED,
				DEFINED
			};

			class ShadowMemory
			{
			    public:
        			/*! \brief Distinguished Secondary Mapping for shadow memory */
        			std::vector <Status> map;

			    public:
			        ShadowMemory();

			        void resize(unsigned int size);

			        unsigned int size();

					/*! \brief Check if region is initialized */
			        Status checkRegion(unsigned int idx, unsigned int size);
				
					/*! \brief Set initialization status of a region */
					void setRegion(unsigned int idx, unsigned int size, Status stat);

					/*! \brief Check if region is initialized */
					Status checkRegister(ir::PTXOperand::RegisterType idx);

					/*! \brief Set initialization status of a register */
					void setRegister(ir::PTXOperand::RegisterType idx, Status stat);
					    
			};
			
		private:
			/*! \brief The block dimensions */
			ir::Dim3 _dim;
		
			/*! \brief The last allocation referenced */
			Allocation _cache;
			
			/*! \brief Parameter memory allocation */
			Allocation _parameter;
			
			/*! \brief Shared memory allocation */
			Allocation _shared;
			
			/*! \brief Local memory allocation */
			Allocation _local;
			
			/*! \brief Global local memory allocation */
			Allocation _globalLocal;
			
			/*! \brief Constant memory allocation */
			Allocation _constant;
			
			/*! \brief A pointer to the executive class */
			const executive::Device* _device;
		
			/*! \brief A pointer to the executable kernel */
			const executive::EmulatedKernel* _kernel;

			/*! \brief Flag to toggle initialization checks */
			bool checkInitialization;

			/*! \brief Shadow maps for checking uninitialized memory */
			ShadowMemory _globalShadow;		

			ShadowMemory _sharedShadow;	

			ShadowMemory _constShadow;

			ShadowMemory _localShadow;
		
			ShadowMemory _registerFileShadow;
			
		private:
			/*! \brief Check the alignment of a memory access */
			void _checkAlignment(const TraceEvent& e);
			
			/*! \brief Check whether or not the access falls within 
				an allocated region */
			void _checkValidity(const TraceEvent& e);
			
			/*! \brief Check for an uninitialized memory access */
			void _checkInitialized(const TraceEvent& e);

			/*! \brief Track initialization status of registers */
			void _checkInstructions(const TraceEvent& e);

			/*! \brief Track register-to-register status and control redirect */
			Status checkInstruction(const TraceEvent& e,
				bool useMemoryFlag=false, ShadowMemory *shadowMem=NULL);
			

			
		public:
			/*! \brief The constructor initializes the cached allocations */
			MemoryChecker();

			/*! \brief Set initialization checking toggle */
			void setCheckInitialization(bool toggle);
			
			/*! \brief Set the cache and get a pointer to the memory mappings */
			virtual void initialize(const executive::ExecutableKernel& kernel);

			/*! \brief Called whenever an event takes place.

				Note, the const reference 'event' is only valid until event() 
				returns
			*/
			virtual void event(const TraceEvent& event);
			
			virtual void postEvent(const TraceEvent& event);

			/*!  \brief Called when a kernel is finished. There will be no more 
					events for this kernel.
			*/
			virtual void finish();
		
	};
}

#endif

