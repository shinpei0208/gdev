/*! \file ExecutableKernel.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 19, 2009
	\brief implements a kernel that is executable on some device
*/

#ifndef EXECUTABLE_KERNEL_H_INCLUDED
#define EXECUTABLE_KERNEL_H_INCLUDED

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/Texture.h>
#include <ocelot/ir/interface/Dim3.h>
#include <ocelot/ir/interface/ExternalFunctionSet.h>

namespace executive {
	class Device;
}

namespace trace {
	class TraceEvent;
	class TraceGenerator;
}

namespace executive {
	class ExecutableKernel : public ir::IRKernel {
	public:
		typedef std::vector< trace::TraceGenerator* > TraceGeneratorVector;
		typedef std::vector< const ir::Texture* > TextureVector;
		
		enum CacheConfiguration {
			CacheConfigurationDefault,
			CachePreferShared,
			CachePreferL1,
			CacheConfiguration_invalid
		};

	public:
		executive::Device* device;

	public:
		ExecutableKernel(const ir::IRKernel& k, 
			executive::Device* d = 0);
		ExecutableKernel(executive::Device* d = 0);
		virtual ~ExecutableKernel();
	
		/*!	\brief Determines whether kernel is executable */
		virtual bool executable() const;
		
		/*!	\brief Launch a kernel on a 2D grid */
		virtual void launchGrid(int width, int height, int depth)=0;

		/*!
			\brief compute argument offsets for argument data
			\return number of bytes required for argument memory
		*/
		virtual size_t mapArgumentOffsets();

		/*!
			\brief given a block of argument memory, sets the values of 
				each argument
			\param argument pointer to argument memory
			\param argumentSize number of bytes to write to argument memory
		*/
		virtual void setArgumentBlock(const unsigned char *argument, 
			size_t argumentSize);

		/*!
			\brief gets the values of each argument as a block of binary data
			\param argument pointer to argument memory
			\param maxSize maximum number of bytes to write to argument memory
			\return actual number of bytes required by argument memory
		*/
		virtual size_t getArgumentBlock(unsigned char *argument,
			size_t maxSize) const;
	
		/*!	\brief Sets the shape of a kernel */
		virtual void setKernelShape(int x, int y, int z)=0;

		/*! \brief Changes the amount of external shared memory */
		virtual void setExternSharedMemorySize(unsigned int)=0;
		
		/*! \brief sets the cache configuration of the kernele */
		virtual void setCacheConfiguration(CacheConfiguration config);
		
		/*! \brief sets the cache configuration of the kernele */
		virtual CacheConfiguration getCacheConfiguration() const;
		
		/*! \brief Sets the max number of pthreads this kernel can use */
		virtual void setWorkerThreads(unsigned int workerThreadLimit)=0;
			
		/*! \brief Indicate that the kernels arguments have been updated */
		virtual void updateArgumentMemory()=0;
		
		/*! \brief Indicate that other memory has been updated */
		virtual void updateMemory()=0;
		
		/*! \brief Get a vector of all textures references by the kernel */
		virtual TextureVector textureReferences() const=0;

		/*!	Notifies all attached TraceGenerators of an event */
		void traceEvent(const trace::TraceEvent & event) const;

		/*!	Notifies all attached TraceGenerators of completion of an event */
		void tracePostEvent(const trace::TraceEvent & event) const;
		
		virtual void setTraceGenerators(const TraceGeneratorVector &traceGenerators);
		
		/*!	adds a trace generator to the EmulatedKernel */
		virtual void addTraceGenerator(trace::TraceGenerator* generator);

		/*!	removes a trace generator from an EmulatedKernel */
		virtual void removeTraceGenerator(trace::TraceGenerator* generator);

		/*! sets an external function table for the emulated kernel */
		virtual void setExternalFunctionSet(
			const ir::ExternalFunctionSet& s) = 0;
		
		/*! clear the external function table for the emulated kernel */
		virtual void clearExternalFunctionSet() = 0;

		/*! Find an external function */
		ir::ExternalFunctionSet::ExternalFunction* findExternalFunction(
			const std::string& name) const;

	public:
		/*! attribute accessors - things every executable kernel should know */
		unsigned int constMemorySize() const;
		unsigned int localMemorySize() const;
		unsigned int globalLocalMemorySize() const;
		unsigned int maxThreadsPerBlock() const;
		unsigned int registerCount() const;
		unsigned int sharedMemorySize() const;
		unsigned int externSharedMemorySize() const;
		unsigned int totalSharedMemorySize() const;
		unsigned int argumentMemorySize() const;
		unsigned int parameterMemorySize() const;
		const ir::Dim3& blockDim() const;
		const ir::Dim3& gridDim() const;
		
	protected:
	
		void initializeTraceGenerators();
		void finalizeTraceGenerators();

	protected:
		/*! \brief Total amount of allocated constant memory size */
		unsigned int _constMemorySize;
		/*! \brief Total amount of allocated local memory size */
		unsigned int _localMemorySize;
		/*! \brief Total amount of allocated global local memory per thread */
		unsigned int _globalLocalMemorySize;
		/*! \brief Maxmimum number of threads launched per block */
		unsigned int _maxThreadsPerBlock;
		/*! \brief Number of registered required by each thread */
		unsigned int _registerCount;
		/*! \brief The amount of allocated static shared memory */
		unsigned int _sharedMemorySize;
		/*! \brief The amount of allocated dynamic shared memory */
		unsigned int _externSharedMemorySize;
		/*! \brief Total amount of packed parameter memory */
		unsigned int _argumentMemorySize;
        /*! \brief Kernel stack parameter memory space */
        unsigned int _parameterMemorySize;
		/*! \brief The block dimensions */
		ir::Dim3 _blockDim;
		/*!	\brief Dimension of grid in blocks */
		ir::Dim3 _gridDim;
		/*! \brief Attached trace generators */
		TraceGeneratorVector _generators;
		/*! \brief Registered external functions */
		const ir::ExternalFunctionSet* _externals;
		/*! \brief configuration of cache */
		CacheConfiguration _cacheConfiguration;
	};
	
}

#endif
