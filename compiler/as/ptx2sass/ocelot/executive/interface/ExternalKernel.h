/*! \file ExternalKernel.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 28, 2010
	\brief interface for a kernel defined externally to Ocelot that is to be loaded at runtime
*/

#ifndef EXECUTIVE_EXTERNAL_KERNEL_H_INCLUDED
#define EXECUTIVE_EXTERNAL_KERNEL_H_INCLUDED

#include <ocelot/ir/interface/ExecutableKernel.h>

namespace executive {

	class LLVMContext;
	class EmulatedKernel;
	class LLVMExecutableKernel;

	/*!
		\brief defines a method in which kernel implementations may be overridden by externally
			specified kernels

		These kernels may be PTX or LLVM source files that are executed as EmulatedKernel and LLVMKernels
		respectively or objects with or without hooks into the Ocelot CTA runtime
	*/
	class ExternalKernel : public ir::ExecutableKernel {
	public:

		typedef int (*ManagedFunction)(LLVMContext *context);
		typedef int (*UnmanagedFunction)(char *local, char *param);

		//! \brief source for external kernel
		enum LoadingType {
			LoadingType_invalid=0,		//! kernel is not a valid loading type

			PTX_Source,				//! kernel is loaded from a PTX source file and executed as an EmulatedKernel
			LLVM_Source,			//! kernel is loaded from a source file in a directory known to Ocelot  
			Managed_Object,		//! kernel is an object file with hooks into the Ocelot runtime
			Unmanaged_Object	//! kernel is a regular C function
		};

		static LoadingType fromString(const std::string &str);

	public:

		ExternalKernel(const std::string &name, LoadingType type, const std::string &path, 
			ir::Module *module, const executive::Executive* c = 0);

		~ExternalKernel();

	public:

		/*!	\brief Determines whether kernel is executable */
		virtual bool executable();
		
		/*!	\brief Launch a kernel on a 2D grid */
		virtual void launchGrid(int width, int height, int depth);
	
		/*!	\brief Sets the shape of a kernel */
		virtual void setKernelShape(int x, int y, int z);

		/*! \brief Changes the amount of external shared memory */
		virtual void setExternSharedMemorySize(unsigned int);
		
		/*! \brief Describes the device used to execute the kernel */
		virtual void setDevice(const executive::Device* device, unsigned int limit);
			
		/*! \brief Indicate that the kernels parameters have been updated */
		virtual void updateParameterMemory();
		
		/*! \brief Indicate that other memory has been updated */
		virtual void updateMemory();
		
		/*! \brief Get a vector of all textures references by the kernel */
		virtual ir::ExecutableKernel::TextureVector textureReferences() const;

		/*!	Notifies all attached TraceGenerators of an event */
		void traceEvent(const trace::TraceEvent & event) const;

		/*!	adds a trace generator to the EmulatedKernel */
		virtual void addTraceGenerator(trace::TraceGenerator* generator);

		/*!	removes a trace generator from an EmulatedKernel */
		virtual void removeTraceGenerator(trace::TraceGenerator* generator);

	protected:

		//! loads a PTX kernel and merges into module
		bool loadAsPTXSource(const std::string & path);

		//! loads an LLVm module and merges into module
		bool loadAsLLVMSource(const std::string & path);

	public:

		//! indicates data source for the kernel
		LoadingType loadingType;

		//! identifies the source file or binary defining the external kernel
		std::string sourcePath;

	protected:

		//! pointer to emulated PTX kernel
		executive::EmulatedKernel *emulatedKernel;

		//! pointer to LLVM executable kernel
		executive::LLVMExecutableKernel *llvmKernel;		

		//! pointer to dynamically loaded object
		void *objectHandle;

		//! pointer to callable symbol in dynamically loaded object
		ManagedFunction managedFunction;

		//! pointer to callable symbol in dynamically loaded object
		UnmanagedFunction unmanagedFunction;
	};

}

#endif

