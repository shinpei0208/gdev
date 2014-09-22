/*! \file OcelotConfiguration.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief defines a configuration class for GPU Ocelot

	By centralizing the configuration object, it should be fairly clear what the structure 
	of an Ocelot configuration is and how to extend it.
*/

#ifndef GPUOCELOT_API_OCELOTCONFIGURATION_H_INCLUDED
#define GPUOCELOT_API_OCELOTCONFIGURATION_H_INCLUDED

// C++ stdlib includes
#include <string>
#include <fstream>

namespace api {

	//! \brief configuration object structure for GPU Ocelot
	class OcelotConfiguration {
	public:
	
		class Checkpoint {
		public:
			
		public:
			Checkpoint();
			
		public:
			//! \brief indicates checkpointing is enabled
			bool enabled;
			
			//! \brief path in which checkpoint files are stored
			std::string path;
			
			//! \brief prefix for all checkpoint files
			std::string prefix;
			
			//! \brief suffix for all checkpoint files
			std::string suffix;
			
			//! \brief if checkpoints exist, verify allocations match them
			bool verify;

			//! \brief Only attach to kernels matching this substring
			std::string kernelFilter;
		};

		//! Configuration for trace generation facilities
		class TraceGeneration {
		public:
			/*! \brief configuration properties for trace::InteractiveDebugger */
			class Debugger
			{
			public:
					Debugger();
				
			public:
					//! \brief indicates trace generator is enabled
					bool enabled;

					//! \brief Only attach to kernels matching this substring
					std::string kernelFilter;
				
					//! \brief Attach to all kernels by default?
					bool alwaysAttach;
			};
			
			/*! \brief configuration properties for trace::RaceDetector */
			class RaceDetector
			{
			public:
					RaceDetector();
				
			public:
					//! \brief indicates trace generator is enabled
					bool enabled;
					
					//! \brief Do not report races that write over a value
					//	with the same value
					bool ignoreIrrelevantWrites;
			};
			
			class KernelTimer {
			public:
				KernelTimer();
				bool enabled;
				std::string outputFile;
			};

		public:
			TraceGeneration();

		public:
			//! \brief Check memory errors
			class MemoryChecker
			{
			public:
					MemoryChecker();
			public:
					bool enabled;
					bool checkInitialization;
			};
			
			//! \brief Race detection
			RaceDetector raceDetector;
                        
			//! \brief Interactive Debugger
			Debugger debugger;

			//! \brief Memory checker
			MemoryChecker memoryChecker;

			//! \brief measures the total runtime of kernels launched by the application
			KernelTimer kernelTimer;
		};

		class CudaRuntimeImplementation {
		public:
			CudaRuntimeImplementation();

		public:

			//! \brief names particular CUDA Runtime implementation 
			//         to instantiate
			std::string implementation;
			
			//! \brief for TraceGeneratingRuntime, path to output file
			std::string runtimeApiTrace;
		};

		class Executive {
		public:
			Executive();

		public:
			//! \brief ID of device to select by default
			int defaultDeviceID;

			/*! \brief selects preferred ISA [if multiple are available] 
				- cast to ir::Instruction::Architecture
			*/
			int preferredISA;

			/*!  \brief selects optimization level - cast to 
				translator::Translator::OptimizationLevel */
			int optimizationLevel;

			//! \brief if true, Ocelot will either choose only 
			//                  the preferred ISA or die
			bool required;

			//! \brief filter enabling LLVM devices if present
			bool enableLLVM;

			//! \brief filter enbabling Emulated PTX devices if present
			bool enableEmulated;

			//! \brief filter enabling CUDA GPU devices if present
			bool enableNVIDIA;
			
			//! \brief filter enabling AMD GPU devices if present
			bool enableAMD;
			
			//! \brief filter enabling remote GPU devices if present
			bool enableRemote;
			
			//! \brief if true, enables asynchronous kernel launches
			bool asynchronousKernelLaunch;
			
			//! \brief The port to connect to an ocelot server for remoting
			int port;

			//! \brief The hostname of the ocelot server to connect
			std::string host;

			//! \brief Reconvergence mechanism
			int reconvergenceMechanism;

			//! \brief maximum number of worker threads - a non-positive 
			//         number indicates no limit
			int workerThreadLimit;

			//! \brief suggested warp size
			int warpSize;
		};

		class Optimizations {
		public:
			Optimizations();

		public:
			//! \brief The target subkernel size in instructions
			unsigned int subkernelSize;
		
			/*! \brief Enable the structural transform */
			bool structuralTransform;
			
			/*! \brief Enable convert predication to select */
			bool predicateToSelect;
			
			/*! \brief Enable linear scan register allocation pass */
			bool linearScanAllocation;
			
			/*! \brief Enable MIMD thread scheduling pass */
			bool mimdThreadScheduling;
			
			/*! \brief Sync elimination pass */
			bool syncElimination;	
			
			/*! \brief Hoist special values pass */
			bool hoistSpecialValues;	
			
			/*! \brief Simplify control graph pass */
			bool simplifyCFG;		
			
			/*! \brief Enforce lock-step execution pass */
			bool enforceLockStepExecution;	
			
			/*! \brief Function inlining pass */
			bool inlining;		
		};


	public:

		//! \brief construct with default configuration
		OcelotConfiguration();

		//! \brief load configuration from stream as JSON
		OcelotConfiguration(std::istream &stream);

		//! \brief loads the configuration object from a path
		OcelotConfiguration(const std::string &path);

		//! \brief initializes configuration object from a stream as JSON
		void *initialize(std::istream &stream, bool preserve = false);
		
		//! \brief parses and returns configuration object if need be
		void *configuration() const;

    //! \brief gets singleton configuration object or 
    //	constructs from 'configure.ocelot'
    static const OcelotConfiguration& get();

    //! \brief destroys the singleton
    static void destroy();

	public:
		//! \brief path to configuration file for reparsing 
		std::string path;

		//! \brief name of Ocelot implementation
		std::string ocelot;

		//! \brief version of current Ocelot configuration
		std::string version;

		//! \brief configuration for trace generator
		TraceGeneration trace;

		//! \brief configuration for CUDA runtime implementation
		CudaRuntimeImplementation cuda;

		//! \brief configuration for Ocelot Executive
		Executive executive;
		
		//! \brief configuration for Ocelot checkpoint
		Checkpoint checkpoint;
		
		//! \brief configuration for Ocelot optimizations
		Optimizations optimizations;
	};

}

#endif

