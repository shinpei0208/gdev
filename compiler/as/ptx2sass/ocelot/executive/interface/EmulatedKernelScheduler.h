/*	\file   EmulatedKernelScheduler.h
	\date   Tuesday July 3, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the EmulatedKernelScheduler class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/ir/interface/Dim3.h>
#include <ocelot/executive/interface/ExecutableKernel.h>

// Standard Library Includes
#include <list>
#include <vector>
#include <map>

// Forward Declarations
namespace executive { class EmulatorDevice;         }
namespace executive { class EmulatedKernel;         }
namespace executive { class CooperativeThreadArray; }

namespace executive
{

/*! \brief Supports task switching between multiple emulated kernels */
class EmulatedKernelScheduler
{
public:
	typedef ExecutableKernel::TraceGeneratorVector TraceGeneratorVector;

public:
	EmulatedKernelScheduler(EmulatorDevice* owningDevice);

public:
	#ifndef _WIN32
	EmulatedKernelScheduler(const EmulatedKernelScheduler&)            = delete;
	EmulatedKernelScheduler& operator=(const EmulatedKernelScheduler&) = delete;
	#endif

public:
	/*! \brief Launch a new kernel through the emulated kernel interface */
	void launch(EmulatedKernel* kernel, const ir::Dim3& dimensions,
		TraceGeneratorVector*);

public:
	/*! \brief Launch a nested kernel from the current context */
	void launch(ir::PTXU64 pc, ir::PTXU64 parameterBuffer, const ir::Dim3& gridDim,
		const ir::Dim3& ctaDim, ir::PTXU32 sharedMemory, ir::PTXU64 stream);
	/*! \brief Get the argument memory for the current context */
	ir::PTXU64 argumentMemory() const;

private:
	class Context
	{
	public:
		typedef unsigned int          Priority;
		typedef unsigned int          Id;
		typedef std::list<Context>    ContextList;
		typedef ContextList::iterator iterator;
		typedef std::vector<iterator> ContextIteratorVector;
		typedef std::vector<uint8_t>  ByteVector;

	public:
		Context(Id id, EmulatedKernel* kernel, const ir::Dim3& gridDimensions,
			const ir::Dim3& ctaDim,
			iterator parent, EmulatedKernelScheduler* scheduler, Priority p);
		Context(Id id, const EmulatedKernel* kernel, uint64_t pc,
			const ir::Dim3& gridDimensions, const ir::Dim3& ctaDim,
			const void* argumentData, size_t argumentSize, iterator parent,
			EmulatedKernelScheduler* scheduler, Priority p);

	public:
		void executeUntilYield();
		bool exited() const;

		bool hasParent() const;
		bool blockedOnChildren() const;

	public:
		Id              id;
		EmulatedKernel* kernel;
		uint64_t        startingPC;
		ir::Dim3        gridDimensions;
		ByteVector      argumentMemory;	
		bool            isSuspended;

	public:
		iterator              parent;
		ContextIteratorVector children;
		Priority              priority;

	private:
		void _yieldExit();
		void _yieldBarrier();

	private:
		EmulatedKernelScheduler* _scheduler;
		CooperativeThreadArray*  _cta;
		ir::Dim3                 _positionInGrid;
	};

	typedef Context::ContextList                  ContextList;
	typedef Context::Priority                     Priority;
	typedef std::map<Priority, Context::iterator> PriorityContextMap;

private:
	void _scheduler();

private:
	Context::iterator _getExecutingContext() const;
	const EmulatedKernel* _getKernelAtPC(unsigned int PC) const;

private:
	Context::Id        _nextId;
	ContextList        _contexts;
	PriorityContextMap _executingContexts;

private:
	EmulatorDevice*       _device;
	TraceGeneratorVector* _generators;
};

}


