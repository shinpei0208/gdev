/*! \file   LLVMCooperativeThreadArray.cpp
	\date   Monday September 27, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the LLVMCooperativeThreadArray class.
*/

#ifndef LLVM_COOPERATIVE_THREAD_ARRAY_CPP_INCLUDED
#define LLVM_COOPERATIVE_THREAD_ARRAY_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMCooperativeThreadArray.h>
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/LLVMExecutableKernel.h>

#include <ocelot/executive/implementation/LLVMRuntimeLibrary.inl>

#include <ocelot/api/interface/OcelotConfiguration.h>

#include <ocelot/ir/interface/Module.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace executive
{

LLVMCooperativeThreadArray::LLVMCooperativeThreadArray(LLVMWorkerThread* w) :
	_warpSize(std::max(api::OcelotConfiguration::get().executive.warpSize, 1)),
	_worker(w)
{

}

void LLVMCooperativeThreadArray::setup(const LLVMExecutableKernel& kernel)
{
	report("Setting up LLVM-CTA to execute kernel " << kernel.name);
	_functions.resize(LLVMModuleManager::totalFunctionCount(), 0);
	_queuedThreads.resize(_functions.size());

	_entryPoint = _worker->getFunctionId(kernel.module->path(), kernel.name);
	report(" Entry point is function " << _entryPoint);

	if(_functions[_entryPoint] == 0)
	{
		report("  Loading entry point into cache.");
		_functions[_entryPoint] = _worker->getFunctionMetaData(_entryPoint);
	}

	const unsigned int threads = kernel.blockDim().x *
		kernel.blockDim().y * kernel.blockDim().z;
	report(" Creating contexts for " << threads << " threads.");

	_contexts.resize(threads);
	_stacks.resize(threads);
	_sharedMemory.resize(kernel.externSharedMemorySize()
		+ _functions[_entryPoint]->sharedSize);
	_globallyScopedLocalMemory.resize(
		threads*_functions[_entryPoint]->globalLocalSize);
	_kernel = &kernel;

	_freeContexts.resize(threads);
	for(ThreadList::iterator context = _freeContexts.begin();
		context != _freeContexts.end(); ++context)
	{
		*context = std::distance(_freeContexts.begin(), context);
	}
}


void LLVMCooperativeThreadArray::executeCta(unsigned int id)
{
	_nextFunction = _entryPoint;

	if(_functions[_nextFunction]->subkernels != 0)
	{
		_executeComplexCta(id);
	}
	else
	{
		_executeSimpleCta(id);
	}
}

void LLVMCooperativeThreadArray::flushTranslatedKernels()
{
	report("Flushing translated kernels.");
	_functions.clear();
}

void LLVMCooperativeThreadArray::_executeSimpleCta(unsigned int id)
{
	unsigned int contextId = _initializeNewContext(0, id);
	LLVMContext& context = _contexts[contextId];

	LLVMModuleManager::MetaData* metadata = _functions[_nextFunction];
	context.metadata = (char*) metadata;
	
	for(unsigned int z = 0; z < context.ntid.z; ++z)
	{
		context.tid.z = z;
		for(unsigned int y = 0; y < context.ntid.y; ++y)
		{
			context.tid.y = y;
			for(unsigned int x = 0; x < context.ntid.x; ++x)
			{
				context.tid.x = x;
				context.laneid = _threadId(context) % _warpSize;
				metadata->function(&context);
			}		
		}
	}
	
	_destroyContext(contextId);
}

/* We want to execute the CTA as quickly as possibly. Speed is the only 
	concern here, but it requires careful thread scheduling and state 
	management to maximize the opportunities for locality and vector 
	execution. The current algorithm is as follows:
	
	Always issue threads with the widest vector width available.
	
	1) Start by launching threads in order, if they finish before exiting 
		the subkernel, then they die here and their state is reclaimed.  If they
		bail out due to divergence or hit a context switch point, then allocate
		a new thread context for the next thread and save the current thread
		context's id for scheduling in a queue for the next subkernel.  
	2) Sort the queues by thread count.  Pick the queue with the most waiting
		threads.
		a) If it has not been jitted yet, do so now.
		b) Group threads together into warps of fused kernel width 
			(possibly by sorting but FCFS should require less overhead).  
			Launch them all.
		c) If a thread exits, kill it and reclaim the state, otherwise move 
			it to another queue.
		d) If a thread hits a barrier, put it into a barrier queue.
		d) Once all threads have been launched we are done with this sub-kernel.
	3) Reorganize the threads
		a) If all threads are in the barrier queue, move them back into their
			correspoding queues.
		b) If there is at least one thread left in at least one queue, goto 2.
		c) If all threads are finished, the CTA is done.

*/
void LLVMCooperativeThreadArray::_executeComplexCta(unsigned int id)
{
	const unsigned int threads  = _contexts.size();

	const unsigned int warps   = threads / _warpSize;
	const unsigned int remains = threads % _warpSize;

	report(" warp size:        " << _warpSize);
	report(" full warps:       " << warps);
	report(" remaining threas: " << remains);
	report(" entry point:      " << _entryPoint);

	ThreadList warpList;

	unsigned int threadId = 0;

	report("Executing LLVM-CTA " << id << " (" 
		<< _functions[_nextFunction]->kernel->name << ")");

	for(unsigned int warp = 0; warp < warps; ++warp)
	{
		for(unsigned int thread = 0; thread != _warpSize; ++thread)
		{
			warpList.push_back(_initializeNewContext(threadId++, id));
		}
		
		_executeWarp(warpList.begin(), warpList.end());
		
		for(ThreadList::const_iterator context = warpList.begin();
			context != warpList.end(); ++context)
		{
			_reclaimContext(*context);
		}
		
		warpList.clear();
	}
	
	for(unsigned int thread = 0; thread != remains; ++thread)
	{
		warpList.push_back(_initializeNewContext(threadId++, id));
	}
	
	_executeWarp(warpList.begin(), warpList.end());
	
	for(ThreadList::const_iterator context = warpList.begin();
		context != warpList.end(); ++context)
	{
		_reclaimContext(*context);
	}
	
	warpList.clear();

	while(_freeContexts.size() + _reclaimedContexts.size() != threads)
	{
		report("  while( free + reclaimed contexts (" 
			<< (_freeContexts.size() + _reclaimedContexts.size()) 
			<< ") != threads (" << threads << ")");
	
		_computeNextFunction();
		
		warpList = std::move(_queuedThreads[_nextFunction]);
		
		const unsigned int threads = warpList.size();
		const unsigned int warps   = threads / _warpSize;
		const unsigned int remains = threads % _warpSize;

		report("Next sub-kernel is " << _nextFunction << " (" 
			<< _functions[_nextFunction]->kernel->name << ")");

		report(" threads:          " << threads);
		report(" full warps:       " << warps);
		report(" remaining threas: " << remains);
		
		ThreadList::const_iterator begin = warpList.begin();

		for(unsigned int warp = 0; warp != warps; ++warp)
		{
			ThreadList::const_iterator end = begin;
			std::advance(end, _warpSize);
			_executeWarp(begin, end);
			begin = end;
		}
	
		ThreadList::const_iterator end = begin;
		std::advance(end, remains);
		_executeWarp(begin, end);
		
		for(ThreadList::const_iterator context = warpList.begin();
			context != warpList.end(); ++context)
		{
			_destroyContext(*context);
		}

		warpList.clear();
	}
	
	_destroyContexts();
}

void LLVMCooperativeThreadArray::_executeThread(unsigned int contextId)
{
	LLVMContext& context = _contexts[contextId];
	LLVMModuleManager::MetaData* metadata = _functions[_nextFunction];
	context.metadata = (char*) metadata;
	
	report("   executing thread " << _threadId(context)
		<< " in context " << contextId << " of " << _contexts.size() 
		<< ", _nextFunction: " << _nextFunction << " of " << _functions.size());
	
	metadata->function(&context);
}

void LLVMCooperativeThreadArray::_executeWarp(ThreadList::const_iterator begin,
	ThreadList::const_iterator end)
{
	report("  executing warp");
	// this is a stupid implementation of a warp that just loops over threads
	for(ThreadList::const_iterator i = begin; i != end; ++i)
	{
		_executeThread(*i);
	}
}

unsigned int LLVMCooperativeThreadArray::_initializeNewContext(
	unsigned int threadId, unsigned int ctaId)
{
	unsigned int contextId = 0;

	if(_reclaimedContexts.empty())
	{
		contextId = _freeContexts.back();
		_freeContexts.pop_back();
	
		LLVMContext& context                  = _contexts[contextId];
		LLVMFunctionCallStack& stack          = _stacks[contextId];
		LLVMModuleManager::MetaData& metadata = *_functions[_nextFunction];
		
		stack.setKernelArgumentMemory(_kernel->argumentMemory(),
			_kernel->argumentMemorySize());
			
		report(" Pushing call stack (" << metadata.localSize 
			<< " local) (" << metadata.parameterSize << " parameter)");
		stack.call(metadata.localSize, metadata.parameterSize, _nextFunction);

		context.nctaid.x  = _kernel->gridDim().x;
		context.nctaid.y  = _kernel->gridDim().y;
		context.nctaid.z  = _kernel->gridDim().z;
		context.ctaid.x   = ctaId % context.nctaid.x;
		context.ctaid.y   = (ctaId / context.nctaid.x) % context.nctaid.y;
		context.ctaid.z   = ((ctaId / context.nctaid.x) / context.nctaid.y) %
			context.nctaid.z;
		context.ntid.x    = _kernel->blockDim().x;
		context.ntid.y    = _kernel->blockDim().y;
		context.ntid.z    = _kernel->blockDim().z;
		context.tid.x     = threadId % context.ntid.x;
		context.tid.y     = (threadId / context.ntid.x) % context.ntid.y;
		context.tid.z     = threadId / (context.ntid.x * context.ntid.y);
		context.shared    = reinterpret_cast<char*>(_sharedMemory.data());
		context.argument  = stack.argumentMemory();
		context.local     = stack.localMemory();
		context.parameter = stack.parameterMemory();
		context.constant  = _kernel->constantMemory();
		context.globallyScopedLocal =
			reinterpret_cast<char*>(_globallyScopedLocalMemory.data()) +
			metadata.globalLocalSize * threadId;
		context.externalSharedSize  = _kernel->externSharedMemorySize();
		context.metadata            = reinterpret_cast<char*>(&metadata);
		context.laneid    = threadId %  _warpSize;
	}
	else
	{
		contextId = _reclaimedContexts.back();
		_reclaimedContexts.pop_back();
	
		LLVMContext& context = _contexts[contextId];
	
		context.tid.x = threadId % context.ntid.x;
		context.tid.y = (threadId / context.ntid.x) % context.ntid.y;
		context.tid.z = threadId / (context.ntid.x * context.ntid.y);
		context.laneid    = threadId %  _warpSize;
	}	
	
	return contextId;
}

void LLVMCooperativeThreadArray::_computeNextFunction()
{
	if(_queuedThreads[0].size() == _contexts.size())
	{
		_nextFunction = 0;
	}
	else if(_queuedThreads[_guessFunction].size() >= _contexts.size()/2)
	{
		_nextFunction = _guessFunction;
	}
	else
	{
		_nextFunction      = 1;
		unsigned int total = 0;
		unsigned int count = _queuedThreads[1].size();
		
		ThreadListVector::iterator queue = _queuedThreads.begin();
		
		for(std::advance(queue, 2); queue != _queuedThreads.end(); ++queue)
		{
			if(queue->size() > count)
			{
				_nextFunction = std::distance(_queuedThreads.begin(), queue);
				count = queue->size();
			}
			
			total += queue->size();
			if(total > _contexts.size() / 2) break;
		}
	}
	
	// lazily compile the function
	if(_functions[_nextFunction] == 0)
	{
		_functions[_nextFunction] = _worker->getFunctionMetaData(_nextFunction);
	}
	
	assert(!_queuedThreads[_nextFunction].empty());
}

void LLVMCooperativeThreadArray::_reclaimContext(unsigned int contextId)
{
	if(_finishContext(contextId))
	{
		_reclaimedContexts.push_back(contextId);
		report("    thread hit exit point, reclaiming context.");
	}
}

void LLVMCooperativeThreadArray::_destroyContext(unsigned int contextId)
{
	if(_finishContext(contextId))
	{
		_freeContexts.push_back(contextId);
		report("    thread hit exit point, destroying context.");
		LLVMFunctionCallStack& stack = _stacks[contextId];
		stack.returned();
	}
}

bool LLVMCooperativeThreadArray::_finishContext(unsigned int contextId)
{
	LLVMContext& context         = _contexts[contextId];
	LLVMFunctionCallStack& stack = _stacks[contextId];

	unsigned int* localMemory = reinterpret_cast<unsigned int*>(context.local);
	
	report("   thread context " << contextId  << " finished.");
	
	unsigned int callType     = localMemory[0];
	unsigned int nextFunction = -1;

	switch(callType)
	{
	case LLVMExecutableKernel::ExitCall:
	{
		return true;
	}
	break;
	case LLVMExecutableKernel::BarrierCall:
	{
		assertM(0 < _queuedThreads.size(), "Next function " 
			<< 0 << " is out of range of function table with "
			<< _queuedThreads.size() << " entries.");

		_queuedThreads[0].push_back(contextId);

		report("     hit barrier, removing from scheduling pool.");
		return false;
	}
	break;
	case LLVMExecutableKernel::TailCall:
	{
		nextFunction = localMemory[1];

		// adjust the next function by the function base
		nextFunction += stack.functionId();

		if(nextFunction == _nextFunction)
		{
			_queuedThreads[nextFunction].push_back(contextId);
			return false;
		}		
		
		// lazily compile the function, get the stack size
		if(_functions[nextFunction] == 0)
		{
			_functions[nextFunction] = 
				_worker->getFunctionMetaData(nextFunction);
		}

		LLVMModuleManager::MetaData& metadata = *_functions[nextFunction];
		
		stack.resizeCurrentLocalMemory(metadata.localSize);

		context.local         = stack.localMemory();
		context.parameter     = stack.parameterMemory();
		context.argument      = stack.argumentMemory();
		
		report("     hit tail call, saving thread context at resume point "
			<< nextFunction << ", setting local memory size to "
			<< metadata.localSize << ".");
	}
	break;
	case LLVMExecutableKernel::NormalCall:
	{
		nextFunction = localMemory[1];

		if(_functions[nextFunction] == 0)
		{
			_functions[nextFunction] =
				_worker->getFunctionMetaData(nextFunction);
		}

		LLVMModuleManager::MetaData& metadata = *_functions[nextFunction];

		report("     hit function call, saving thread context at resume point "
			<< nextFunction << ", pushing stack (" << metadata.localSize 
			<< " local) (" << metadata.parameterSize << " parameter).");

		stack.call(metadata.localSize, metadata.parameterSize,
			nextFunction, _nextFunction);
		context.local         = stack.localMemory();
		context.parameter     = stack.parameterMemory();
		context.argument      = stack.argumentMemory();
	}
	break;
	case LLVMExecutableKernel::ReturnCall:
	{
		nextFunction      = stack.returned();
		context.local     = stack.localMemory();
		context.parameter = stack.parameterMemory();
		context.argument  = stack.argumentMemory();

		report("     hit return, saving thread context at resume point "
			<< nextFunction << ", popping stack.");
	}
	break;
	}

	_guessFunction = nextFunction;
	if (nextFunction != 0xffffffff) {
		assertM(nextFunction < _queuedThreads.size(), "Next function " 
			<< nextFunction << " is out of range of function table with "
			<< _queuedThreads.size() << " entries.");
		_queuedThreads[nextFunction].push_back(contextId);
	}
	else {
		return true;
	}

	return false;
}

void LLVMCooperativeThreadArray::_destroyContexts()
{
	for(ThreadList::const_iterator context = _reclaimedContexts.begin();
		context != _reclaimedContexts.end(); ++context)
	{
		LLVMFunctionCallStack& stack = _stacks[*context];
		stack.returned();
	}
	
	_freeContexts.insert(_freeContexts.end(), _reclaimedContexts.begin(), 
		_reclaimedContexts.end());

	_reclaimedContexts.clear();
}

unsigned int LLVMCooperativeThreadArray::_threadId(const LLVMContext& context)
{
	return context.tid.x + context.tid.y * context.ntid.x
		+ context.tid.z * context.ntid.y * context.ntid.z;
}

}

#endif

