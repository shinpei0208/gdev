/*! \file LLVMWorkerThread.cpp
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the LLVMWorkerThread class.
*/

#ifndef LLVM_WORKER_THREAD_CPP_INCLUDED
#define LLVM_WORKER_THREAD_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMWorkerThread.h>
#include <ocelot/executive/interface/LLVMCooperativeThreadArray.h>
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/LLVMExecutableKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Defines
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace executive
{

class WorkerMessage
{
public:
	enum Type
	{
		Kill,
		SetupCta,
		LaunchCta,
		FlushCta,
		Exception,
		Invalid
	};
	
public:
	Type        type;
	std::string errorMessage;

public:
	union
	{
		unsigned int ctaId;
		const LLVMExecutableKernel* kernel;
	};
	
};

LLVMWorkerThread::LLVMWorkerThread()
{
	report("Creating new LLVM Worker Thread");
	start();
}

LLVMWorkerThread::~LLVMWorkerThread()
{
	WorkerMessage message;
	message.type = WorkerMessage::Kill;
	send(&message);

	WorkerMessage* reply;
	receive(reply);
	assert(reply == &message);
	report("Tearing down LLVM Worker Thread " << id());
}

void LLVMWorkerThread::setupCta(const LLVMExecutableKernel& kernel)
{
	WorkerMessage message;
	message.type = WorkerMessage::SetupCta;
	message.kernel = &kernel;
	
	send(&message);
	
	WorkerMessage* reply;
	receive(reply);
	assert(reply == &message);

	if(message.type == WorkerMessage::Exception)
	{
		report("Re-throwing exception");
		throw hydrazine::Exception(message.errorMessage);
	}
}

void LLVMWorkerThread::launchCta(unsigned int ctaId)
{
	WorkerMessage* message = new WorkerMessage;
	message->type = WorkerMessage::LaunchCta;
	message->ctaId = ctaId;
	
	send(message);
}

void LLVMWorkerThread::finishCta()
{
	WorkerMessage* message;
	receive(message);

	if(message->type == WorkerMessage::Exception)
	{
		report("Re-throwing exception");
		hydrazine::Exception exception(message->errorMessage);
		delete message;
		throw exception;
	}
	
	delete message;
}

void LLVMWorkerThread::flushTranslatedKernels()
{
	WorkerMessage message;
	message.type = WorkerMessage::FlushCta;
	
	send(&message);
	
	WorkerMessage* reply;
	receive(reply);
	assert(reply == &message);	

	if(message.type == WorkerMessage::Exception)
	{
		report("Re-throwing exception");
		throw hydrazine::Exception(message.errorMessage);
	}
}

LLVMModuleManager::FunctionId LLVMWorkerThread::getFunctionId(
	const std::string& moduleName, const std::string& functionName)
{
	LLVMModuleManager::GetFunctionMessage message;
	
	message.type       = LLVMModuleManager::DatabaseMessage::GetId;
	message.moduleName = moduleName;
	message.kernelName = functionName;
	
	threadSend(&message, LLVMModuleManager::id());
	
	LLVMModuleManager::GetFunctionMessage* reply = 0;
	threadReceive(reply);
	assert(reply == &message);
	
	if(message.type == LLVMModuleManager::DatabaseMessage::Exception)
	{
		report("Re-throwing exception");
		throw hydrazine::Exception(message.errorMessage);
	}
	
	return message.id;
}

LLVMModuleManager::MetaData* LLVMWorkerThread::getFunctionMetaData(
	const LLVMModuleManager::FunctionId& id)
{
	LLVMModuleManager::GetFunctionMessage message;
	
	message.type = LLVMModuleManager::DatabaseMessage::GetFunction;
	message.id   = id;
	
	threadSend(&message, LLVMModuleManager::id());
	
	LLVMModuleManager::GetFunctionMessage* reply = 0;
	threadReceive(reply);
	assert(reply == &message);
	
	if(message.type == LLVMModuleManager::DatabaseMessage::Exception)
	{
		report("Re-throwing exception");
		throw hydrazine::Exception(message.errorMessage);
	}
	
	assert(message.type == LLVMModuleManager::DatabaseMessage::GetFunction);
	
	return message.metadata;
}

void LLVMWorkerThread::execute()
{
	WorkerMessage*             message;
	LLVMCooperativeThreadArray cta(this);
	
	report("LLVMWorker thread is alive, waiting for command.");
	
	threadReceive(message);
	while(message->type != WorkerMessage::Kill)
	{
		try
		{
			switch(message->type)
			{
			case WorkerMessage::LaunchCta:
			{
				report(" Launching CTA " << message->ctaId 
					<< " on thread " << id() << ".");
				cta.executeCta(message->ctaId);
				threadSend(message);
			}
			break;
			case WorkerMessage::SetupCta:
			{
				report(" Setting up CTA for kernel '" << message->kernel->name 
					<< "' on thread " << id() << ".");
				cta.setup(*message->kernel);
				threadSend(message);
			}
			break;
			case WorkerMessage::FlushCta:
			{
				report(" Flushing translation cache on thread " << id() << ".");
				cta.flushTranslatedKernels();
				threadSend(message);
			}
			break;
			default: assertM(false, "Invalid message type.");
			}
		}
		catch(const hydrazine::Exception& e)
		{
			report("Operation failed, replying with exception.");
			message->type         = WorkerMessage::Exception;
			message->errorMessage = e.what();
			threadSend(message);
		}
		threadReceive(message);
	}

	report("Received kill command, joining.");
	threadSend(message);
}

}

#endif

