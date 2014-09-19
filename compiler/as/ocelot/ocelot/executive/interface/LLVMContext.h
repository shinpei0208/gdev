/*!
	\file LLVMContext.h
	\date Tuesday September 8, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMContext class.
*/

#ifndef LLVM_CONTEXT_H_INCLUDED
#define LLVM_CONTEXT_H_INCLUDED

#include <cstring>

namespace executive
{

/*! \brief A class contains the state for executing a kernel */
class LLVMContext
{
public:
	/*! \brief A 3-D dimension corresponding to the CUDA notion */
	class Dimension
	{
		public:
			unsigned int x;
			unsigned int y;
			unsigned int z;
	};

public:
	Dimension tid;    //! Thread ids
	Dimension ntid;   //! CTA dimensions
	Dimension ctaid;  //! CTA ids
	Dimension nctaid; //! Kernel dimensions

public:
	char* local;               //! Pointer to local memory
	char* shared;              //! Pointer to shared memory
	char* constant;            //! Pointer to constant memory
	char* parameter;           //! Pointer to parameter memory
	char* argument;            //! Pointer to argument memory
	char* globallyScopedLocal; //! Pointer to globally scoped local memory
	
public:
	unsigned int laneid; //! lane id

public:
	unsigned int externalSharedSize; //! External shared size

public:
	/*! \brief Generic pointer back to other state */
	char* metadata;
};

}

#endif

