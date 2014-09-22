/*!
	\file FatBinaryContext.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief object for interacting with CUDA Fat Binaries
*/

#ifndef OCELOT_CUDA_FATBINARYCONTEXT_H_INCLUDED
#define OCELOT_CUDA_FATBINARYCONTEXT_H_INCLUDED

// Ocelot Includes
#include <ocelot/cuda/interface/cudaFatBinary.h>

// Standard Library Includes
#include <vector>

namespace cuda {
	/*!	\brief Class allowing sharing of a fat binary among threads	*/
	class FatBinaryContext {
	public:
		FatBinaryContext(const void *);
		FatBinaryContext();
	
		//! pointer to CUBIN structure
		const void *cubin_ptr;
		
	public:
		const char *name() const;
		const char *ptx() const;

	private:
		void _decompressPTX(unsigned int compressedBinarySize);

	private:
		const char* _name;
		const char* _ptx;

	private:
		typedef std::vector<char> ByteVector;

	private:
		ByteVector _decompressedPTX;
	
	};
}

#endif

