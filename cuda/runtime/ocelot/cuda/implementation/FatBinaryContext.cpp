/*!
	\file FatBinaryContext.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief object for interacting with CUDA Fat Binaries
*/


// Ocelot Includes
#include <ocelot/cuda/interface/FatBinaryContext.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/compression.h>
#include <hydrazine/interface/ELFFile.h>

// Standard Library Includes
#include <cstring>
#include <sstream>
#include <fstream>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////
cuda::FatBinaryContext::FatBinaryContext(const void *ptr): cubin_ptr(ptr) {

	report("FatBinaryContext(" << ptr << ")");

	_name = 0;
	_ptx = 0;
	_cubin = 0;

	if(*(int*)cubin_ptr == __cudaFatMAGIC) {
		__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)cubin_ptr;

		_name = binary->ident;

		//assertM(binary->ptx != 0, "binary contains no PTX");
		//assertM(binary->ptx->ptx != 0, "binary contains no PTX");

		unsigned int ptxVersion = 0;
		unsigned int cubinVersion = 0;

		if (binary->ptx) {
			report("Getting the highest PTX version");

			for(unsigned int i = 0; ; ++i)
			{
				if((binary->ptx[i].ptx) == 0) break;
		
				std::string computeCapability = binary->ptx[i].gpuProfileName;
				std::string versionString(computeCapability.begin() + 8,
					computeCapability.end());
		
				std::stringstream version;
				unsigned int thisVersion = 0;
			
				version << versionString;
				version >> thisVersion;
				if(thisVersion > ptxVersion)
				{
					ptxVersion = thisVersion;
					_ptx = binary->ptx[i].ptx;
				}
			}		
			report(" Selected version " << ptxVersion);
		}
		if (binary->cubin) {
			report("Getting the highest CUBIN version");

			for(unsigned int i = 0; ; ++i)
			{
				if((binary->cubin[i].cubin) == 0) break;
		
				std::string computeCapability = binary->cubin[i].gpuProfileName;
				std::string versionString(computeCapability.begin() + 8,
					computeCapability.end());
		
				std::stringstream version;
				unsigned int thisVersion = 0;
			
				version << versionString;
				version >> thisVersion;
				if(thisVersion > cubinVersion)
				{
					cubinVersion = thisVersion;
					_cubin = binary->cubin[i].cubin;
				}
			}		
			report(" Selected version " << cubinVersion);
		}
	}
	else if (*(int*)cubin_ptr == __cudaFatMAGIC2) {
		report("Found new fat binary format!");
		__cudaFatCudaBinary2* binary = (__cudaFatCudaBinary2*) cubin_ptr;
		__cudaFatCudaBinary2Header* header =
			(__cudaFatCudaBinary2Header*) binary->fatbinData;
		
		report(" binary size is: " << header->length << " bytes");
				
		char* base = (char*)(header + 1);
		long long unsigned int offset = 0;
		__cudaFatCudaBinary2EntryRec* entry = (__cudaFatCudaBinary2EntryRec*)(base);
		
#if 1
		while (offset <= header->length) {
			_name = (char*)entry + entry->name;
			if (entry->type & FATBIN_2_PTX) {
				if (!_ptx) {
					_ptx  = (char*)entry + entry->binary;
					if(entry->flags & COMPRESSED_PTX)
					{
						_decompressedPTX.resize(entry->uncompressedBinarySize + 1);
						_decompressPTX(entry->binarySize);
					}
				}
			}
			if (entry->type & FATBIN_2_ELF) {
				if (!_cubin)
					_cubin  = (char*)entry + entry->binary;
			}
#if 0
			if (entry->type & FATBIN_2_OLDCUBIN) {
				if (!_cubin)
					_cubin  = (char*)entry + entry->binary;
			}
#endif

			entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
			offset += entry->binary + entry->binarySize;
		}
#else
		while (!(entry->type & FATBIN_2_PTX) && offset < header->length) {
			_name = (char*)entry + entry->name;
			entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
			offset += entry->binary + entry->binarySize;
		}
		if (entry->type & FATBIN_2_PTX) {
			_ptx  = (char*)entry + entry->binary;
		}
		if(entry->flags & COMPRESSED_PTX)
		{
			_decompressedPTX.resize(entry->uncompressedBinarySize + 1);
			_decompressPTX(entry->binarySize);
		}
#endif
	}
	else {
		assertM(false, "unknown fat binary magic number "
			<< std::hex << *(int*)cubin_ptr);		
	}
	
	if (!_ptx) {
		report("registered, contains NO PTX");
	}
	else {
		report("registered, contains PTX");	
	}
}

cuda::FatBinaryContext::FatBinaryContext(): cubin_ptr(0), _name(0), _ptx(0) {

}

const char* cuda::FatBinaryContext::name() const {
	return _name;
}	

const char* cuda::FatBinaryContext::ptx() const {
	if(!_decompressedPTX.empty()) return (const char*) _decompressedPTX.data();
	
	return _ptx;
}

const char* cuda::FatBinaryContext::cubin() const {
	return _cubin;
}

void cuda::FatBinaryContext::_decompressPTX(unsigned int compressedBinarySize) {
	uint64_t decompressedSize = _decompressedPTX.size() - 1;

	hydrazine::decompress(_decompressedPTX.data(), decompressedSize, _ptx,
		compressedBinarySize - 1);

	_decompressedPTX.back() = '\0';
}

////////////////////////////////////////////////////////////////////////////////

