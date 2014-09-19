/*!
	\file cudaFatBinary.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief this was extracted from cuda_runtime.h as these structures are shared by
		both the CUDA Runtime API and Driver APIs
*/

#ifndef OCELOT_CUDA_CUDAFATBINARY_H_INCLUDED
#define OCELOT_CUDA_CUDAFATBINARY_H_INCLUDED

/*----------------------------------- Types ----------------------------------*/

/*
 * Cubin entry type for __cudaFat binary. 
 * Cubins are specific to a particular gpu profile,
 * although the gpuInfo module might 'know'
 * that cubins will also run on other gpus.
 * Based on the recompilation strategy, 
 * fatGetCubinForGpu will return an existing
 * compatible load image, or attempt a recompilation.
 */
typedef struct {
    char*            gpuProfileName;
    char*            cubin;
} __cudaFatCubinEntry;


/*
 * Ptx entry type for __cudaFat binary.
 * PTX might use particular chip features
 * (such as double precision floating points).
 * When attempting to recompile for a certain 
 * gpu architecture, a ptx needs to be available
 * that depends on features that are either 
 * implemented by the gpu, or for which the ptx
 * translator can provide an emulation. 
 */
typedef struct {
    char*            gpuProfileName;            
    char*            ptx;
} __cudaFatPtxEntry;


/*
 * Debug entry type for __cudaFat binary.
 * Such information might, but need not be available
 * for Cubin entries (ptx files compiled in debug mode
 * will contain their own debugging information) 
 */
typedef struct __cudaFatDebugEntryRec {
    char*                   gpuProfileName;            
    char*                   debug;
    struct __cudaFatDebugEntryRec *next;
    unsigned int                   size;
} __cudaFatDebugEntry;

typedef struct __cudaFatElfEntryRec {
    char*                 gpuProfileName;            
    char*                 elf;
    struct __cudaFatElfEntryRec *next;
    unsigned int                 size;
} __cudaFatElfEntry;

typedef enum {
      __cudaFatDontSearchFlag = (1 << 0),
      __cudaFatDontCacheFlag  = (1 << 1),
      __cudaFatSassDebugFlag  = (1 << 2)
} __cudaFatCudaBinaryFlag;

/*
 * Imported/exported symbol descriptor, needed for 
 * __cudaFat binary linking. Not much information is needed,
 * because this is only an index: full symbol information 
 * is contained by the binaries.
 */
typedef struct {
    char* name;
} __cudaFatSymbol;

/*
 * Fat binary container.
 * A mix of ptx intermediate programs and cubins,
 * plus a global identifier that can be used for 
 * further lookup in a translation cache or a resource
 * file. This key is a checksum over the device text.
 * The ptx and cubin array are each terminated with 
 * entries that have NULL components.
 */
 
typedef struct __cudaFatCudaBinaryRec {
    unsigned long            magic;
    unsigned long            version;
    unsigned long            gpuInfoVersion;
    char*                   key;
    char*                   ident;
    char*                   usageMode;
    __cudaFatPtxEntry             *ptx;
    __cudaFatCubinEntry           *cubin;
    __cudaFatDebugEntry           *debug;
    void*                  debugInfo;
    unsigned int                   flags;
    __cudaFatSymbol               *exported;
    __cudaFatSymbol               *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int                   characteristic;
    __cudaFatElfEntry             *elf;
} __cudaFatCudaBinary;

typedef struct __cudaFatCudaBinary2HeaderRec { 
    unsigned int            magic;
    unsigned int            version;
	unsigned long long int  length;
} __cudaFatCudaBinary2Header;

enum FatBin2EntryType {
	FATBIN_2_PTX = 0x1
};

typedef struct __cudaFatCudaBinary2EntryRec { 
	unsigned int           type;
	unsigned int           binary;
	unsigned long long int binarySize;
	unsigned int           unknown2;
	unsigned int           kindOffset;
	unsigned int           unknown3;
	unsigned int           unknown4;
	unsigned int           name;
	unsigned int           nameSize;
	unsigned long long int flags;
	unsigned long long int unknown7;
	unsigned long long int uncompressedBinarySize;
} __cudaFatCudaBinary2Entry;

#define COMPRESSED_PTX 0x0000000000001000LL

typedef struct __cudaFatCudaBinaryRec2 {
	int magic;
	int version;
	const unsigned long long* fatbinData;
	char* f;
} __cudaFatCudaBinary2;

/*
 * Current version and magic numbers:
 */
#define __cudaFatVERSION   0x00000004
#define __cudaFatMAGIC     0x1ee55a01
#define __cudaFatMAGIC2    0x466243b1
#define __cudaFatMAGIC3    0xba55ed50

/*
 * Version history log:
 *    1  : __cudaFatDebugEntry field added to __cudaFatCudaBinary struct
 *    2  : flags and debugInfo field added.
 *    3  : import/export symbol list
 *    4  : characteristic added, elf added
 */


/*--------------------------------- Functions --------------------------------*/

typedef enum {
    __cudaFatAvoidPTX,
    __cudaFatPreferBestCode,
    __cudaFatForcePTX
} __cudaFatCompilationPolicy;

#endif

