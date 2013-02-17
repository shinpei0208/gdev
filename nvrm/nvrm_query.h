#ifndef NVRM_QUERY_H
#define NVRM_QUERY_H

#include <inttypes.h>

/* only seen on device so far */

/* XXX reads PBFB.CFG1 on NVCF */
struct nvrm_query_gpu_params {
	uint32_t unk00;
	uint32_t unk04;
	uint32_t unk08;
	uint32_t unk0c;
	uint32_t unk10;
	uint32_t unk14;
	uint32_t unk18;
	uint32_t unk1c;
	uint32_t unk20;
	uint32_t nv50_gpu_units;
	uint32_t unk28;
	uint32_t unk2c;
};
#define NVRM_QUERY_GPU_PARAMS 0x00000125

struct nvrm_query_object_classes {
	uint32_t cnt;
	uint32_t _pad;
	uint64_t ptr;
};
#define NVRM_QUERY_OBJECT_CLASSES 0x0000014c

struct nvrm_query_unk019a {
	uint32_t unk00;
};
#define NVRM_QUERY_UNK019A 0x0000019a

#endif
