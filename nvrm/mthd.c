#include "nvrm_priv.h"
#include "nvrm_mthd.h"

int nvrm_mthd_context_list_devices(struct nvrm_context *ctx, uint32_t handle, uint32_t *gpu_id) {
	struct nvrm_mthd_context_list_devices arg = {
	};
	int res = nvrm_ioctl_call(ctx, handle, NVRM_MTHD_CONTEXT_LIST_DEVICES, &arg, sizeof arg);
	int i;
	if (res)
		return res;
	for (i = 0; i < 32; i++)
		gpu_id[i] = arg.gpu_id[i];
	return 0;
}

int nvrm_mthd_context_enable_device(struct nvrm_context *ctx, uint32_t handle, uint32_t gpu_id) {
	struct nvrm_mthd_context_enable_device arg = {
		.gpu_id = gpu_id,
		.unk04 = { 0xffffffff },	
	};
	return nvrm_ioctl_call(ctx, handle, NVRM_MTHD_CONTEXT_ENABLE_DEVICE, &arg, sizeof arg);
}

int nvrm_mthd_context_disable_device(struct nvrm_context *ctx, uint32_t handle, uint32_t gpu_id) {
	struct nvrm_mthd_context_disable_device arg = {
		.gpu_id = gpu_id,
		.unk04 = { 0xffffffff },	
	};
	return nvrm_ioctl_call(ctx, handle, NVRM_MTHD_CONTEXT_DISABLE_DEVICE, &arg, sizeof arg);
}

int nvrm_device_get_chipset(struct nvrm_device *dev, uint32_t *major, uint32_t *minor, uint32_t *stepping) {
	struct nvrm_mthd_subdevice_get_chipset arg;
	int res = nvrm_ioctl_call(dev->ctx, dev->osubdev, NVRM_MTHD_SUBDEVICE_GET_CHIPSET, &arg, sizeof arg);
	if (res)
		return res;
	if (major) *major = arg.major;
	if (minor) *minor = arg.minor;
	if (stepping) *stepping = arg.stepping;
	return 0;
}
