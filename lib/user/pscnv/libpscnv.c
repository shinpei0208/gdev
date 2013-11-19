#include <string.h>
#include <stdint.h>
#include <sys/errno.h>
#include <sys/ioctl.h>
#include "drm.h"
#include "libpscnv.h"
#include "pscnv_drm.h"

int drmIoctl(int fd, unsigned long request, void *arg)
{
	int	ret;
	
	do {
		ret = ioctl(fd, request, arg);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));
	return ret;
}

/**
 * Send a device-specific write command.
 *
 * \param fd file descriptor.
 * \param drmCommandIndex command index 
 * \param data source pointer of the data to be written.
 * \param size size of the data to be written.
 * 
 * \return zero on success, or a negative value on failure.
 * 
 * \internal
 * It issues a write ioctl given by 
 * \code DRM_COMMAND_BASE + drmCommandIndex \endcode.
 */
int drmCommandWrite(int fd, unsigned long drmCommandIndex, void *data,
					unsigned long size)
{
	unsigned long request;
	
	request = DRM_IOC( DRM_IOC_WRITE, DRM_IOCTL_BASE, 
					   DRM_COMMAND_BASE + drmCommandIndex, size);
	
	if (drmIoctl(fd, request, data)) {
		return -errno;
	}
	return 0;
}


/**
 * Send a device-specific read-write command.
 *
 * \param fd file descriptor.
 * \param drmCommandIndex command index 
 * \param data source pointer of the data to be read and written.
 * \param size size of the data to be read and written.
 * 
 * \return zero on success, or a negative value on failure.
 * 
 * \internal
 * It issues a read-write ioctl given by 
 * \code DRM_COMMAND_BASE + drmCommandIndex \endcode.
 */
int drmCommandWriteRead(int fd, unsigned long drmCommandIndex, void *data,
						unsigned long size)
{
	unsigned long request;
	
	request = DRM_IOC( DRM_IOC_READ | DRM_IOC_WRITE, DRM_IOCTL_BASE, 
					   DRM_COMMAND_BASE + drmCommandIndex, size);
	
	if (drmIoctl(fd, request, data))
		return -errno;
	return 0;
}

int pscnv_getparam(int fd, uint64_t param, uint64_t *value) {
	int ret;
	struct drm_pscnv_getparam req;
	req.param = param;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_GETPARAM, &req, sizeof(req));
	if (ret)
		return ret;
	if (value)
		*value = req.value;
	return 0;
}

int pscnv_gem_new(int fd, uint32_t cookie, uint32_t flags, uint32_t tile_flags, uint64_t size, uint32_t *user, uint32_t *handle, uint64_t *map_handle) {
	int ret;
	struct drm_pscnv_gem_info req;
	req.cookie = cookie;
	req.flags = flags;
	req.tile_flags = tile_flags;
	req.size = size;
	if (user)
		memcpy(req.user, user, sizeof(req.user));
	ret = drmCommandWriteRead(fd, DRM_PSCNV_GEM_NEW, &req, sizeof(req));
	if (ret)
		return ret;
	if (handle)
		*handle = req.handle;
	if (map_handle)
		*map_handle = req.map_handle;
	return 0;
}

int pscnv_gem_info(int fd, uint32_t handle, uint32_t *cookie, uint32_t *flags, uint32_t *tile_flags, uint64_t *size, uint64_t *map_handle, uint32_t *user) {
	int ret;
	struct drm_pscnv_gem_info req;
	req.handle = handle;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_GEM_INFO, &req, sizeof(req));
	if (ret)
		return ret;
	if (cookie)
		*cookie = req.cookie;
	if (flags)
		*flags = req.flags;
	if (tile_flags)
		*tile_flags = req.tile_flags;
	if (size)
		*size = req.size;
	if (map_handle)
		*map_handle = req.map_handle;
	if (user)
		memcpy(user, req.user, sizeof(req.user));
	return 0;
}

int pscnv_gem_close(int fd, uint32_t handle) {
	struct drm_gem_close req;
	req.handle = handle;
	return drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &req);
}

int pscnv_gem_flink(int fd, uint32_t handle, uint32_t *name) {
	int ret;
	struct drm_gem_flink req;
	req.handle = handle;
	ret = drmIoctl(fd, DRM_IOCTL_GEM_FLINK, &req);
	if (ret)
		return ret;
	if (name)
		*name = req.name;
	return 0;
}

int pscnv_gem_open(int fd, uint32_t name, uint32_t *handle, uint64_t *size) {
	int ret;
	struct drm_gem_open req;
	req.name = name;
	ret = drmIoctl(fd, DRM_IOCTL_GEM_OPEN, &req);
	if (ret)
		return ret;
	if (handle)
		*handle = req.handle;
	if (size)
		*size = req.size;
	return 0;
}

int pscnv_vspace_new(int fd, uint32_t *vid) {
	int ret;
	struct drm_pscnv_vspace_req req;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VSPACE_NEW, &req, sizeof(req));
	if (ret)
		return ret;
	if (vid)
		*vid = req.vid;
	return 0;
}

int pscnv_vspace_free(int fd, uint32_t vid) {
	struct drm_pscnv_vspace_req req;
	req.vid = vid;
	return drmCommandWriteRead(fd, DRM_PSCNV_VSPACE_FREE, &req, sizeof(req));
}

int pscnv_vspace_map(int fd, uint32_t vid, uint32_t handle, uint64_t start, uint64_t end, uint32_t back, uint32_t flags, uint64_t *offset) {
	int ret;
	struct drm_pscnv_vspace_map req;
	req.vid = vid;
	req.handle = handle;
	req.start = start;
	req.end = end;
	req.back = back;
	req.flags = flags;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VSPACE_MAP, &req, sizeof(req));
	if (ret)
		return ret;
	if (offset)
		*offset = req.offset;
	return 0;
}

int pscnv_vspace_unmap(int fd, uint32_t vid, uint64_t offset) {
	struct drm_pscnv_vspace_unmap req;
	req.vid = vid;
	req.offset = offset;
	return drmCommandWriteRead(fd, DRM_PSCNV_VSPACE_UNMAP, &req, sizeof(req));
}

int pscnv_chan_new(int fd, uint32_t vid, uint32_t *cid, uint64_t *map_handle) {
	int ret;
	struct drm_pscnv_chan_new req;
	req.vid = vid;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_CHAN_NEW, &req, sizeof(req));
	if (ret)
		return ret;
	if (cid)
		*cid = req.cid;
	if (map_handle)
		*map_handle = req.map_handle;
	return 0;
}

int pscnv_chan_free(int fd, uint32_t cid) {
	struct drm_pscnv_chan_free req;
	req.cid = cid;
	return drmCommandWriteRead(fd, DRM_PSCNV_CHAN_FREE, &req, sizeof(req));
}

int pscnv_obj_vdma_new(int fd, uint32_t cid, uint32_t handle, uint32_t oclass, uint32_t flags, uint64_t start, uint64_t size) {
	struct drm_pscnv_obj_vdma_new req;
	req.cid = cid;
	req.handle = handle;
	req.oclass = oclass;
	req.flags = flags;
	req.start = start;
	req.size = size;
	return drmCommandWriteRead(fd, DRM_PSCNV_OBJ_VDMA_NEW, &req, sizeof(req));
}

int pscnv_fifo_init(int fd, uint32_t cid, uint32_t pb_handle, uint32_t flags, uint32_t slimask, uint64_t pb_start) {
	struct drm_pscnv_fifo_init req;
	req.cid = cid;
	req.pb_handle = pb_handle;
	req.flags = flags;
	req.slimask = slimask;
	req.pb_start = pb_start;
	return drmCommandWriteRead(fd, DRM_PSCNV_FIFO_INIT, &req, sizeof(req));
}

int pscnv_fifo_init_ib(int fd, uint32_t cid, uint32_t pb_handle, uint32_t flags, uint32_t slimask, uint64_t ib_start, uint32_t ib_order) {
	struct drm_pscnv_fifo_init_ib req;
	req.cid = cid;
	req.pb_handle = pb_handle;
	req.flags = flags;
	req.slimask = slimask;
	req.ib_start = ib_start;
	req.ib_order = ib_order;
	return drmCommandWriteRead(fd, DRM_PSCNV_FIFO_INIT_IB, &req, sizeof(req));
}

int pscnv_obj_eng_new(int fd, uint32_t cid, uint32_t handle, uint32_t oclass, uint32_t flags) {
	struct drm_pscnv_obj_eng_new req;
	req.cid = cid;
	req.handle = handle;
	req.oclass = oclass;
	req.flags = flags;
	return drmCommandWriteRead(fd, DRM_PSCNV_OBJ_ENG_NEW, &req, sizeof(req));
}

int pscnv_vm_read32(int fd, uint32_t vid, uint32_t handle, uint64_t addr, uint32_t *ptr) {
	int ret;
	struct drm_pscnv_vm_rw32 req;
	req.vid = vid;
	req.handle = handle;
	req.addr = addr;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VM_READ32, &req, sizeof(req));
	if (ret)
		return ret;
	*ptr = req.val;
	return 0;
}

int pscnv_vm_write32(int fd, uint32_t vid, uint32_t handle, uint64_t addr, uint32_t val) {
	int ret;
	struct drm_pscnv_vm_rw32 req;
	req.vid = vid;
	req.handle = handle;
	req.addr = addr;
	req.val = val;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VM_WRITE32, &req, sizeof(req));
	if (ret)
		return ret;
	return 0;
}

int pscnv_vm_read(int fd, uint32_t vid, uint32_t handle, uint64_t addr, void *buf, uint32_t size) {
	int ret;
	struct drm_pscnv_vm_rw req;
	req.vid = vid;
	req.handle = handle;
	req.addr = addr;
	req.buf_rd = buf;
	req.size = size;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VM_READ, &req, sizeof(req));
	if (ret)
		return ret;
	return 0;
}

int pscnv_vm_write(int fd, uint32_t vid, uint32_t handle, uint64_t addr, const void *buf, uint32_t size) {
	int ret;
	struct drm_pscnv_vm_rw req;
	req.vid = vid;
	req.handle = handle;
	req.addr = addr;
	req.buf_wr = buf;
	req.size = size;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VM_WRITE, &req, sizeof(req));
	if (ret)
		return ret;
	return 0;
}

int pscnv_vm_map(int fd, uint32_t vid, uint32_t handle, uint64_t *map_handle) {
	int ret;
	struct drm_pscnv_vm_map req;
	req.vid = vid;
	req.handle = handle;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VM_MAP, &req, sizeof(req));
	if (ret)
		return ret;
	*map_handle = req.map_handle;
	return 0;
}

int pscnv_vm_unmap(int fd, uint32_t vid, uint32_t handle) {
	int ret;
	struct drm_pscnv_vm_map req;
	req.vid = vid;
	req.handle = handle;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_VM_UNMAP, &req, sizeof(req));
	if (ret)
		return ret;
	return 0;
}

int pscnv_phys_getaddr(int fd, uint32_t vid, uint32_t handle, uint64_t addr, uint64_t offset, uint64_t *phys) {
	int ret;
	struct drm_pscnv_phys_getaddr req;
	req.vid = vid;
	req.handle = handle;
	req.addr = addr;
	req.offset = offset;
	ret = drmCommandWriteRead(fd, DRM_PSCNV_PHYS_GETADDR, &req, sizeof(req));
	if (ret)
		return ret;
	*phys = req.phys;
	return 0;
}
