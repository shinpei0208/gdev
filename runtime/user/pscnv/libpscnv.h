#ifndef LIBPSCNV_H
#define LIBPSCNV_H
#include <stdint.h>

#define DRM_DIR_NAME  "/dev/dri"
#define DRM_DEV_NAME  "%s/card%d"
#define DRM_IOCTL_NR(n)		_IOC_NR(n)
#define DRM_IOC_VOID		_IOC_NONE
#define DRM_IOC_READ		_IOC_READ
#define DRM_IOC_WRITE		_IOC_WRITE
#define DRM_IOC_READWRITE	_IOC_READ|_IOC_WRITE
#define DRM_IOC(dir, group, nr, size) _IOC(dir, group, nr, size)

#define PSCNV_GETPARAM_PCI_VENDOR      3
#define PSCNV_GETPARAM_PCI_DEVICE      4
#define PSCNV_GETPARAM_BUS_TYPE        5
#define PSCNV_GETPARAM_FB_SIZE         8
#define PSCNV_GETPARAM_CHIPSET_ID      11
#define PSCNV_GETPARAM_GRAPH_UNITS     13
#define PSCNV_GETPARAM_PTIMER_TIME     14
#define PSCNV_GETPARAM_MP_COUNT        100

#define PSCNV_GEM_CONTIG		0x00000001	/* needs to be contiguous in VRAM */
#define PSCNV_GEM_MAPPABLE		0x00000002	/* intended to be mmapped by host */
#define PSCNV_GEM_MEMTYPE_MASK		0x0000000c
#define PSCNV_GEM_VRAM_SMALL		0x00000000	/* VRAM with small pages */
#define PSCNV_GEM_SYSRAM_SNOOP		0x00000004
#define PSCNV_GEM_VRAM_LARGE		0x00000008	/* VRAM with large pages */
#define PSCNV_GEM_SYSRAM_NOSNOOP	0x0000000c
#define PSCNV_GEM_GART			PSCNV_GEM_SYSRAM_SNOOP	/* compat */

int pscnv_getparam(int fd, uint64_t param, uint64_t *value);
int pscnv_gem_new(int fd, uint32_t cookie, uint32_t flags, uint32_t tile_flags, uint64_t size, uint32_t *user, uint32_t *handle, uint64_t *map_handle);
int pscnv_gem_info(int fd, uint32_t handle, uint32_t *cookie, uint32_t *flags, uint32_t *tile_flags, uint64_t *size, uint64_t *map_handle, uint32_t *user);
int pscnv_gem_close(int fd, uint32_t handle);
int pscnv_gem_flink(int fd, uint32_t handle, uint32_t *name);
int pscnv_gem_open(int fd, uint32_t name, uint32_t *handle, uint64_t *size);
int pscnv_vspace_new(int fd, uint32_t *vid);
int pscnv_vspace_free(int fd, uint32_t vid);
int pscnv_vspace_map(int fd, uint32_t vid, uint32_t handle, uint64_t start, uint64_t end, uint32_t back, uint32_t flags, uint64_t *offset);
int pscnv_vspace_unmap(int fd, uint32_t vid, uint64_t offset);
int pscnv_chan_new(int fd, uint32_t vid, uint32_t *cid, uint64_t *map_handle);
int pscnv_chan_free(int fd, uint32_t cid);
int pscnv_obj_vdma_new(int fd, uint32_t cid, uint32_t handle, uint32_t oclass, uint32_t flags, uint64_t start, uint64_t size);
int pscnv_fifo_init(int fd, uint32_t cid, uint32_t pb_handle, uint32_t flags, uint32_t slimask, uint64_t pb_start);
int pscnv_fifo_init_ib(int fd, uint32_t cid, uint32_t pb_handle, uint32_t flags, uint32_t slimask, uint64_t ib_start, uint32_t ib_order);
int pscnv_obj_eng_new(int fd, uint32_t cid, uint32_t handle, uint32_t oclass, uint32_t flags);
int pscnv_vm_read32(int fd, uint32_t vid, uint32_t handle, uint64_t addr, uint32_t *ptr);
int pscnv_vm_write32(int fd, uint32_t vid, uint32_t handle, uint64_t addr, uint32_t val);
int pscnv_vm_read(int fd, uint32_t vid, uint32_t handle, uint64_t addr, void *buf, uint32_t size);
int pscnv_vm_write(int fd, uint32_t vid, uint32_t handle, uint64_t addr, const void *buf, uint32_t size);
int pscnv_vm_map(int fd, uint32_t vid, uint32_t handle, uint64_t *map_handle);
int pscnv_vm_unmap(int fd, uint32_t vid, uint32_t handle);
int pscnv_phys_getaddr(int fd, uint32_t vid, uint32_t handle, uint64_t addr, uint64_t offset, uint64_t *phys);

#define pscnv_obj_gr_new pscnv_obj_eng_new

#endif
