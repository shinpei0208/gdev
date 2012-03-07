#ifndef __PSCNV_IOCTL_H__
#define __PSCNV_IOCTL_H__

extern int pscnv_ioctl_getparam(struct drm_device *, void *data, struct drm_file *);
int pscnv_ioctl_gem_new(struct drm_device *dev, void *data,	struct drm_file *file_priv);
int pscnv_ioctl_gem_info(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vspace_new(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vspace_free(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vspace_map(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vspace_unmap(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_chan_new(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_chan_free(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_obj_vdma_new(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_obj_eng_new(struct drm_device *dev, void *data,	struct drm_file *file_priv);
int pscnv_ioctl_fifo_init(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_fifo_init_ib(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vm_read32(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vm_write32(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vm_read(struct drm_device *dev, void *data,	struct drm_file *file_priv);
int pscnv_ioctl_vm_write(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vm_map(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_vm_unmap(struct drm_device *dev, void *data, struct drm_file *file_priv);
int pscnv_ioctl_phys_getaddr(struct drm_device *dev, void *data, struct drm_file *file_priv);

extern void pscnv_chan_cleanup(struct drm_device *dev, struct drm_file *file_priv);
extern void pscnv_vspace_cleanup(struct drm_device *dev, struct drm_file *file_priv);
/* XXX: nuke it from here */
struct pscnv_chan *pscnv_get_chan(struct drm_device *dev, struct drm_file *file_priv, int cid);

#endif
