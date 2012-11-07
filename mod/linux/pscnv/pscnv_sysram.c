#include "drmP.h"
#include "drm.h"
#include "nouveau_drv.h"
#include "pscnv_mem.h"
#include <linux/list.h>
#include <linux/kernel.h>
#include <linux/mutex.h>
#include <linux/gfp.h>

int
pscnv_sysram_alloc(struct pscnv_bo *bo)
{
	int numpages, i, j;
	gfp_t gfp_flags;
	numpages = bo->size >> PAGE_SHIFT;
	if (numpages > 1 && bo->flags & PSCNV_GEM_CONTIG)
		return -EINVAL;
	bo->pages = kmalloc(numpages * sizeof *bo->pages, GFP_KERNEL);
	if (!bo->pages)
		return -ENOMEM;
	bo->dmapages = kmalloc(numpages * sizeof *bo->dmapages, GFP_KERNEL);
	if (!bo->dmapages) {
		kfree(bo->pages);
		return -ENOMEM;
	}
	if (bo->dev->pdev->dma_mask > 0xffffffff)
		gfp_flags = GFP_KERNEL;
	else
		gfp_flags = GFP_DMA32;
	for (i = 0; i < numpages; i++) {
		bo->pages[i] = alloc_pages(gfp_flags, 0);
		if (!bo->pages[i]) {
			for (j = 0; j < i; j++)
				put_page(bo->pages[j]);
			kfree(bo->pages);
			kfree(bo->dmapages);
			return -ENOMEM;
		}
	}
	for (i = 0; i < numpages; i++) {
		bo->dmapages[i] = pci_map_page(bo->dev->pdev, bo->pages[i], 0, PAGE_SIZE, PCI_DMA_BIDIRECTIONAL);
		if (pci_dma_mapping_error(bo->dev->pdev, bo->dmapages[i])) {
			for (j = 0; j < i; j++)
				pci_unmap_page(bo->dev->pdev, bo->dmapages[j], PAGE_SIZE, PCI_DMA_BIDIRECTIONAL);
			for (j = 0; j < numpages; j++)
				put_page(bo->pages[j]);
			kfree(bo->pages);
			kfree(bo->dmapages);
			return -ENOMEM;
		}
	}
	return 0;
}

int
pscnv_sysram_free(struct pscnv_bo *bo)
{
	int numpages, i;
	numpages = bo->size >> PAGE_SHIFT;
	for (i = 0; i < numpages; i++)
		pci_unmap_page(bo->dev->pdev, bo->dmapages[i], PAGE_SIZE, PCI_DMA_BIDIRECTIONAL);
	for (i = 0; i < numpages; i++)
		put_page(bo->pages[i]);
	kfree(bo->pages);
	kfree(bo->dmapages);
	return 0;
}

extern int pscnv_sysram_vm_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	struct drm_gem_object *obj = vma->vm_private_data;
	struct pscnv_bo *bo = obj->driver_private;
	uint64_t offset = (uint64_t)vmf->virtual_address - vma->vm_start;
	struct page *res;
	if (offset > bo->size)
		return VM_FAULT_SIGBUS;
	res = bo->pages[offset >> PAGE_SHIFT];
	get_page(res);
	vmf->page = res;
	return 0;
}
