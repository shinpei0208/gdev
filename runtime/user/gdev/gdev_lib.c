/*
 * Copyright 2011 Shinpei Kato
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <sys/errno.h>
#include <sys/ioctl.h>
#include "gdev_lib.h"
#include "drm.h"

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
