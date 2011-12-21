#ifndef __BFS_H__
#define __BFS_H__

#define MAX_THREADS_PER_BLOCK 512

#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
#endif

/* Structure to hold a node information */
struct Node
{
	int starting;
	int no_of_edges;
};

#endif
