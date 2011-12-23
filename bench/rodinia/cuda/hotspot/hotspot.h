#ifndef __HOTSPOT_H__
#define __HOTSPOT_H__

#define BLOCK_SIZE 16
#define STR_SIZE 256

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#endif
