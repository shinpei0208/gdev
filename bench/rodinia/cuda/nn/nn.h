#ifndef __NN_H__
#define __NN_H__

#define MAX_ARGS 10
#define REC_LENGTH 49 // size of a record in db
#define LATITUDE_POS 28 // position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors

typedef struct latLong {
	float lat;
	float lng;
} LatLong;

typedef struct record {
	char recString[REC_LENGTH];
	float distance;
} Record;


#endif
