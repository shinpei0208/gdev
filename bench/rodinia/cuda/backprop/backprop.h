#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff

#define GPU
#define THREADS 256
#define WIDTH 16  // shared memory width  
#define HEIGHT 16 // shared memory height

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define NUM_THREAD 4  //OpenMP threads


typedef struct {
	int input_n;                  /* number of input units */
	int hidden_n;                 /* number of hidden units */
	int output_n;                 /* number of output units */
	
	float *input_units;          /* the input units */
	float *hidden_units;         /* the hidden units */
	float *output_units;         /* the output units */
	
	float *hidden_delta;         /* storage for hidden unit error */
	float *output_delta;         /* storage for output unit error */
	
	float *target;               /* storage for target vector */
	
	float **input_weights;       /* weights from input to hidden layer */
	float **hidden_weights;      /* weights from hidden to output layer */
	
	/*** The next two are for momentum ***/
	float **input_prev_weights;  /* previous change on input to hidden wgt */
	float **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;


/*** User-level functions ***/

void bpnn_initialize();

BPNN *bpnn_create();
void bpnn_free();

void bpnn_train();
void bpnn_feedforward();

void bpnn_save();
BPNN *bpnn_read();

void bpnn_train_cuda(BPNN *net, float *eo, float *eh);

void load(BPNN *net);
void backprop_face(void);
int setup(int argc, char *argv[]);

extern int layer_size;

#endif
