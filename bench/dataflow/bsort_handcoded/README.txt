/*****************************************************************************
* Name        : README.txt for bitonicSort
* Author      : Jason Aumiller
*
/****************************************************************************/


Implements a bitonic sorting algorithm using the CUDA driver API as a 
micro-benchmark. 

To build:

	make

Usage: 

	bitonicSort [-vO] size | -i | -p pow

Generates an array of 'size' integers (or 2^'pow') and sorts them. If the -i 
option is used then the array is read from stdin. Prints the total gpu running 
time in microseconds. 

Options:
	-i   Read from stdin.
	-p   Specify 'size' as 2^'pow'
	-v 	 Verbose.
	-O 	 Print sorted list to stdout.

Examples: 

	./bitonicSort -p 21
	
	cat file_to_sort | ./bitonicSort -v -O


