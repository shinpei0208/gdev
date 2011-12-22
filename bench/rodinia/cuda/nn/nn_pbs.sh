#!/bin/sh
#PBS -l nodes=2:ppn=1
#PBS -o nn_out
#PBS -j oe
#PBS -M mag3dn@virginia.edu

export OMP_NUM_THREADS=8
cd Desktop/upbench/nn_openMP
./nn filelist_4 3 30 90 
