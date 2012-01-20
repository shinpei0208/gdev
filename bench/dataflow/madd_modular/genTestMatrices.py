#!/usr/bin/python

import random

def genZeroMatrix(rows, cols):
   matrix = []

   for i in range(0, rows * cols):
      matrix.append(0)

   return matrix

def genIdentityMatrix(rows, cols):
   matrix = []

   for i in range(0, rows * cols):
      matrix.append(1)

   return matrix

def genRandomMatrix(rows, cols):
   matrix = []

   for i in range(0, rows * cols):
      matrix.append(random.randint(0, 50))

   return matrix

def main():
   rows = 4
   cols = 4

   print genZeroMatrix(rows, cols)
   print genIdentityMatrix(rows, cols)
   print genRandomMatrix(rows, cols)

if __name__ == '__main__':
   main()
