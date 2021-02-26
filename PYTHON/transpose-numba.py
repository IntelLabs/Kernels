#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#*******************************************************************
#
# NAME:    transpose
#
# PURPOSE: This program measures the time for the transpose of a
#          column-major stored matrix into a row-major stored matrix.
#
# USAGE:   Program input is the matrix order and the number of times to
#          repeat the operation:
#
#          transpose <# iterations> <matrix_size>
#
#          The output consists of diagnostics to make sure the
#          transpose worked and timing statistics.
#
# HISTORY: Written by  Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#          Numba version by Babu Pillai, June 2020.
# *******************************************************************

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import perf_counter as timer
else:
    from timeit import default_timer as timer
import numpy
print('Numpy version  = ', numpy.version.version)
import numba
print('Numba version  = ', numba.__version__)

def main():

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ') #, PRKVERSION
    print('Python Numpy Matrix transpose: B = A^T')

    if len(sys.argv) not in [3,4]:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./transpose <# iterations> <matrix order> [<N,R,C>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])
    if order < 1:
        sys.exit("ERROR: order must be >= 1")
    style = 0
    if len(sys.argv)==4:
        style = {"N":2, "R":0, "C":1}[sys.argv[3]]

    print('Number of iterations = ', iterations)
    print('Matrix order         = ', order)
    print('style (0=row, 1=column, 2=numpy) = ', style)

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    A = numpy.fromfunction(lambda i,j: i*order+j, (order,order), dtype=float)
    B = numpy.zeros((order,order))

    @numba.njit(parallel=True)
    def do_it(A,B,iters, style):
        if style==0:
            for k in range(0,iters):
                for i in numba.prange(A.shape[0]):
                    for j in range(A.shape[0]):
                        B[i,j] += A[j,i]
                        A[j,i] += 1.0
        elif style==1:
            for k in range(0,iters):
                for i in numba.prange(A.shape[0]):
                    for j in range(A.shape[0]):
                        B[j,i] += A[i,j]
                        A[i,j] += 1.0
        else:
            for k in range(0,iters):
                B += A.T
                A += 1.0


    do_it(A,B,1, style)
    t0=timer()
    do_it(A,B,iterations, style)
    t1 = timer()
    trans_time = t1 - t0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    A = numpy.fromfunction(lambda i,j: ((iterations/2.0)+(order*j+i))*(iterations+1.0), (order,order), dtype=float)
    abserr = numpy.linalg.norm(numpy.reshape(B-A,order*order),ord=1)

    epsilon=1.e-8
    nbytes = 2 * order**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if abserr < epsilon:
        print('Solution validates')
        avgtime = trans_time/iterations
        print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime)
    else:
        print('error ',abserr, ' exceeds threshold ',epsilon)
        sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()

