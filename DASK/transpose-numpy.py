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
#          Distributed on Dask by Babu Pillai, August 2020.
# *******************************************************************



import sys
from timeit import default_timer as timer
import numpy

import time
import os


def main():

    from dask.distributed import Client
    # ********************************************************************
    # initialize Dask
    # ********************************************************************

    try:
    	cl = Client("127.0.0.1:8786",timeout=1)
    except:
    	cl = Client()
    cores = sum(cl.ncores().values())
    print ("Dask initialized;  available cores:", cores)
    
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
    print('Numpy version  = ', numpy.version.version)
    print('Parallel Research Kernels version ') #, PRKVERSION
    print('Python Numpy Matrix transpose: B = A^T')

    if len(sys.argv) not in [3,4]:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./transpose <# iterations> <matrix order> [<num workers>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])

    nworkers = int(cores)
    if len(sys.argv)==4:
        nworkers = int(sys.argv[3])
    if nworkers<1:
        sys.exit("ERROR: Num workers must be >=1")
    if nworkers>cores:
        sys.exit("ERROR: Too many workers requested")
    if order < nworkers:
        sys.exit("ERROR: order must be at least # of workers")
    if order%nworkers != 0:
        sys.exit("ERROR: order must be divisible by # of workers")

    print('Number of iterations = ', iterations)
    print('Matrix order         = ', order)
    print('Number of workers    = ', nworkers)

    # ********************************************************************
    # ** Initialize workers, matrices.
    # ********************************************************************

    import dask
    import dask.array as da

    A = da.fromfunction(lambda i,j: i*order+j, shape=(order,order), dtype=float, chunks=(order//nworkers,order))
    B = da.zeros((order,order), chunks=(order//nworkers,order))

    # ********************************************************************
    # ** Run iterations.
    # ********************************************************************

    for k in range(0,iterations+1):

        if k==1:
            #A, B = dask.persist(A, B)
            B[:,0].compute()
            t0 = timer()

        # this actually forms the transpose of A
        # B += numpy.transpose(A)
        # this only uses the transpose _view_ of A
        B += da.transpose(A)
        A += 1.0
        A, B = dask.persist(A, B)


    #A, B = dask.persist(A, B)
    B[:,0].compute()
    t1 = timer()
    trans_time = t1 - t0


    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    A = da.fromfunction(lambda i,j: ((iterations/2.0)+(order*j+i))*(iterations+1.0), shape=(order,order), dtype=float)
    abserr = da.linalg.norm(da.reshape(B-A,shape=(order*order,1)),ord=1).compute()

    epsilon=1.e-8
    nbytes = 2 * order**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if abserr < epsilon:
        print('Solution validates')
        avgtime = trans_time/iterations
        print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime, '(max worker)')
    else:
        print('error ',abserr, ' exceeds threshold ',epsilon)
        sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()

