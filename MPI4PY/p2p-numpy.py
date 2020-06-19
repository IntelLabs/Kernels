#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
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
# NAME:    Pipeline
#
# PURPOSE: This program tests the efficiency with which point-to-point
#          synchronization can be carried out. It does so by executing
#          a pipelined algorithm on an m*n grid. The first array dimension
#          is distributed among the threads (stripwise decomposition).
#
# USAGE:   The program takes as input the
#          dimensions of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <m> <n>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - Converted to Python by Jeff Hammond, February 2016.
#          - Converted to MPI4PY by Babu Pillai, June 2020.
#
# *******************************************************************

import sys
if sys.version_info >= (3, 3):
    from time import perf_counter as timer
else:
    from timeit import default_timer as timer
import numpy
from mpi4py import MPI

def main():

    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_id = comm.Get_rank()

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    m,n = 0,0
    iterations=0
    bs = 1

    if my_id==0:
        print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
        print('Numpy version  = ', numpy.version.version)
        print('Parallel Research Kernels version ')#, PRKVERSION
        print('Python pipeline execution on 2D grid')

        if len(sys.argv) not in [4,5]:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: python -m mpi4py p2p-numpy.py <# iterations> <first array dimension> <second array dimension> [<batch>]")

        iterations = int(sys.argv[1])
        if iterations < 1:
            sys.exit("ERROR: iterations must be >= 1")

        m = int(sys.argv[2])
        if m < 1:
            sys.exit("ERROR: array dimension must be >= 1")

        n = int(sys.argv[3])
        if n < 1:
            sys.exit("ERROR: array dimension must be >= 1")

        if len(sys.argv) > 4:
            bs = int(sys.argv[4])
        if bs<0:
            sys.exit("ERROR: Batch size must be positive, or zero to indicate all at once")

        print('Number of ranks:         = ', num_procs)
        print('Grid sizes               = ', m, n)
        print('Number of iterations     = ', iterations)
        print('Batch size               = ', bs)

    m = comm.bcast(m, root=0)
    n = comm.bcast(n, root=0)
    bs = comm.bcast(bs, root=0)
    iterations = comm.bcast(iterations, root=0)

    width = (m-1)//num_procs
    leftover = (m-1)%num_procs
    if my_id < leftover:
        start = (width+1)* my_id + 1
        width+=1
    else:
        start = width*my_id+leftover+1

    grid = numpy.zeros((width+1,n))
    grid[:,0] = list(range(start-1,start+width))
    if my_id==0:
        grid[0,:] = list(range(0,n))

    numbatches = (n-1+bs-1)//bs
    for k in range(iterations+1):
        if k==1:
           comm.Barrier()
           t0 = timer()
        for b in range(numbatches):
            js = b*bs+1
            je = min(n,js+bs)
            if my_id>0:
                # get data from neighbor
                grid[0,js:je] = comm.recv(source=my_id-1)
            for i in range(1,width+1):
                for j in range(js,je):
                    grid[i,j] = grid[i-1,j] + grid[i,j-1] - grid[i-1,j-1]
            if my_id<num_procs-1:
                # send data to neighbor
                comm.send(grid[width,js:je], dest=my_id+1)

        # copy top right corner value to bottom left corner to create dependency
        if num_procs>1:
            if my_id==num_procs-1:
                corner_val = -grid[width,n-1]
                comm.send(corner_val, dest=0)
            if my_id==0:
                grid[0,0] = comm.recv(source=num_procs-1)
        else:
            grid[0,0] = -grid[m-1,n-1]


    t1 = timer()
    pipeline_time = t1 - t0
    pipeline_time = comm.reduce(pipeline_time, op=MPI.MAX, root=0)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    if my_id==0:
        epsilon=1.e-8

        # verify correctness, using top right value
        corner_val = float((iterations+1)*(n+m-2))
        if (abs((-grid[0,0]) - corner_val)/corner_val) < epsilon:
            print('Solution validates')
            avgtime = pipeline_time/iterations
            print('Rate (MFlops/s): ',1.e-6*2*(m-1)*(n-1)/avgtime,' Avg time (s): ',avgtime)
        else:
            print('ERROR: checksum ',-grid[0,0],' does not match verification value', corner_val)
            sys.exit()


if __name__ == '__main__':
    main()

