#!/usr/bin/env python
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
#          - Numba version by Babu Pillai, June 2020.
#
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

    print('Parallel Research Kernels version ')#, PRKVERSION
    print('Python Numba pipeline execution on 2D grid')

    if len(sys.argv) != 4:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./synch_p2p <# iterations> <first array dimension> <second array dimension>")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    m = int(sys.argv[2])
    if m < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    n = int(sys.argv[3])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    print('Grid sizes               = ', m, n)
    print('Number of iterations     = ', iterations)

    grid = numpy.zeros((m,n))
    grid[0,:] = list(range(n))
    grid[:,0] = list(range(m))

    #@numba.njit(parallel=True)
    #def do_it(grid, m, n, iters):
    #    nw = numba.get_num_threads()
    #    flags = [ False for i in range(nw) ]
    #    for k in range(iters):
    #        for w in numba.prange(nw):
    #            for i in range(1,m):
    #                if w>0:
    #                    f = False
    #                    while not f: f=flags[w-1]
    #                    flags[w-1]=False
    #                for j in range(max(1,w*n//nw),(w+1)*n//nw):
    #                    grid[i,j] = grid[i-1,j] + grid[i,j-1] - grid[i-1,j-1]
    #                if w<nw-1:
    #                    f = True
    #                    while f: f=flags[w]
    #                    flags[w]=True
    #        grid[0,0] = -grid[m-1,n-1]

    #@numba.njit(parallel=True)
    #def do_it(grid, m, n, iters):
    #    nw = numba.get_num_threads()
    #    for k in range(iters):
    #        for i in range(1,m+nw-1):
    #            for w in numba.prange(nw):
    #                if i-w>0 and i-w<m:
    #                    for j in range(max(1,w*n//nw),(w+1)*n//nw):
    #                        grid[i-w,j] = grid[i-w-1,j] + grid[i-w,j-1] - grid[i-w-1,j-1]
    #        grid[0,0] = -grid[m-1,n-1]

    @numba.njit(parallel=True)
    def do_it(grid, m, n, iters):
        nw = numba.get_num_threads()
        bs = 100 #m//(4*nw)+1
        for k in range(iters):
            for ii in range((m+bs-1)//bs+nw-1):
                for w in numba.prange(nw):
                    for i in range((ii-w)*bs,(ii-w+1)*bs):
                            if i>0 and i<m:
                                for j in range(max(1,w*n//nw),(w+1)*n//nw):
                                    grid[i,j] = grid[i-1,j] + grid[i,j-1] - grid[i-1,j-1]
            grid[0,0] = -grid[m-1,n-1]

    do_it(grid, m, n, 1)
    t0 = timer()
    do_it(grid, m, n, iterations)
    t1 = timer()
    pipeline_time = t1 - t0

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    epsilon=1.e-8

    # verify correctness, using top right value
    corner_val = float((iterations+1)*(n+m-2))
    if (abs(grid[m-1,n-1] - corner_val)/corner_val) < epsilon:
        print('Solution validates')
        avgtime = pipeline_time/iterations
        print('Rate (MFlops/s): ',1.e-6*2*(m-1)*(n-1)/avgtime,' Avg time (s): ',avgtime)
    else:
        print('ERROR: checksum ',grid[m-1,n-1],' does not match verification value', corner_val)
        sys.exit()


if __name__ == '__main__':
    main()

