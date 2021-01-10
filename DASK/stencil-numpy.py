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
#
#
# *******************************************************************
#
# NAME:    Stencil
#
# PURPOSE: This program tests the efficiency with which a space-invariant,
#          linear, symmetric filter (stencil) can be applied to a square
#          grid or image.
#
# USAGE:   The program takes as input the linear
#          dimension of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <grid size>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, February 2016.
#          - Distributed on Dask by Babu Pillai, August 2020.
#
# *******************************************************************

import sys
from time import perf_counter as timer
import numpy 

def factor(n):
    for i in range(int((n+1)**0.5),0,-1):
        if n%i==0:
            return i, int(n/i)

def stencil(A, W, pattern, r):
    (nx, ny) = A.shape
    #print(A.shape)
    if nx<=2*r or ny<=2*r:
        return numpy.zeros((1,1))
    B = numpy.zeros((nx-2*r, ny-2*r))
    if pattern == 'star':
        if r==2:
            B += W[2,2] * A[2:nx-2,2:ny-2] \
               + W[2,0] * A[2:nx-2,0:ny-4] \
               + W[2,1] * A[2:nx-2,1:ny-3] \
               + W[2,3] * A[2:nx-2,3:ny-1] \
               + W[2,4] * A[2:nx-2,4:ny-0] \
               + W[0,2] * A[0:nx-4,2:ny-2] \
               + W[1,2] * A[1:nx-3,2:ny-2] \
               + W[3,2] * A[3:nx-1,2:ny-2] \
               + W[4,2] * A[4:nx-0,2:ny-2]
        else:
            bx = nx-r
            by = ny-r
            B += W[r,r] * A[r:bx,r:by]
            for s in range(1,r+1):
                B += W[r,r-s] * A[r:bx,r-s:by-s] \
                   + W[r,r+s] * A[r:bx,r+s:by+s] \
                   + W[r-s,r] * A[r-s:bx-s,r:by] \
                   + W[r+s,r] * A[r+s:bx+s,r:by]
    else: # stencil
        if r>0:
            bx = nx-r
            by = ny-r
            for s in range(-r, r+1):
                for t in range(-r, r+1):
                    B += W[r+t,r+s] * A[r+t:bx+t,r+s:by+s]
    return B


def main():

    # ********************************************************************
    # initialize Dask
    # ********************************************************************

    from dask.distributed import Client
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
    print('Python stencil execution on 2D grid')

    if len(sys.argv) not in [3,4,5,6]:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./stencil <# iterations> <array dimension> [<star/stencil> <radius> <# workers>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    if len(sys.argv) > 3:
        if sys.argv[3]!='star':
            sys.exit("ERROR: Only supports star pattern")
    pattern = 'star'

    if len(sys.argv) > 4:
        r = int(sys.argv[4])
        if r < 1:
            sys.exit("ERROR: Stencil radius should be positive")
        if (2*r+1) > n:
            sys.exit("ERROR: Stencil radius exceeds grid size")
    else:
        r = 2 # radius=2 is what other impls use right now

    nworkers = cores
    if len(sys.argv) > 5:
        nworkers = int(sys.argv[5])
    if nworkers<1:
        sys.exit("ERROR: Num workers must be >=1")
    if nworkers>cores:
        sys.exit("ERROR: Too many workers requested")

    nwx, nwy = factor(nworkers)
    minx = int(n/nwx)
    miny = int(n/nwy)
    if 2*r+1>minx or 2*r+1>miny:
        sys.exit("ERROR: min tile dimensions %d*%d is smaller than stencil"%(minx,miny))

    print('Grid size            = ', n)
    print('Tiles                =  %d*%d'%(nwx,nwy))
    print('Radius of stencil    = ', r)
    print('Type of stencil      =  star')
    print('Data type            =  double precision')
    print('Compact representation of stencil loop body')
    print('Number of iterations = ', iterations)

    # ********************************************************************
    # ** Initialize workers, matrices.
    # ********************************************************************

    # there is certainly a more Pythonic way to initialize W,
    # but it will have no impact on performance.
    W = numpy.zeros(((2*r+1),(2*r+1)))
    if pattern == 'star':
        stencil_size = 4*r+1
        for i in range(1,r+1):
            W[r,r+i] = +1./(2*i*r)
            W[r+i,r] = +1./(2*i*r)
            W[r,r-i] = -1./(2*i*r)
            W[r-i,r] = -1./(2*i*r)

    else:
        stencil_size = (2*r+1)**2
        for j in range(1,r+1):
            for i in range(-j+1,j):
                W[r+i,r+j] = +1./(4*j*(2*j-1)*r)
                W[r+i,r-j] = -1./(4*j*(2*j-1)*r)
                W[r+j,r+i] = +1./(4*j*(2*j-1)*r)
                W[r-j,r+i] = -1./(4*j*(2*j-1)*r)

            W[r+j,r+j]    = +1./(4*j*r)
            W[r-j,r-j]    = -1./(4*j*r)

    csx = [minx+1 if x<n%nwx else minx for x in range(nwx)]
    csy = [miny+1 for x in range(n%nwy)]+[miny for x in range(nwy-n%nwy)]
    csA = ( tuple(csx), tuple(csy) )
    csx[0] -= r
    csx[-1] -= r
    csy[0] -= r
    csy[-1] -= r
    csB = ( tuple(csx), tuple(csy) )

    import dask
    import dask.array as da

    A = da.fromfunction(lambda i,j: i+j, shape=(n,n), dtype=float, chunks=csA)
    B = da.zeros(shape=(n-2*r,n-2*r), chunks=csB)

    # ********************************************************************
    # ** Run iterations.
    # ********************************************************************


    for k in range(iterations+1):
        # start timer after a warmup iteration
        if k==1:
            A, B = dask.persist(A, B)
            B[0,0].compute()
            t0 = timer()

        B += da.overlap.overlap(A, depth=r, boundary='none').map_blocks(lambda x: stencil(x,W,pattern,r),chunks=csB)
        A += 1.0

    A, B = dask.persist(A, B)
    B[0,0].compute()
    t1 = timer()
    stencil_time = t1 - t0
    print (stencil_time)

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    active_points = (n-2*r)**2
    norm = da.linalg.norm(da.reshape(B,shape=(active_points,1)),ord=1).compute()
    norm /= active_points

    epsilon=1.e-8

    # verify correctness
    reference_norm = 2*(iterations+1)
    if abs(norm-reference_norm) < epsilon:
        print('Solution validates')
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        print('Rate (MFlops/s): ',1.e-6*flops/avgtime, ' Avg time (s): ',avgtime)
    else:
        print('ERROR: L1 norm = ', norm,' Reference L1 norm = ', reference_norm)
        sys.exit()


if __name__ == '__main__':
    main()

