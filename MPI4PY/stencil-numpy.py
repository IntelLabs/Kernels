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

def factor(n):
    for i in range(int((n+1)**0.5),0,-1):
        if n%i==0:
            return i, int(n/i)

def main():

    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_id = comm.Get_rank()

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    r,n = 0,0
    iterations=0
    pattern = 'star'

    if my_id==0:
        print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
        print('Numpy version  = ', numpy.version.version)
        print('Parallel Research Kernels version ') #, PRKVERSION
        print('Python stencil execution on 2D grid')

        if len(sys.argv) < 3:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: python -m mpi4y stencil-numpy.py <# iterations> <array dimension> [<star/stencil> <radius>]")

        iterations = int(sys.argv[1])
        if iterations < 1:
            sys.exit("ERROR: iterations must be >= 1")

        n = int(sys.argv[2])
        if n < 1:
            sys.exit("ERROR: array dimension must be >= 1")

        if len(sys.argv) > 3:
            if sys.argv[3]!='star':
                sys.exit("ERROR: Only supports star pattern")

        if len(sys.argv) > 4:
            r = int(sys.argv[4])
            if r < 1:
                sys.exit("ERROR: Stencil radius should be positive")
            if (2*r+1) > n:
                sys.exit("ERROR: Stencil radius exceeds grid size")
        else:
            r = 2 # radius=2 is what other impls use right now

        nwx, nwy = factor(num_procs)
        minx = int(n/nwx)
        miny = int(n/nwy)
        if 2*r+1>minx or 2*r+1>miny:
            sys.exit("ERROR: min tile dimensions %d*%d is smaller than stencil"%(minx,miny))

        print('Number of ranks:     = ', num_procs)
        print('Grid size            = ', n)
        print('Tiles                = %d*%d'%(nwx,nwy))
        print('Radius of stencil    = ', r)
        print('Type of stencil      = star')

        print('Data type            = double precision')
        print('Compact representation of stencil loop body')
        print('Number of iterations = ', iterations)

    r = comm.bcast(r, root=0)
    n = comm.bcast(n, root=0)
    iterations = comm.bcast(iterations, root=0)

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

    nwx, nwy = factor(num_procs)
    idx = my_id%nwx
    idy = int(my_id/nwx)
    left_nbr, right_nbr, top_nbr, bot_nbr = my_id-1, my_id+1, my_id-nwx, my_id+nwx # left, right, up, down
    width = int(n/nwx)
    leftover = n%nwx
    if idx < leftover:
        istart = (width+1)*idx
        iend = istart+width
    else:
        istart = (width+1)*leftover + width*(idx - leftover)
        iend = istart + width-1
    width = iend-istart+1
    height = int(n/nwy)
    leftover = n%nwy
    if idy < leftover:
        jstart = (height+1)*idy
        jend = jstart+height
    else:
        jstart = (height+1)*leftover + height*(idy-leftover)
        jend = jstart+height-1
    height = jend-jstart+1
    
    A = numpy.zeros((width+2*r,height+2*r))
    B = numpy.zeros((width,height))
    for i in range(r, r+width):
        for j in range(r, r+height):
            A[i,j] = i+j+istart+jstart


    h = height
    w = width
    i_s = max(r, istart)-istart
    i_e = min(n-r, iend+1)-istart
    j_s = max(r, jstart)-jstart
    j_e = min(n-r, jend+1)-jstart

    # allocate buffers for nonblocking send and receive
    top_s = numpy.zeros((w,r))
    top_r = numpy.zeros((w,r))
    bot_s = numpy.zeros((w,r))
    bot_r = numpy.zeros((w,r))
    left_s = numpy.zeros((r,h))
    left_r = numpy.zeros((r,h))
    right_s = numpy.zeros((r,h))
    right_r = numpy.zeros((r,h))

    for k in range(iterations+1):
        # start timer after a warmup iteration
        if k==1:
           comm.Barrier()
           t0 = timer()

        # Communication Phase
        # Send data in y direction
        if idy>0:
            req1 = comm.Irecv(top_r,source=top_nbr)
            top_s[:,:] = A[r:w+r,r:2*r]
            req2 = comm.Isend(top_s,dest=top_nbr)
        if idy < nwy-1:
            req3 = comm.Irecv(bot_r,source=bot_nbr)
            bot_s[:,:] = A[r:w+r,h:h+r]
            req4 = comm.Isend(bot_s, dest=bot_nbr)
        if idy>0:
            req1.Wait()
            req2.Wait()
            A[r:w+r,0:r] = top_r
        if idy<nwy-1:
            req3.Wait()
            req4.Wait()
            A[r:w+r,h+r:] = bot_r
        # Send data in x direction
        if idx>0:
            req1 = comm.Irecv(left_r,source=left_nbr)
            left_s[:,:] = A[r:2*r,r:h+r]
            req2 = comm.Isend(left_s,dest=left_nbr)
        if idx < nwx-1:
            req3 = comm.Irecv(right_r,source=right_nbr)
            right_s[:,:] = A[w:w+r,r:h+r]
            req4 = comm.Isend(right_s, dest=right_nbr)
        if idx>0:
            req1.Wait()
            req2.Wait()
            A[0:r,r:h+r] = left_r
        if idx<nwx-1:
            req3.Wait()
            req4.Wait()
            A[w+r:,r:h+r] = right_r


        # Local Compute Phase 
        if pattern == 'star':
            if r==2:
                B[i_s:i_e,j_s:j_e] += W[2,2] * A[i_s+2:i_e+2,j_s+2:j_e+2] \
                                    + W[2,0] * A[i_s+2:i_e+2,j_s+0:j_e+0] \
                                    + W[2,1] * A[i_s+2:i_e+2,j_s+1:j_e+1] \
                                    + W[2,3] * A[i_s+2:i_e+2,j_s+3:j_e+3] \
                                    + W[2,4] * A[i_s+2:i_e+2,j_s+4:j_e+4] \
                                    + W[0,2] * A[i_s+0:i_e+0,j_s+2:j_e+2] \
                                    + W[1,2] * A[i_s+1:i_e+1,j_s+2:j_e+2] \
                                    + W[3,2] * A[i_s+3:i_e+3,j_s+2:j_e+2] \
                                    + W[4,2] * A[i_s+4:i_e+4,j_s+2:j_e+2]
            else:
                B[i_s:i_e,j_s:j_e] += W[r,r] * A[i_s+r:i_e+r,j_s+r:j_e+r]
                for s in range(1,r+1):
                    B[i_s:i_e,j_s:j_e] += W[r,r-s] * A[i_s+r:i_e+r,j_s+r-s:j_e+r-s] \
                                        + W[r,r+s] * A[i_s+r:i_e+r,j_s+r+s:j_e+r+s] \
                                        + W[r-s,r] * A[i_s+r-s:i_e+r-s,j_s+r:j_e+r] \
                                        + W[r+s,r] * A[i_s+r+s:i_e+r+s,j_s+r:j_e+r]
        else: # stencil
            if r>0:
                for s in range(-r, r+1):
                    for t in range(-r, r+1):
                        B[i_s:i_e,j_s:j_e] += W[r+t,r+s] * A[i_s+r+t:i_e+r+t,j_s+r+s:j_e+r+s]

        A[r:r+w,r:r+h] += 1.0

    t1 = timer()
    stencil_time = t1 - t0
    stencil_time = comm.reduce(stencil_time, op=MPI.MAX, root=0)

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    norm = numpy.linalg.norm(numpy.reshape(B,w*h),ord=1)
    norm = comm.reduce(norm, op=MPI.SUM, root=0)

    if my_id==0:
        active_points = (n-2*r)**2
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

