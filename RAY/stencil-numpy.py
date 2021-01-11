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
#          - Parallelized on Ray by Padmanabhan Pillai, April 2020.
#
# *******************************************************************

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer
import numpy
print('Numpy version  = ', numpy.version.version)

import time
import ray
import os
import threading

def factor(n):
    for i in range(int((n+1)**0.5),0,-1):
        if n%i==0:
            return i, int(n/i)

@ray.remote(num_cpus=1)
class stencil(threading.Thread):
    def __init__(self, my_id):
        threading.Thread.__init__(self)
        self.my_id = my_id

    def init(self, robjs, n, iterations, r, pattern):
        self.robjs = robjs
        self.n = n
        self.nw = len(robjs)
        self.nwx, self.nwy = factor(self.nw)
        self.idx = self.my_id%self.nwx
        self.idy = int(self.my_id/self.nwx)
        self.nbr = [ self.my_id-1, self.my_id+1, self.my_id-self.nwx, self.my_id+self.nwx ] # left, right, up, down
        #self.ok_to_send = [ threading.Semaphore(0) for i in range(4) ]
        self.data_ready = [ threading.Semaphore(0) for i in range(4) ]
        self.sent_data  = [ [] for i in range(4) ]
        self.iterations = iterations
        self.r = r
        self.pattern = pattern
        width = int(self.n/self.nwx)
        leftover = self.n%self.nwx
        if self.idx < leftover:
            self.istart = (width+1)*self.idx
            self.iend = self.istart+width
        else:
            self.istart = (width+1)*leftover + width*(self.idx - leftover)
            self.iend = self.istart + width-1
        self.width = self.iend-self.istart+1
        height = int(self.n/self.nwy)
        leftover = self.n%self.nwy
        if self.idy < leftover:
            self.jstart = (height+1)*self.idy
            self.jend = self.jstart+height
        else:
            self.jstart = (height+1)*leftover + height*(self.idy-leftover)
            self.jend = self.jstart+height-1
        self.height = self.jend-self.jstart+1
        # there is certainly a more Pythonic way to initialize W,
        # but it will have no impact on performance.
        self.W = numpy.zeros(((2*r+1),(2*r+1)))
        if pattern == 'star':
            for i in range(1,r+1):
                self.W[r,r+i] = +1./(2*i*r)
                self.W[r+i,r] = +1./(2*i*r)
                self.W[r,r-i] = -1./(2*i*r)
                self.W[r-i,r] = -1./(2*i*r)
        else:
            for j in range(1,r+1):
                for i in range(-j+1,j):
                    self.W[r+i,r+j] = +1./(4*j*(2*j-1)*r)
                    self.W[r+i,r-j] = -1./(4*j*(2*j-1)*r)
                    self.W[r+j,r+i] = +1./(4*j*(2*j-1)*r)
                    self.W[r-j,r+i] = -1./(4*j*(2*j-1)*r)
                self.W[r+j,r+j]    = +1./(4*j*r)
                self.W[r-j,r-j]    = -1./(4*j*r)

        self.A = numpy.zeros((self.width+2*r,self.height+2*r))
        self.B = numpy.zeros((self.width,self.height))
        for i in range(r, r+self.width):
            for j in range(r, r+self.height):
                self.A[i,j] = i+j+self.istart+self.jstart
        return self.my_id

    # Note:
    #    this runs as a separate thread.  To launch this,
    #    call the start() method (from Thread)
    #    To check if done, poll using is_active() (from Thread)
    #    use wait_end() to get the results
    def run(self):
        # For convenience
        A = self.A
        B = self.B
        W = self.W
        r = self.r
        h = self.height
        w = self.width
        i_s = max(r, self.istart)-self.istart
        i_e = min(self.n-r, self.iend+1)-self.istart
        j_s = max(r, self.jstart)-self.jstart
        j_e = min(self.n-r, self.jend+1)-self.jstart

        for k in range(self.iterations+1):
            # start timer after a warmup iteration
            if k==1: t0 = timer()

            # Communication Phase
            # queues data at neighbor, indicates data ready
            if (self.idy < self.nwy-1): self.robjs[self.nbr[3]].send.remote( 3, A[r:w+r,h:h+r] )
            if (self.idy > 0):          self.robjs[self.nbr[2]].send.remote( 2, A[r:w+r,r:2*r] )
            if (self.idx < self.nwx-1): self.robjs[self.nbr[1]].send.remote( 1, A[w:w+r,r:h+r] )
            if (self.idx > 0):          self.robjs[self.nbr[0]].send.remote( 0, A[r:2*r,r:h+r] )

            # receive data, indicate ok to send for next round
            if (self.idy > 0):          A[r:w+r,0:r]  = self.recv(3)
            if (self.idy < self.nwy-1): A[r:w+r,h+r:] = self.recv(2)
            if (self.idx > 0):          A[0:r,r:h+r]  = self.recv(1)
            if (self.idx < self.nwx-1): A[w+r:,r:h+r] = self.recv(0)

            # wait for send done
            #if (self.idy < self.nwy-1): self.waitsend( 3 )
            #if (self.idy > 0):          self.waitsend( 2 )
            #if (self.idx < self.nwx-1): self.waitsend( 1 )
            #if (self.idx > 0):          self.waitsend( 0 )

            # Local Compute Phase 
            if self.pattern == 'star':
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
        self.stencil_time = t1 - t0

    # NOTE: this is a remote call
    def send(self, n, d):
        self.sent_data[n].append(d)
        self.data_ready[n].release()

    ## NOTE: this is a local call
    #def waitsend(self, n):
    #    self.ok_to_send[n].acquire()

    # NOTE: this is a local call
    def recv(self, n):
        self.data_ready[n].acquire()
        d = self.sent_data[n].pop(0)
        #self.robjs[self.nbr[n^1]].ack.remote(n)
        return d

    ## NOTE: this is a remote call
    #def ack(self, n):
    #   self.ok_to_send[n].release()

    def wait_end(self):
        self.join()
        return self.stencil_time

    def validate(self):
        norm = numpy.linalg.norm(numpy.reshape(self.B,self.width*self.height),ord=1)
        return norm

def main():

    # ********************************************************************
    # initialize Ray
    # ********************************************************************

    ray_address = os.getenv("ray_address")
    ray_redis_pass = os.getenv("ray_redis_password")
    if ray_address==None:
    	ray_address = "auto"
    if ray_redis_pass==None:
    	ray_redis_pass = ""
    try:
    	ray.init(address=ray_address, _redis_password=ray_redis_pass)
    except:
    	ray.init()
    assert ray.is_initialized() == True
    cores = ray.available_resources()['CPU']
    print ("Ray initialized;  available cores:", cores)    

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

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

    robjs = [stencil.remote(x) for x in range(0,nworkers)]
    ray.get([x.init.remote(robjs, n, iterations, r, pattern) for x in robjs])  
    # getting the results to make sure objs ready before timing

    # ********************************************************************
    # ** Run iterations.
    # ********************************************************************

    t0 = time.time()  # wall-clock time
    ray.get([x.start.remote() for x in robjs])
    # poll to see if all are done
    while sum(ray.get([x.is_alive.remote() for x in robjs]))>0:
        time.sleep(1)
    # get results (times measured by run() on each worker)
    fut_times = [x.wait_end.remote() for x in robjs]
    stencil_time = max(ray.get(fut_times))
    t1 = time.time()
    stencil_time2 = t1-t0
    print (stencil_time, stencil_time2)

    #******************************************************************************
    #* Analyze and output results.
    #******************************************************************************

    norm = sum(ray.get( [x.validate.remote() for x in robjs] ))
    active_points = (n-2*r)**2
    norm /= active_points

    epsilon=1.e-8

    if pattern == 'star':
        stencil_size = 4*r+1
    else:
        stencil_size = (2*r+1)**2

    # verify correctness
    reference_norm = 2*(iterations+1)
    if abs(norm-reference_norm) < epsilon:
        print('Solution validates')
        flops = (2*stencil_size+1) * active_points
        avgtime = stencil_time/iterations
        avgtime2 = stencil_time2/iterations
        print('Rate (MFlops/s): ',1.e-6*flops/avgtime, ' Avg time (s): ',avgtime,'(max worker)')
        print('Rate (MFlops/s): ',1.e-6*flops/avgtime2, ' Avg time (s): ',avgtime2,'(wall-clock)')
    else:
        print('ERROR: L1 norm = ', norm,' Reference L1 norm = ', reference_norm)
        sys.exit()


if __name__ == '__main__':
    main()

