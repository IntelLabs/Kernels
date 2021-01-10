#!/usr/bin/env python3
#
# Copyright (c) 2017, Intel Corporation
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
# NAME:    nstream
#
# PURPOSE: To compute memory bandwidth when adding a vector of a given
#          number of double precision values to the scalar multiple of
#          another vector of the same length, and storing the result in
#          a third vector.
#
# USAGE:   The program takes as input the number
#          of iterations to loop over the triad vectors, the length of the
#          vectors, and the offset between vectors
#
#          <progname> <# iterations> <vector length> <offset> [<num_workers>]
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#          
#          Ray will connect to local cluster.  Provide address
#          parameters and database password in ray_address and 
#          ray_redis_password environment variables if needed.
#
# NOTES:   Bandwidth is determined as the number of words read, plus the
#          number of words written, times the size of the words, divided
#          by the execution time. For a vector length of N, the total
#          number of words read and written is 4*N*sizeof(double).
#
#
# HISTORY: This code is loosely based on the Stream benchmark by John
#          McCalpin, but does not follow all the Stream rules. Hence,
#          reported results should not be associated with Stream in
#          external publications
#
#          Converted to Python by Jeff Hammond, October 2017.
#          Distributed on Dask by Babu Pillai, August 2020.
#
# *******************************************************************

import sys
import numpy
from time import process_time as timer

import time
import os

class nstream:
    def __init__(self, length):
        # 0.0 is a float, which is 64b (53b of precision)
        self.A = numpy.zeros(length)
        self.B = numpy.full(length, 2.0)
        self.C = numpy.full(length, 2.0)
        self.scalar = 3.0

    def run(self, iterations):
        for k in range(0,iterations+1):
            if k==1: t0 = timer()
            self.A += self.B + self.scalar*self.C
        t1 = timer()
        return t1 - t0

    def cksum(self):
        asum = numpy.linalg.norm(self.A, ord=1)
        return asum


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
    print('Python Numpy STREAM triad: A = B + scalar * C')

    if len(sys.argv) not in [3,4,5]:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: python nstream.py <# iterations> <vector length> <offset> [num Workers>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    total_length = int(sys.argv[2])
    if total_length < 1:
        sys.exit("ERROR: length must be positive")

    #offset = int(sys.argv[3])
    #if offset < 0:
    #    sys.exit("ERROR: offset must be nonnegative")

    nworkers = int(cores)
    if len(sys.argv)==5:
        nworkers = int(sys.argv[4])
    if nworkers<1:
        sys.exit("ERROR: Num workers must be >=1")
    if nworkers>cores:
        sys.exit("ERROR: Too many workers requested")
    if nworkers>total_length:
        sys.exit("ERROR: Length must be at least number of workers")
    length = total_length//nworkers

    print('Number of iterations = ', iterations)
    print('Vector length        = ', length*nworkers)
    print('Number of workers    = ', nworkers)
    #print('Offset               = ', offset)

    # ********************************************************************
    # ** Initialize workers, vectors.
    # ********************************************************************

    robjs = cl.gather([cl.submit(nstream, length, actor=True) for x in range(0,nworkers)])

    # ********************************************************************
    # ** Run iterations.
    # ********************************************************************

    t0 = time.time()  # wall-clock time
    fut_times = [x.run(iterations) for x in robjs]
    nstream_times = [x.result() for x in fut_times]
    print(nstream_times)
    nstream_time=max(nstream_times)
    t1 = time.time()
    nstream_time2 = t1-t0
    #print (nstream_time, nstream_time2)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    ar = 0.0
    br = 2.0
    cr = 2.0
    scalar = 3.0
    ref = 0.0
    for k in range(0,iterations+1):
        ar += br + scalar * cr
    ar *= length
    asum = robjs[0].cksum().result()

    epsilon=1.e-8
    if abs(ar-asum)/asum > epsilon:
        print('Failed Validation on output array');
        print('        Expected checksum: ',ar);
        print('        Observed checksum: ',asum);
        sys.exit("ERROR: solution did not validate")
    else:
        print('Solution validates')
        avgtime = nstream_time/iterations
        avgtime2 = nstream_time2/(iterations+1)
        nbytes = 4.0 * length * nworkers * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
        print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime, '(max worker)')
        print('Rate (MB/s): ',1.e-6*nbytes/avgtime2, ' Avg time (s): ', avgtime2, '(wall-clock)')


if __name__ == '__main__':
    main()
