from multiprocessing import process
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import sample, shuffle
from scipy import ndimage, signal
import mkl_fft
import multiprocessing
import os
import gc
import scipy
import random
import warnings
from scipy.ndimage.filters import uniform_filter1d
from tqdm import tqdm

class FilterHilbert:
    filterArray = None
    samplingFreq = np.NaN
    fhResult = None
    # Protected
    _initString = None
    # Private
    
    def __init__(self):
        self.filterArray = None
        self.samplingFreq = np.NaN
        self._initString = 'Not Initialized'
        self.fhResult = None
    
    def setStrict(self):
        warnings.simplefilter('error')
    def unsetStrict(self):
        warnings.simplefilter('default')
    
    def generateFilterArray(self, notchWidthInHz, samplingFrequency, filterOrder):
    # Create our set of filters
        filterWidth = int(notchWidthInHz) #Hz
        halfFilterWidth = (filterWidth / 2)
        upperLimitFrequency = int(samplingFrequency / 2)

        self.filterArray = []
        for centerFrequency in range(0, upperLimitFrequency, filterWidth):
            beginSection = (centerFrequency - halfFilterWidth)
            endSection = (centerFrequency + halfFilterWidth)
            if(beginSection <= 0):
                beginSection = 0.1
            if(endSection >= upperLimitFrequency):
                endSection = upperLimitFrequency - 0.1
            self.filterArray.append(signal.butter(int(filterOrder), [beginSection, endSection], btype='bandpass', output='sos', fs=samplingFrequency))
        self._initString = 'Initialized'
        self.samplingFreq = samplingFrequency


    def runFilterHilbert(self, data):
        # Right now, we only accept a 2D matrix as the input data, with the samples taken along axis 0.
        
        # This method takes a LOT of memory. Should add checks here to ensure that the machine that is running this has enough.
        # memory check here

        # Setup a starmap for faster processing. Works only on Linux.

        parallelArguments = [(notchFilt, data, 0) for notchFilt in self.filterArray]
        with multiprocessing.Pool(processes=60) as p:
            mapResult = p.starmap(signal.sosfiltfilt, parallelArguments)

        res = (np.vstack(mapResult))
        # Free up memory.
        del mapResult
        gc.collect()
        # Now that we have res, we can operate on this and create the end result for this set of contacts.
        self.fhResult = self.__hilbertFast(res, axis=0)


    def calculatePhaseDifferencesSlow(self,fhSignal1,fhSignal2, cyclesToCapture, intraWindowThreshold, seed = 0, shuffles = 200):
        '''
        The algorithm for this is as follows:

        1. Start at frequency band 0.
        2. Inside frequency band 0, multiply the number of cycles you wish to capture by the sampling rate. This will give you a window to average over.
        3. From each fhSignal passed in (~200 signals) get the np.exp(1j*phase angle difference) and then run a 1d uniform filter over the band (with the window calculated in step 2)
        4. Take the absolute value of this.
        5. Should have differences that are not filtered by statistical significance.
        '''

        (bands, cols) = np.shape(fhSignal1)
        (bands2, cols2) = np.shape(fhSignal2)

        assert(bands == bands2)
        assert(cols == cols2)

        phase1 = np.angle(fhSignal1)
        phase2 = np.angle(fhSignal2)
        result = np.zeros_like(fhSignal1)
        for band in range(bands): # Good opportunity to parallelize. Each core will take one band. Don't try to be fancy and split the shuffling amongst cores as there will undoubtedly be issues with threads finishing first, having data available and making the other threads miss.
            freq = band+1
            samplesToTake = (cyclesToCapture * 200 * 1/(freq))
            result[band, :] = np.imag(np.exp(1j * (phase2[band,:] - phase1[band,:]))) # testing
            # Add shuffling here to separate random matches from actual "phase coupling"
            result[band, :] = scipy.ndimage.uniform_filter1d(result[band, :], int(samplesToTake))

        result = np.abs(result)
        return result



    def calculatePLV(self, sampleA, sampleB, intraWindowThreshold, seed=0, shuffles = 200):
        rng = np.random.default_rng(seed)
        sampleWindowSize = len(sampleA)
        shuffleSample = np.copy(sampleA)
        shuffleSamples = []
        for x in range(0, shuffles):
            np.random.shuffle(shuffleSample)
            shuffleSamples.append(np.copy(shuffleSample))
        shuffleSamples = np.reshape(shuffleSamples, (shuffles, len(shuffleSample)))
        sampleBMatrix = [sampleB for x in shuffleSamples]
        sampleBMatrix = np.reshape(sampleBMatrix, np.shape(shuffleSamples))
        histData = np.abs(np.mean(np.exp(1j * (sampleBMatrix - shuffleSamples)), axis=1))
        resData = np.abs(np.mean(np.exp(1j * (sampleB - sampleA))))
        histData.sort()
        resultIdx = self.__find_nearest_idx(histData, resData)
        place = resultIdx / (len(histData) - 1)
        # We would also like to reject things that are below a certain threshold when shuffling.
        if(place > intraWindowThreshold):
            return resData
        else:
            return 0


    def __angularSubtract(self, data1, data2):
        return np.abs(np.mean(np.exp(1j*(data1 - data2))))


    def __hilbertFast(self, x, N=None, axis=-1):
        x = np.asarray(x)
        if np.iscomplexobj(x):
            raise ValueError("x must be real.")
        if N is None:
            N = x.shape[axis]
        if N <= 0:
            raise ValueError("N must be positive.")
    
        Xf = mkl_fft.fft(x, N, axis=axis)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
    
        if x.ndim > 1:
            ind = [np.newaxis] * x.ndim
            ind[axis] = slice(None)
            h = h[tuple(ind)]
        x = mkl_fft.ifft(Xf * h, axis=axis)
        return x

    def __find_nearest_idx(self, array, value):
        array = np.asarray(array)
        idx = np.nanargmin(np.abs(array - value))
        return idx