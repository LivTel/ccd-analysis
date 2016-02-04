from scipy import fftpack
import numpy as np
import pylab as py

class fft:
    def __init__(self):
        pass

    def _1DFFT(self, data, fs):
        xdft = fftpack.fft(data)							# take fft
        xdft_posi_only = xdft[0:len(xdft)/2+1]					        # take only positive terms and DC offset (negative terms past n/2 + 1)
        psdx = (1./(fs*len(data)))*pow(abs(xdft_posi_only), 2)				# get magnitude
        psdx[1:len(psdx)-1] = 2*psdx[1:len(psdx)-1]					# in order to conserve total power, multiply the frequencies by two (as neg freqs disregarded)

        freq = np.arange(float(len(psdx)))
        freq *= float(fs)/float(len(data))				

        return freq, psdx

    def do1DReadoutFFT(self, data, fs):
        data_1D = np.fliplr(data).ravel()
        this_freq, this_psdx = self._1DFFT(data_1D, fs)
        return this_freq, this_psdx





