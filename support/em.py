#!/usr/bin/env python3 

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter,lfilter

SLICE_LEFT = 5000

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

class WaveHelper:
  def __init__(self,wave_fn):
    self.wave_fn = wave_fn
    print("WaveHelper initialized: %s" % wave_fn)
    self.peakSlices = []
    self.sampleBuffer = np.load(wave_fn)
    # plt.plot(self.sampleBuffer)
    # plt.show()
    return

  def getLabel(self):
    wfn_base = os.path.basename(self.wave_fn)
    if "-" in wfn_base:
      return wfn_base.split("-")[0]
    else:
      return wfn_base.split(".")[0]

  def findPeaks(self):
    return self.sampleBuffer

  def extractFeatures(self):
    self.sampleBuffer = self.sampleBuffer / np.max(np.abs(self.sampleBuffer))*1000
    self.sampleBuffer = butter_bandpass_filter(self.sampleBuffer,8000000,16000000,124999999,order=1)
    peaks,_ = scipy.signal.find_peaks(abs(self.sampleBuffer[65000:]),height=50,distance=4000)
    peaks = [peak + 65000 for peak in peaks]
    peakFeat = []
    peakVal = []
    for i in range(0,5):
      if i < len(peaks):
        peakFeat.append(peaks[i])
        peakVal.append(abs(self.sampleBuffer)[peaks[i]])
      else:
        peakFeat.append(0)
        peakVal.append(0)
    print([peakFeat + peakVal]) 
    return [peakFeat + peakVal]


if __name__ == "__main__":
  print("em.py: this is not meant to be called directly.")
