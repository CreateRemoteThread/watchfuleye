#!/usr/bin/env python3 

import os
import numpy as np
import matplotlib.pyplot as plt

SLICE_LEFT = 5000

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
    # ghetto normalize
    self.sampleBuffer = self.sampleBuffer / np.max(np.abs(self.sampleBuffer))*1000
    plt.plot(self.sampleBuffer)
    plt.show()
    return [self.sampleBuffer]


if __name__ == "__main__":
  print("em.py: this is not meant to be called directly.")
