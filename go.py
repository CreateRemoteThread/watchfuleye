#!/usr/bin/env python3

import sys
import wave
import glob
import struct
import numpy as np
import sklearn
import os

import librosa
from scipy import signal
from scipy.signal import butter,lfilter
from librosa.feature import mfcc
import matplotlib.pyplot as plt

SLICE_LEFT=3250
SLICE_RIGHT=9000

def butter_highpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='high', analog=False)
  return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
  b, a = butter_highpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y

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
    self.fw = wave.open(wave_fn,"rb")
    self.n_channels = self.fw.getnchannels()
    self.n_sampw = self.fw.getsampwidth()
    self.n_fr = self.fw.getframerate()
    self.n_fn = self.fw.getnframes()
    print("WaveHelper initialized: %s" % wave_fn)
    print("- Channels: %d" % self.n_channels)
    print("- Sample Width: %d" % self.n_sampw)
    self.sampleBuffer = []
    for framecount in range(0,self.n_fn):
      dx = self.fw.readframes(1)
      if self.n_sampw == 2:
        dx = dx[0:2]
        self.sampleBuffer.append(struct.unpack("<h",dx)[0])  
      else:
        print("Fatal: sample width != 2")
        sys.exit(0)
    # ghetto normalize
    self.sampleBuffer = self.sampleBuffer / np.max(np.abs(self.sampleBuffer))*1000
    plt.plot(self.sampleBuffer,color="blue")
    self.filt = butter_highpass_filter(self.sampleBuffer,4000,44100,order=1)
    plt.plot(self.filt,color="red")
    plt.show()
    self.filt_abs = abs(self.filt)
    self.fw.close()
    self.peakSlices = []
    self.mfccSlices = []
    self.findPeaks_welch()

  def getLabel(self):
    wfn_base = os.path.basename(self.wave_fn)
    if "-" in wfn_base:
      return wfn_base.split("-")[0]
    else:
      return wfn_base.split(".")[0]

  def findPeaks_welch(self):
    psd = signal.welch(self.sampleBuffer,fs=self.n_fr,window="blackman",nfft=256)
    plt.plot(psd)
    plt.show()

  def findPeaks(self):
    peaks,peak_h = signal.find_peaks(self.filt_abs,200,distance=10000)
    self.peakSlices = []
    plt.plot(self.filt)
    for peak in peaks:
      if peak-SLICE_LEFT < 0 or peak + SLICE_RIGHT > len(self.filt):
        pass
      else:
        plt.axvline(x=peak-SLICE_LEFT)
        plt.axvline(x=peak+SLICE_RIGHT)
        self.peakSlices.append(self.filt[peak-SLICE_LEFT:peak+SLICE_RIGHT])
    print(len(self.peakSlices))
    plt.show()
    return peaks

  def extractMFCC(self):
    if len(self.peakSlices) == 0:
      self.findPeaks()
    self.mfccSlices = []
    for i in range(0,len(self.peakSlices)):
      spec = mfcc(y=self.peakSlices[i],sr=44100,n_mfcc=16,n_fft=220,hop_length=110)
      self.mfccSlices.append(list(spec.flatten()))
    return self.mfccSlices
      # plt.plot(spec.flatten(),color="blue")

  def __del__(self):
    print("WaveHelper destroyed: %s" % self.wave_fn)

labelMap = {}

if __name__ == "__main__":
  import random
  random.seed()
  if len(sys.argv) < 2:
    print("needs wave as argv1")
    sys.exit(0)
  featureArray = []
  labelArray = []
  for i in range(1,len(sys.argv)):
    wh = WaveHelper(sys.argv[i])
    f = wh.extractMFCC()
    l = wh.getLabel()
    labelMap[i] = l
    featureArray += f
    labelArray += [i] * len(f)
  clf = sklearn.svm.SVC(gamma=0.001,C=100.)
  for f in featureArray:
    print(len(f))
  print(labelArray)
  feature_train,feature_test,label_train,label_test = sklearn.model_selection.train_test_split(featureArray,labelArray,test_size=0.2)
  clf.fit(feature_train,label_train)
  print("PREDICTION:")
  print(clf.predict(feature_test))
  print("LABEL_TEST:")
  print(label_test)
