#!/usr/bin/env python3

import os
import sys
import wave
import numpy as np
import librosa_nollvm
import struct
from scipy import signal
from scipy.signal import butter,lfilter
from librosa_nollvm.feature import mfcc

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
    # print("- Channels: %d" % self.n_channels)
    # print("- Sample Width: %d" % self.n_sampw)
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
    # plt.specgram(self.sampleBuffer,NFFT=1024,Fs=44100,noverlap=900)
    # plt.plot(self.sampleBuffer,color="blue")
    self.filt = butter_bandpass_filter(self.sampleBuffer,2500,15000,44100,order=1)
    # plt.plot(self.filt,color="red")
    # plt.show()
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
    plt.show()

  def findPeaks(self,plot_helper=None):
    peaks,peak_h = signal.find_peaks(self.filt_abs,100,distance=10000)
    self.peakSlices = []
    # plt.plot(self.filt)
    for peak in peaks:
      if peak-SLICE_LEFT < 0 or peak + SLICE_RIGHT > len(self.filt):
        pass
      else:
        if plot_helper is not None:
          # plot_helper.plot([peak-SLICE_LEFT],[self.sampleBuffer[peak-SLICE_LEFT]],marker='x')
          # plot_helper.plot([peak+SLICE_RIGHT],[self.sampleBuffer[peak-SLICE_RIGHT]],marker='x')
          plot_helper.axvline(x=peak-SLICE_LEFT)
          plot_helper.axvline(x=peak+SLICE_RIGHT)
        normalized_slice = self.filt[peak-SLICE_LEFT:peak+SLICE_RIGHT]
        normalized_slice = normalized_slice / np.max(np.abs(normalized_slice))*1000
        self.peakSlices.append(normalized_slice)
        # self.peakSlices.append(self.filt[peak-SLICE_LEFT:peak+SLICE_RIGHT])
    # print(len(self.peakSlices))
    # plt.show()
    return peaks

  def extractFeatures(self):
    if len(self.peakSlices) == 0:
      self.findPeaks()
    self.mfccSlices = []
    for i in range(0,len(self.peakSlices)):
      spec = mfcc(y=self.peakSlices[i],sr=44100,n_mfcc=16,n_fft=220,hop_length=110)
      # plt.plot(spec.flatten(),color="blue")
      self.mfccSlices.append(list(spec.flatten()))
    # plt.show()
    return self.mfccSlices

  def __del__(self):
    print("WaveHelper destroyed: %s" % self.wave_fn)

if __name__ == "__main__":
  print("wave.py: this is not meant to be called directly")
  sys.exit(0)
