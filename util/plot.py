#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter,lfilter

def butter_lowpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y

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

if len(sys.argv) == 2:
  buf = np.load(sys.argv[1])
  # plt.plot(buf_filt)
  buf = buf / np.max(np.abs(buf)) * 1000
  buf_filt = butter_bandpass_filter(buf,8000000,16000000,124999999,order=1)
  # buf_filt = butter_lowpass_filter(buf,16000000,124999999,order=1)
  plt.plot(abs(buf_filt))
  # plt.specgram(buf,NFFT=1024,Fs=124999999,noverlap=900)
  plt.show()
elif len(sys.argv) == 3:
  buf1 = np.load(sys.argv[1])
  buf2 = np.load(sys.argv[2])
  buf1_filt = butter_bandpass_filter(buf1,8000000,16000000,124999999,order=1)
  buf2_filt = butter_bandpass_filter(buf2,8000000,16000000,124999999,order=1)
  # buf1_filt = butter_lowpass_filter(buf1,16000000,124999999,order=1)
  # buf2_filt = butter_lowpass_filter(buf2,16000000,124999999,order=1)
  fig, (ax1, ax2) = plt.subplots(nrows=2)
  ax1.plot(buf1)
  ax2.plot(buf2)
  plt.show()
