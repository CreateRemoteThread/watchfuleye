#!/usr/bin/env python3

from picoscope import ps2000a
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter,lfilter,freqz
import tkinter as tk
import numpy as np
import sys
import support
import getopt
import time

mpl.use("Agg")
import matplotlib.pyplot as plt
CONFIG_SAMPLECOUNT = 400000
CONFIG_SAMPLERATE = 124999999
CONFIG_CAPTURES = -1
CONFIG_DISPLAY_SAMPLES = 50000

CONFIG_THRESHOLD = support.CONFIG_THRESHOLD
CONFIG_BACKOFF = 0.1
CONFIG_VRANGE = 0.2

CONFIG_FPREFIX = None
CONFIG_DONOTSAVE = True
CONFIG_SPECGRAM = False

def butter_lowpass(cutoff,fs,order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b,a,data)
  return y

def confirmSettings():
  print("Number of captures: %d" % CONFIG_CAPTURES)
  print("Sample rate: %d" % CONFIG_SAMPLERATE)
  print("Sample count: %d" % CONFIG_SAMPLECOUNT)
  print("Trigger level: %f" % CONFIG_THRESHOLD)
  print("Analog range: %f" % CONFIG_VRANGE)
  if CONFIG_THRESHOLD > CONFIG_VRANGE:
    print("Trigger cannot exceed analog range. Bye!")
    sys.exit(0)
  if CONFIG_FPREFIX:
    print("Capturing data for: %s" % CONFIG_FPREFIX)
  x = input("Are these settings correct? [y/n] ")
  if x.rstrip() not in ("y","Y"):
    print("Capture action cancelled. Bye!")
    sys.exit(0)

class Application(tk.Frame):
  def __init__(self,master=None):
    super().__init__(master)
    self.master=master
    self.f = Figure(figsize=(8,6),dpi=100)
    self.mainPlot = self.f.add_subplot(111)
    self.canvas=FigureCanvasTkAgg(self.f,self.master)
    self.cid = self.canvas.mpl_connect("button_press_event",self.canvasClick)
    self.canvas.draw()
    self.canvas_tk = self.canvas.get_tk_widget().pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)

  def canvasClick(self,event):
    global CONFIG_VRANGE,CONFIG_SAMPLERATE,CONFIG_SAMPLECOUNT,CONFIG_FPREFIX,CONFIG_THRESHOLD,CONFIG_CAPTURES,CONFIG_BACKOFF,CONFIG_DONOTSAVE,CONFIG_SPECGRAM,CONFIG_DISPLAY_SAMPLES
    try:
      ps = ps2000a.PS2000a()
      self.canvas.mpl_disconnect(self.cid)
      print("Committing capture...")
    except:
      print("Failed to commit scope")
      return
    ps.setChannel('A','DC',VRange=CONFIG_VRANGE,VOffset=0.0,enabled=True,BWLimited=False,probeAttenuation=10.0)
    (freq,maxSamples) = ps.setSamplingFrequency(CONFIG_SAMPLERATE,CONFIG_SAMPLECOUNT)
    print(" > Asked for %d Hz, got %d Hz" % (CONFIG_SAMPLERATE, freq))
    if CONFIG_FPREFIX is not None:
      print(" > Configured in training mode with prefix %s" % CONFIG_FPREFIX)
    ps.setSimpleTrigger('A',CONFIG_THRESHOLD,'Rising',enabled=True)
    i = 0
    nCount = 0
    nMax = CONFIG_CAPTURES
    print("Capture is committed...")
    while True:
      if nMax == nCount:
        break
      ps.runBlock(pretrig=0.2)
      ps.waitReady()
      dataA = ps.getDataV("A",CONFIG_SAMPLECOUNT,returnOverflow=False)
      # print(len(dataA))
      SLICE_START = int(CONFIG_SAMPLECOUNT * 0.2)
      SLICE_END = SLICE_START + 500
      if float(max(dataA[SLICE_START:SLICE_END])) < float(CONFIG_THRESHOLD):
        # print("failed capture")
        continue
      data_DISPLAY = dataA[SLICE_START-CONFIG_DISPLAY_SAMPLES:SLICE_START+CONFIG_DISPLAY_SAMPLES]
      if CONFIG_DONOTSAVE is False:
        if CONFIG_FPREFIX is None:
          print("Saving training capture...")
          np.save("floss/%d.npy" % nCount,data_DISPLAY)
        else:
          print("Saving real capture...")
          np.save("%s-%d.npy" % (CONFIG_FPREFIX,nCount),data_DISPLAY)
      else:
        print("No save mode specified, discarding save")
      self.mainPlot.clear()
      if CONFIG_SPECGRAM:
        self.mainPlot.specgram(data_DISPLAY,NFFT=1024,Fs=CONFIG_SAMPLERATE,noverlap=900)
      else:
        self.mainPlot.plot(support.block_preprocess_function(data_DISPLAY))
      self.canvas.draw()
      self.canvas.flush_events()
      time.sleep(CONFIG_BACKOFF)
      nCount += 1
    print("Captured %d slices" % nCount)

def usage():
  print("./capture-em.py: a keyboard EM manual capture tool (for pico2000a)")
  print(" -t / --train [label]: capture training samples, label them")
  print(" -c / --count [count]: captures [count] number of keystrokes")
  print(" -s / --specgram: changes the display to a spectrogram view")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    args,opts = getopt.getopt(sys.argv[1:],"ht:c:s",["help","train=","count=","specgram"])
    for arg,opt in args:
      if arg in ("-t","--train"):
        CONFIG_FPREFIX = opt
        CONFIG_DONOTSAVE = False
      elif arg in ("-c","--count"):
        CONFIG_CAPTURES = int(opt)
        CONFIG_DONOTSAVE = False
      elif arg in ("-s","--specgram"):
        CONFIG_SPECGRAM = True
      elif arg in ("-h","--help"):
        usage()
        sys.exit(0)
  print("Click the graph to begin capture...")
  root = tk.Tk()
  root.title("capture.py")
  root.geometry("800x600")
  app = Application(master=root)
  app.mainloop()

