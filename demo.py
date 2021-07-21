#!/usr/bin/env python3

import time
import random
import getopt
import sys
import support
import glob
import sklearn
import sklearn.linear_model
import sklearn.neighbors
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy.testing
import tkinter as tk

mpl.use("TkAgg")

class Application(tk.Frame):
  def __init__(self,master=None):
    super().__init__(master)
    self.action_started = False
    self.master = master
    self.f = Figure(figsize=(8,6),dpi=100)
    self.mainPlot = self.f.add_subplot(111)
    self.canvas = FigureCanvasTkAgg(self.f,self.master)
    self.master.after(1000,self.start)
    # self.cid = self.canvas.mpl_connect("button_press_event",self.canvasClick)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)

  def start(self):
    if self.action_started is True:
      return
    else:
      self.action_started = True
      doProcess(self)

  def canvasClick(self,event):
    self.canvas.mpl_disconnect(self.cid)
    doProcess(self)    

def usage():
  print("-t / --train: train and test a single folder")
  print("-f / --format: specify a feature extraction helper")

CFG_MODEL = None
CFG_FOLDER = None
CFG_HOLDOUT = None

i = 0
labelMap = {}
labelArray = []
featureArray = []

def doProcess(app_master):
  global i,labelMap,labelArray,featureArray,CFG_MODEL,CFG_FOLDER,CFG_HOLDOUT
  if len(sys.argv) < 2:
    usage()
    sys.exit(0)
  opts,args = getopt.getopt(sys.argv[1:],"m:f:h:",["model=","folder=","holdout="])
  for arg,val in opts:
    if arg in ("-m","--model"):
      CFG_MODEL = val
    elif arg in ("-f","--folder"):
      CFG_FOLDER = val
    elif arg in ("-h","--holdout"):
      CFG_HOLDOUT = val
  if CFG_MODEL is None:
    print("go.py: missing -m/--model argument. must be one of wave,em") 
    sys.exit(0)
  if CFG_FOLDER is None:
    print("go.py: missing -f/--format argument.")
    sys.exit(0)
  try:
    print("Using signal processing model '%s'" % CFG_MODEL)
    exec("import support.%s" % CFG_MODEL)
    __wave_model = eval("support.%s.WaveHelper" % CFG_MODEL)
  except Exception as e:
    print("go.py: could not import WaveHelper from support.%s" % CFG_MODEL)
    print(e)
    sys.exit(0)
  print("Using training data from '%s'" % CFG_FOLDER)
  sw = numpy.testing.suppress_warnings()
  for fn in glob.glob("%s/*" % CFG_FOLDER):
    wh = __wave_model(fn)
    time.sleep(0.5)
    app_master.mainPlot.clear()
    app_master.mainPlot.set_title("Training Data")
    app_master.mainPlot.plot(wh.sampleBuffer)
    # time.sleep(1.0)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      p = wh.findPeaks(plot_helper = app_master.mainPlot)
      f = wh.extractFeatures()    # returns: an array of features
    l = wh.getLabel()           # returns: a single label per file
    if l in labelMap.keys():
      l_a = labelMap[l]
    else:
      labelMap[l] = i
      l_a = i
      i += 1
    app_master.canvas.draw()
    app_master.canvas.flush_events()
    featureArray += f
    labelArray += [l_a] * len(f)
  clf = sklearn.linear_model.LogisticRegression()
  if CFG_HOLDOUT is None:
    print("Using train_test_split approach")
    feature_train,feature_test,label_train,label_test = sklearn.model_selection.train_test_split(featureArray,labelArray,test_size=0.2)
  else:
    print("Neural network ready...")
    time.sleep(1.5)
    print("Using holdout data set approach")
    feature_train = featureArray
    label_train = labelArray
    wh = __wave_model(CFG_HOLDOUT)
    app_master.mainPlot.clear()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      f = wh.extractFeatures()    # returns: an array of features
    wh.findPeaks(plot_helper = app_master.mainPlot)
    app_master.mainPlot.set_title("Sample Data")
    app_master.mainPlot.plot(wh.sampleBuffer)
    app_master.canvas.draw()
    app_master.canvas.flush_events()
    feature_test = f
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    clf.fit(feature_train,label_train)
    # print("PREDICTION:")
    pl = list(clf.predict(feature_test))
  unmapped = unmapLabel(pl,labelMap) 
  time.sleep(0.5)
  for i in range(0,len(pl)):
    app_master.mainPlot.text(wh.peakPoints[i]+700,500,str(unmapped[i]),color="red")
  app_master.canvas.draw()
  app_master.canvas.flush_events()
  print(unmapped)
  if CFG_HOLDOUT is None:
    print("LABEL_TEST:")
    print(label_test) 
    print(unmapLabel(label_test,labelMap))

def unmapLabel(labels,labelmap):
  label_unmap = {}
  for l in labelmap.keys():
    label_unmap[labelmap[l]] = l
  x = [label_unmap[i] for i in labels]
  return x

if __name__ == "__main__":
  root = tk.Tk()
  root.title("ML Keystroke Recovery")
  root.geometry("800x600")
  app = Application(master=root)
  app.mainloop()
