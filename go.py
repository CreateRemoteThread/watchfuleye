#!/usr/bin/env python3

import getopt
import sys
import support.wave
import glob
import sklearn

def usage():
  print("-t / --train: train and test a single folder")
  print("-f / --format: specify a feature extraction helper")

CFG_MODEL = None
CFG_FOLDER = None

i = 0
labelMap = {}
labelArray = []
featureArray = []

if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
    sys.exit(0)
  opts,args = getopt.getopt(sys.argv[1:],"m:f:",["model=","folder="])
  for arg,val in opts:
    if arg in ("-m","--model"):
      CFG_MODEL = val
    elif arg in ("-f","--folder"):
      CFG_FOLDER = val
  # todo: implement mode selector
  if CFG_MODEL is None:
    print("go.py: missing -m/--model argument. must be one of wave,em") 
    sys.exit(0)
  if CFG_FOLDER is None:
    print("go.py: missing -f/--format argument.")
    sys.exit(0)
  try:
    __wave_model = eval("support.%s.WaveHelper" % CFG_MODEL)
  except:
    print("go.py: could not import WaveHelper from support.%s" % CFG_MODEL)
    sys.exit(0)
  for fn in glob.glob("%s/*" % CFG_FOLDER):
    wh = __wave_model(fn)
    f = wh.extractFeatures()    # returns: an array of features
    l = wh.getLabel()           # returns: a single label per file
    labelMap[i] = l
    featureArray += f
    labelArray += [i] * len(f)
    i += 1
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
