#!/usr/bin/env python3

import getopt
import sys
import support
import glob
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.neighbors
import warnings

def unmapProba(probs,labelmap):
  label_unmap = [None] * len(labelmap.keys())
  for l in labelmap.keys():
    li = labelmap[l]
    label_unmap[li] = l
  probaArray = []
  i = 0
  for prob_l in probs:
    this_letter_array = []
    top_args = prob_l.argsort()[::-1][:3]
    print("Extracting character probability for slot %d" % i)
    i += 1
    for ti in top_args:
      print("%s : %f" % (label_unmap[ti],prob_l[ti]))
      this_letter_array.append(label_unmap[ti])
    probaArray.append(this_letter_array)
  return probaArray

def unmapLabel(labels,labelmap):
  label_unmap = {}
  for l in labelmap.keys():
    label_unmap[labelmap[l]] = l
  x = [label_unmap[i] for i in labels]
  return x

def usage():
  print("-t / --train: train and test a single folder")
  print("-f / --format: specify a feature extraction helper")
  print("-h / --holdout: specify a holdout dataset")
  print("-w / --wordlist: specify wordlist for closest match guessing")
  print("-d / --demo: run with extra bells and whistles")

CFG_MODEL = None
CFG_FOLDER = None
CFG_HOLDOUT = None
CFG_WORDLIST = None
CFG_DEMO = False

i = 0
labelMap = {}
labelArray = []
featureArray = []

def closestMatch(proba_top3,wordlist):
  print("Searching closest words from '%s'" % wordlist)
  f = open(wordlist)
  len_proba_top3 = len(proba_top3)
  for wl in f.readlines():
    wl_ = wl.rstrip()
    if len(wl_) != len_proba_top3:
      continue
    skipFlag = False
    for i in range(0,len(wl_)):
      if wl_[i] not in proba_top3[i]:
        skipFlag = True
        break
    if skipFlag:
      continue
    else:
      print(wl_)
  f.close()

if __name__ == "__main__":
  if len(sys.argv) < 2:
    usage()
    sys.exit(0)
  sw = np.testing.suppress_warnings()
  opts,args = getopt.getopt(sys.argv[1:],"m:f:h:w:d",["model=","folder=","holdout=","wordlist=","demo"])
  for arg,val in opts:
    if arg in ("-m","--model"):
      CFG_MODEL = val
    elif arg in ("-h","--holdout"):
      CFG_HOLDOUT = val
    elif arg in ("-f","--folder"):
      CFG_FOLDER = val
    elif arg in ("-w","--wordlist"):
      CFG_WORDLIST = val
    elif arg in ("-d","--demo"):
      import matplotlib.pyplot as plt
      CFG_DEMO = True
  # todo: implement mode selector
  if CFG_MODEL is None:
    print("go.py: missing -m/--model argument. must be one of wave,em") 
    sys.exit(0)
  if CFG_FOLDER is None:
    print("go.py: missing -f/--format argument.")
    sys.exit(0)
  try:
    exec("import support.%s" % CFG_MODEL)
    __wave_model = eval("support.%s.WaveHelper" % CFG_MODEL)
  except Exception as e:
    print("go.py: could not import WaveHelper from support.%s" % CFG_MODEL)
    print(e)
    sys.exit(0)
  print("Constructing ML Classifier...")
  for fn in glob.glob("%s/*" % CFG_FOLDER):
    wh = __wave_model(fn)
    with warnings.catch_warnings():
      if CFG_DEMO:
        warnings.simplefilter("ignore")
      f = wh.extractFeatures()    # returns: an array of features
    l = wh.getLabel()           # returns: a single label per file
    if l in labelMap.keys():
      l_a = labelMap[l]
    else:
      labelMap[l] = i
      l_a = i
      i += 1
    featureArray += f
    labelArray += [l_a] * len(f)
  clf = sklearn.linear_model.LogisticRegression()
  if CFG_HOLDOUT is None:
    print("Using train_test_split approach")
    feature_train,feature_test,label_train,label_test = sklearn.model_selection.train_test_split(featureArray,labelArray,test_size=0.2)
  else:
    print("Using holdout data approach")
    feature_train = featureArray
    label_train = labelArray
    wh = __wave_model(CFG_HOLDOUT)
    f = wh.extractFeatures()
    feature_test = f
  with warnings.catch_warnings():
    if CFG_DEMO:
      warnings.simplefilter("ignore")
    clf.fit(feature_train,label_train)
  print("PREDICTION:")
  l = list(clf.predict(feature_test))
  lf = clf.predict_proba(feature_test)
  unmapped = unmapLabel(l,labelMap)
  print(unmapped)
  prob_array = unmapProba(lf,labelMap)
  if CFG_WORDLIST is not None and CFG_HOLDOUT is not None:
    closestMatch(prob_array,CFG_WORDLIST)
  if CFG_HOLDOUT is None:
    print("LABEL_TEST:")
    unmapped_test = unmapLabel(label_test,labelMap)
    print(unmapped_test)
