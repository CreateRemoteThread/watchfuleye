#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import support.wave

wh = support.wave.WaveHelper(sys.argv[1])
peaks = wh.findPeaks()

plt.title("Keystroke Peak Detection")
plt.plot(wh.sampleBuffer)
plt.plot(peaks,wh.sampleBuffer[peaks],marker='x')
plt.show()
