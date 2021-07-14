#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import support.wave

wh = support.wave.WaveHelper(sys.argv[1])
plt.plot(wh.sampleBuffer)
plt.show()
