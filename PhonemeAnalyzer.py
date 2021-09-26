import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy import fftpack
from scipy import signal
from pydub import AudioSegment


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=sys.float_info.max)
    args = parser.parse_args()
    return args

def main(args):
    source = AudioSegment.from_file(args.input)
    data = np.array(source.get_array_of_samples())

    window = 4096
    amplist = []
    timelist = []
    freqs = fftpack.fftfreq(window, d=1.0/source.frame_rate)
    dimension = int(freqs.shape[0] / 16)
    ptime = []
    pfreq = []
    pamp = []
    hann_window = signal.hann(window)
    acf = 1/(sum(hann_window)/window)

    for start in range(0, len(data)-window*source.channels, int(window*source.channels/8)):
        time = start / (source.channels * source.frame_rate)
        if time < args.start or time > args.end:
            continue
        end = start + window * source.channels
        x = data[start:end:source.channels]
        x = hann_window * x
        spectrum = fftpack.fft(x)
        spectrum = spectrum[:dimension]
        spabs = np.abs(spectrum)
        timelist.append(time)
        amplist.append(spabs)
        peak = signal.argrelmax(spabs, order=5)
        peak_freqs = [x for x in peak[0] if (freqs[x] > 100 and freqs[x] < 3000)]
        for x in peak_freqs:
            ptime.append(time)
            pfreq.append(freqs[x])
            pamp.append(math.log10(spabs[x]))

    freqlist = np.array([np.array(freqs[:dimension])]*len(amplist))
    timelist = np.array([np.array([t] * dimension) for t in timelist])
    amplist = np.array(amplist)

    if args.plot:
        amplist = np.array(amplist)
        fig, ax = plt.subplots()
        ax.pcolormesh(timelist, freqlist, np.log10(amplist), cmap='coolwarm')
        scatter = ax.scatter(ptime, pfreq, s=1, c="black")
        plt.show()
    return 0


if __name__ == '__main__':
    main(parse_args())
