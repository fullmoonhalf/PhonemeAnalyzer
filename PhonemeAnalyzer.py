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
    parser.add_argument('--amp-threshold', type=float, default=20)
    parser.add_argument('--detect-peak-min-freq', type=float, default=100)
    parser.add_argument('--detect-peak-max-freq', type=float, default=3000)
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
        spectrum = np.abs(spectrum)
        amp = spectrum / window * 2
        amp = acf * amp

        maxamp = max(amp)
        amp_splited = amp[:dimension]
        timelist.append(time)
        amplist.append(amp_splited)
        peak = signal.argrelmax(amp_splited, order=5)
        peak_indecies = [x for x in peak[0] if (freqs[x] > args.detect_peak_min_freq and freqs[x] < args.detect_peak_max_freq and amp_splited[x] > args.amp_threshold)]
        peak_freqs = [(freqs[x], amp_splited[x]) for x in peak_indecies]
        for x in peak_indecies:
            ptime.append(time)
            pfreq.append(freqs[x])
            pamp.append(math.log10(amp_splited[x]))


    if args.plot:
        freqlist = np.array([np.array(freqs[:dimension])]*len(amplist))
        timelist = np.array([np.array([t] * dimension) for t in timelist])
        amplist = np.array(amplist)
        amplist = np.array(amplist)
        fig, ax = plt.subplots()
        heatmap = ax.pcolormesh(timelist, freqlist, np.log10(amplist), cmap='coolwarm')
        fig.colorbar(heatmap, ax=ax)
        scatter = ax.scatter(ptime, pfreq, s=1, c="black")
        plt.show()
    return 0


if __name__ == '__main__':
    main(parse_args())
