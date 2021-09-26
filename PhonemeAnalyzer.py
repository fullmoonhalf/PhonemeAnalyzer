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
    parser.add_argument('--plot-formant', action="store_true")
    parser.add_argument('--plot-spectrum', action="store_true")
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=sys.float_info.max)
    parser.add_argument('--detect-peak-min-freq', type=float, default=100)
    parser.add_argument('--detect-peak-max-freq', type=float, default=6000)
    args = parser.parse_args()
    return args

def main(args):
    source = AudioSegment.from_file(args.input)
    data = np.array(source.get_array_of_samples())

    window = 4096
    amplist = []
    timelist = []
    peaklist = []
    freqs = fftpack.fftfreq(window, d=1.0/source.frame_rate)
    dimension = int(freqs.shape[0] / 8)
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

        # FFT
        x = data[start:end:source.channels]
        x = hann_window * x
        spectrum = fftpack.fft(x)

        # Cepstrum
        spectrum_db = np.log10(spectrum)
        cepstrum_db = np.real(fftpack.ifft(spectrum_db))
        cepstrum_index = 50
        cepstrum_db[cepstrum_index:len(cepstrum_db)-cepstrum_index] = 0
        cepstrum_db_low = fftpack.fft(cepstrum_db)
        peak_indecies = signal.argrelmax(cepstrum_db_low, order=5)
        peak_indecies = [x for x in peak_indecies[0] if freqs[x] > args.detect_peak_min_freq and freqs[x] < args.detect_peak_max_freq]

        # 
        spectrum_abs = np.abs(spectrum)
        amp = spectrum_abs / window * 2
        amp = acf * amp
        maxamp = max(amp)
        amp_splited = amp[:dimension]
        timelist.append(time)
        amplist.append(amp_splited)
        peak_freqs = (time, [(freqs[x], amp[x]) for x in peak_indecies[:int(len(peak_indecies)/2)]])
        peaklist.append(peak_freqs)
        for x in peak_indecies:
            ptime.append(time)
            pfreq.append(freqs[x])
            pamp.append(np.log10(amp[x]))

    if args.plot_formant:
        forumant_1st = []
        forumant_2nd = []
        for time, frame in peaklist:
            if len(frame) >= 2:
                forumant_1st.append(frame[0][0])
                forumant_2nd.append(frame[1][0])
        plt.scatter(forumant_1st, forumant_2nd, s=1, c="black")
        plt.show()
    

    if args.plot_spectrum:
        freqlist = np.array([np.array(freqs[:dimension])]*len(amplist))
        timelist = np.array([np.array([t] * dimension) for t in timelist])
        pltamplist = np.log10(np.array(amplist))
        fig, ax = plt.subplots()
        heatmap = ax.pcolormesh(timelist, freqlist, pltamplist, cmap='coolwarm')
        fig.colorbar(heatmap, ax=ax)
        scatter = ax.scatter(ptime, pfreq, s=1, c="black")
        plt.show()

    return 0


if __name__ == '__main__':
    main(parse_args())
