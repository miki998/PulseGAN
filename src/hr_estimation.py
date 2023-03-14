import numpy as np

# signal libs
from scipy import signal, sparse
from scipy.signal import butter, lfilter, filtfilt, freqz,welch

# Different ways to compute heart rate from signals
def find_heart_rate(fft, freqs, freq_min, freq_max):
	"""
	desc: compute HR from FFT peaks 

	args: 
		- fft::[]
		- freqs::[]
		- freq_min::[]
		- freq_max::[]

	ret:
		- HR::[float]
			heart-rate 
	"""

	fft_maximums = []

	for i in range(fft.shape[0]):
		if freq_min <= freqs[i] <= freq_max:
			fftMap = abs(fft[i])
			fft_maximums.append(fftMap.max())
		else:
			fft_maximums.append(0)

	peaks, properties = signal.find_peaks(fft_maximums)
	max_peak = -1
	max_freq = 0

	# Find frequency with max amplitude in peaks
	for peak in peaks:
		if fft_maximums[peak] > max_freq:
			max_freq = fft_maximums[peak]
			max_peak = peak

	HR = freqs[max_peak] * 60
	return HR


# https://github.com/nasir6/rPPG/blob/master/pulse.py
def get_rfft_hr(signal,framerate,minFreq=.75,maxFreq=3.7):
    """
    desc:
    
    args:
    
    ret:
    
    """
    signal_size = len(signal)
    signal = signal.flatten()
    fft_data = np.fft.rfft(signal) # FFT
    fft_data = np.abs(fft_data)

    freq = np.fft.rfftfreq(signal_size, 1./framerate) # Frequency data
    inds = np.where((freq < minFreq) | (freq > maxFreq) )[0]
    fft_data[inds] = 0
    bps_freq = 60.0*freq
    max_index = np.argmax(fft_data)
    HR =  bps_freq[max_index]
    return HR

# https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
def get_hr_welch(signal, fps, minFreq=.75, maxFreq=3.7):
    """
    desc:
    
    args:
    
    ret:
    
    """
    signal = signal.flatten()
    green_f, green_psd = welch(signal, fps, 'flattop', nperseg=len(signal)) #, scaling='spectrum',nfft=2048)

    #green_psd = green_psd.flatten()
    first = np.where(green_f > minFreq)[0] #0.8 for 300 frames
    last = np.where(green_f < maxFreq)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max*60.0
    return hr

