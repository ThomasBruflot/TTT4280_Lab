from time import daylight
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels)).astype("float64")
    return sample_period, data



# Import data from bin file
sample_period, data = raspi_import('/Users/thomasbruflot/Documents/Sensorer-og-instrumentering/PlaneWaveBrownNoise.bin')

#data = data - np.mean(data)

#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data[:,-3:]-np.mean(data[:,-3:]), axis=0)  # takes FFT of all channels

plt.clf()
# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
plt.subplot(2, 3, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, data[:,-3:]-np.mean(data[:,-3:]))

plt.subplot(2, 3, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum[:,-3:]))) # get the power spectrum
#print(len(data[:,-1]))

lags = np.arange(0,2*len([1,2,3,4,5,6])-1)

#corr1 = np.correlate(data[:,-1] - np.mean(data[:,-1]), data[:,-2] - np.mean(data[:,-2]), "full")
corr1 = np.correlate(data[10000:10250,-1] - np.mean(data[10000:10250,-1]), data[10000:10250,-2] - np.mean(data[10000:10250,-2]), "full")
corr2 = np.correlate(data[10000:10250,-1] - np.mean(data[10000:10250,-1]), data[10000:10250,-3] - np.mean(data[10000:10250,-3]), "full")
corr3 = np.correlate(data[10000:10250,-2] - np.mean(data[10000:10250,-2]), data[10000:10250,-3] - np.mean(data[10000:10250,-3]), "full")
#corr1 = np.correlate([0,0,1,2,3,0], [0,0,0,1,2,3], "full")
#corr4 = np.correlate(data[10000:10250,-1] - np.mean(data[10000:10250,-1]), data[10000:10250,-1] - np.mean(data[10000:10250,-1]), "full")


#lags1 = corr1.argmax() - (len(data[:,-1]) - 1)

def angleCalc(corr1,corr2,corr3):
    n_21 = corr1.argmax() - 250
    n_31 = corr2.argmax() - 250
    n_32 = corr3.argmax() - 250
    print(n_21)
    print(n_31)
    print(n_32)

    theta = np.arctan2(np.sqrt(3) * (n_21 + n_31) , (n_21 - n_31 - 2*n_32))
    if(theta < 0):
        theta = np.degrees(theta)
        theta = theta + 360
    else:
        theta = np.degrees(theta)
    return theta

print(angleCalc(corr1,corr2,corr3))

plt.subplot(2, 3, 3)
plt.title("Crosscorrelation of signals")
plt.xlabel("Lags [l]")
plt.ylabel("xCorr(1,2)")
plt.plot(np.abs(corr1)) # get the power spectrum


plt.subplot(2, 3, 4)
plt.title("Crosscorrelation of signals")
plt.xlabel("Lags [l]")
plt.ylabel("xCorr(1,3)")
plt.plot(np.abs(corr2)) # get the power spectrum

plt.subplot(2, 3, 5)
plt.title("Crosscorrelation of signals")
plt.xlabel("Lags [l]")
plt.ylabel("xCorr(2,3)")
plt.plot(np.abs(corr3)) # get the power spectrum

plt.subplot(2, 3, 6)
plt.title("Crosscorrelation of signals")
plt.xlabel("Lags [l]")
plt.ylabel("xCorr")
plt.plot(np.abs(corr1), label="xCorr(1,2)") # get the power spectrum
plt.plot(np.abs(corr2), label="xCorr(1,3)") # get the power spectrum
plt.plot(np.abs(corr3), label="xCorr(2,3)") # get the power spectrum
plt.legend()



plt.show()
