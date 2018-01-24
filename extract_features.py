__author__ = 'shua'

import argparse
import numpy as np
import wave
import os
from os import listdir
from os.path import isfile, join
import math
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter, hamming
from copy import deepcopy
from scipy.fftpack import fft, ifft
from scikits.talkbox.linpred import lpc
import shutil
from helpers.utilities import *

epsilon = 0.0000000001
prefac = .97


def build_data(wav,begin=None,end=None):
    wav_in_file = wave.Wave_read(wav)
    wav_in_num_samples = wav_in_file.getnframes()
    N = wav_in_file.getnframes()
    dstr = wav_in_file.readframes(N)
    data = np.fromstring(dstr, np.int16)
    if begin is not None and end is not None:
        #return data[begin*16000:end*16000] #numpy 1.11.0
        return data[np.int(begin*16000):np.int(end*16000)] #numpy 1.14.0
    X = []
    l = len(data)
    for i in range(0, l-100, 160):
        X.append(data[i:i + 480])
    return X


def periodogram(x, nfft=None, fs=1):
    """Compute the periodogram of the given signal, with the given fft size.

    Parameters
    ----------
    x : array-like
        input signal
    nfft : int
        size of the fft to compute the periodogram. If None (default), the
        length of the signal is used. if nfft > n, the signal is 0 padded.
    fs : float
        Sampling rate. By default, is 1 (normalized frequency. e.g. 0.5 is the
        Nyquist limit).

    Returns
    -------
    pxx : array-like
        The psd estimate.
    fgrid : array-like
        Frequency grid over which the periodogram was estimated.

    Examples
    --------
    Generate a signal with two sinusoids, and compute its periodogram:

    >>> fs = 1000
    >>> x = np.sin(2 * np.pi  * 0.1 * fs * np.linspace(0, 0.5, 0.5*fs))
    >>> x += np.sin(2 * np.pi  * 0.2 * fs * np.linspace(0, 0.5, 0.5*fs))
    >>> px, fx = periodogram(x, 512, fs)

    Notes
    -----
    Only real signals supported for now.

    Returns the one-sided version of the periodogram.

    Discrepency with matlab: matlab compute the psd in unit of power / radian /
    sample, and we compute the psd in unit of power / sample: to get the same
    result as matlab, just multiply the result from talkbox by 2pi"""
    x = np.atleast_1d(x)
    n = x.size

    if x.ndim > 1:
        raise ValueError("Only rank 1 input supported for now.")
    if not np.isrealobj(x):
        raise ValueError("Only real input supported for now.")
    if not nfft:
        nfft = n
    if nfft < n:
        raise ValueError("nfft < signal size not supported yet")

    pxx = np.abs(fft(x, nfft)) ** 2
    if nfft % 2 == 0:
        pn = nfft / 2 + 1
    else:
        pn = (nfft + 1 )/ 2

    fgrid = np.linspace(0, fs * 0.5, pn)
    return pxx[:pn] / (n * fs), fgrid


def arspec(x, order, nfft=None, fs=1):
    """Compute the spectral density using an AR model.

    An AR model of the signal is estimated through the Yule-Walker equations;
    the estimated AR coefficient are then used to compute the spectrum, which
    can be computed explicitely for AR models.

    Parameters
    ----------
    x : array-like
        input signal
    order : int
        Order of the LPC computation.
    nfft : int
        size of the fft to compute the periodogram. If None (default), the
        length of the signal is used. if nfft > n, the signal is 0 padded.
    fs : float
        Sampling rate. By default, is 1 (normalized frequency. e.g. 0.5 is the
        Nyquist limit).

    Returns
    -------
    pxx : array-like
        The psd estimate.
    fgrid : array-like
        Frequency grid over which the periodogram was estimated.
    """

    x = np.atleast_1d(x)
    n = x.size

    if x.ndim > 1:
        raise ValueError("Only rank 1 input supported for now.")
    if not np.isrealobj(x):
        raise ValueError("Only real input supported for now.")
    if not nfft:
        nfft = n
    a, e, k = lpc(x, order)

    # This is not enough to deal correctly with even/odd size
    if nfft % 2 == 0:
        pn = nfft / 2 + 1
    else:
        pn = (nfft + 1 )/ 2

    px = 1 / np.fft.fft(a, nfft)[:pn]
    pxx = np.real(np.conj(px) * px)
    pxx /= fs / e
    fx = np.linspace(0, fs * 0.5, pxx.size)
    return pxx, fx


def taper(n, p=0.1):
    """Return a split cosine bell taper (or window)

    Parameters
    ----------
        n: int
            number of samples of the taper
        p: float
            proportion of taper (0 <= p <= 1.)

    Note
    ----
    p represents the proportion of tapered (or "smoothed") data compared to a
    boxcar.
    """
    if p > 1. or p < 0:
        raise ValueError("taper proportion should be betwen 0 and 1 (was %f)"
                         % p)
    w = np.ones(n)
    ntp = np.floor(0.5 * n * p)
    w[:ntp] = 0.5 * (1 - np.cos(np.pi * 2 * np.linspace(0, 0.5, ntp)))
    w[-ntp:] = 0.5 * (1 - np.cos(np.pi * 2 * np.linspace(0.5, 0, ntp)))

    return w


def atal(x, order, num_coefs):
    x = np.atleast_1d(x)
    n = x.size
    if x.ndim > 1:
        raise ValueError("Only rank 1 input supported for now.")
    if not np.isrealobj(x):
        raise ValueError("Only real input supported for now.")
    a, e, kk = lpc(x, order)
    c = np.zeros(num_coefs)
    c[0] = a[0]
    for m in range(1, order+1):
        c[m] = - a[m]
        for k in range(1, m):
            c[m] += (float(k)/float(m)-1)*a[k]*c[m-k]
    for m in range(order+1, num_coefs):
        for k in range(1, order+1):
            c[m] += (float(k)/float(m)-1)*a[k]*c[m-k]
    return c


def preemp(input, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, input)


def arspecs(input_wav,order,Atal=False):
    epsilon = 0.0000000001
    data = input_wav
    if Atal:
        ar = atal(data, order, 30)
        return ar
    else:
        ar = []
        ars = arspec(data, order, 4096)
        for k, l in zip(ars[0], ars[1]):
            ar.append(math.log(math.sqrt((k**2)+(l**2))))
        for val in range(0,len(ar)):
            if ar[val] == 0.0:
                ar[val] = deepcopy(epsilon)
        mspec1 = np.log10(ar)
        # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
        ar = dct(mspec1, type=2, norm='ortho', axis=-1)
        return ar[:30]


def specPS(input_wav,pitch):
        N = len(input_wav)
        samps = N/pitch
        if samps == 0:
            samps = 1
        frames = N/samps
        data = input_wav[0:frames]
        specs = periodogram(data,nfft=4096)
        for i in range(1,int(samps)):
            data = input_wav[frames*i:frames*(i+1)]
            peri = periodogram(data,nfft=4096)
            for sp in range(len(peri[0])):
                specs[0][sp] += peri[0][sp]
        for s in range(len(specs[0])):
            specs[0][s] /= float(samps)
        peri = []
        for k, l in zip(specs[0], specs[1]):
            if k == 0 and l == 0:
                peri.append(epsilon)
            else:
                peri.append(math.log(math.sqrt((k ** 2) + (l ** 2))))
        # Filter the spectrum through the triangle filterbank
        mspec = np.log10(peri)
        # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
        ceps = dct(mspec, type=2, norm='ortho', axis=-1)
        return ceps[:50]


def build_single_feature_row(data, Atal):
    lpcs = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    arr = []
    periodo = specPS(data, 50)
    arr.extend(periodo)
    for j in lpcs:
        if Atal:
            ars = arspecs(data, j, Atal=True)
        else:
            ars = arspecs(data, j)
        arr.extend(ars)
    for i in range(len(arr)):
        if np.isnan(np.float(arr[i])):
            arr[i] = 0.0
    return arr


def create_features(input_wav_filename, feature_filename, begin=None, end=None, Atal=False):
    tmp_wav16_filename = generate_tmp_filename("wav")
    easy_call("sox " + input_wav_filename + " -c 1 -r 16000 " + tmp_wav16_filename)
    X = build_data(tmp_wav16_filename, begin, end)
    if begin is not None and end is not None:
        arr = [input_wav_filename]
        arr.extend(build_single_feature_row(X, Atal))
        np.savetxt(feature_filename, np.asarray([arr]), delimiter=",", fmt="%s")
        return arr
    arcep_mat = []
    for i in range(len(X)):
        arr = [input_wav_filename + str(i)]
        arr.extend(build_single_feature_row(X[i], Atal))
        arcep_mat.append(arr)
    np.savetxt(feature_filename, np.asarray(arcep_mat), delimiter=",", fmt="%s")
    return arcep_mat


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Extract features for formants estimation.')
    parser.add_argument('wav_file', default='', help="WAV audio filename (single vowel or an whole utternace)")
    parser.add_argument('feature_file', default='', help="output feature text file")
    parser.add_argument('--begin', help="beginning time in the WAV file", default=0.0, type=float)
    parser.add_argument('--end', help="end time in the WAV file", default=-1.0, type=float)
    args = parser.parse_args()

    if args.begin > 0.0 or args.end > 0.0:
        create_features(args.wav_file, args.feature_file, args.begin, args.end)
    else:
        create_features(args.wav_file, args.feature_file)


