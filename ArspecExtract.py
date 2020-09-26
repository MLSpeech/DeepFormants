from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import wave
import os
import math
from scipy.fftpack.realtransforms import dct
from copy import deepcopy
from scipy.fftpack import fft, ifft
from scikits.talkbox.linpred import lpc
np.random.seed(1337)
epsilon = 0.0000000001


def build_data(wav, begin=None, end=None):
    wav_in_file = wave.Wave_read(wav)
    wav_in_num_samples = wav_in_file.getnframes()
    N = wav_in_file.getnframes()
    dstr = wav_in_file.readframes(N)
    data = np.fromstring(dstr, np.int16)
    return data


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
        pn = (nfft + 1) / 2

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
        pn = (nfft + 1) / 2

    px = 1 / np.fft.fft(a, nfft)[:pn]
    pxx = np.real(np.conj(px) * px)
    pxx /= fs / e
    fx = np.linspace(0, fs * 0.5, pxx.size)
    return pxx, fx


def arspecs(input_wav, order, Atal=False):
    epsilon = 0.0000000001
    data = input_wav
    ar = []
    ars = arspec(data, order, 4096)
    for k, l in zip(ars[0], ars[1]):
        ar.append(math.log(math.sqrt((k ** 2) + (l ** 2))))
    for val in range(0, len(ar)):
        if ar[val] == 0.0:
            ar[val] = deepcopy(epsilon)
    mspec1 = np.log10(ar)
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ar = dct(mspec1, type=2, norm='ortho', axis=-1)
    return ar[:30]


def specPS(input_wav, pitch):
    N = len(input_wav)
    samps = N / pitch
    if samps == 0:
        samps = 1
    frames = N / samps
    data = input_wav[0:frames]
    specs = periodogram(data, nfft=4096)
    for i in range(1, int(samps)):
        data = input_wav[frames * i:frames * (i + 1)]
        peri = periodogram(data, nfft=4096)
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


def build_single_feature_row(data):
    lpcs = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    arr = []
    periodo = specPS(data, 50)
    arr.extend(periodo)
    for j in lpcs:
        ars = arspecs(data, j)
        arr.extend(ars)
    for i in range(len(arr)):
        if np.isnan(np.float(arr[i])):
            arr[i] = 0.0
    return arr


def get_y():
    data = np.load('timit.npy')
    timit = []
    for row in data:
        y = open('Y/' + str(row[0]).replace("timit", "VTRFormants") + ".y").readline().split()
        arr = []
        arr.append(float(y[0]))
        arr.append(float(y[1]))
        arr.append(float(y[2]))
        arr.append(float(y[3]))
        arr.extend(row)
        timit.append(arr)
    nump = np.asarray(timit)
    np.save('timit_train_arspec',nump)
    return


def build_timit_data():
    arcep_mat = []
    path = 'X_test/'
    for file in [f for f in os.listdir(path) if f.endswith('.wav')]:
        name = file.replace('.wav', '')
        y = open('Y_test' + '/' + str(name).replace("timit", "VTRFormants") + ".y").readline().split()
        X = build_data(path + file)
        arr = [name]
        arr.append(float(y[0]))
        arr.append(float(y[1]))
        arr.append(float(y[2]))
        arr.append(float(y[3]))
        arr.extend(build_single_feature_row(X))
        arcep_mat.append(arr)
    nump = np.asarray(arcep_mat)
    np.save('timitTest',nump)

    arcep_mat = []
    path = 'X/'
    for file in [f for f in os.listdir(path) if f.endswith('.wav')]:
        name = file.replace('.wav', '')
        y = open('Y/' + str(name).replace("timit", "VTRFormants") + ".y").readline().split()
        X = build_data(path + file)
        arr = [name]
        arr.append(float(y[0]))
        arr.append(float(y[1]))
        arr.append(float(y[2]))
        arr.append(float(y[3]))
        arr.extend(build_single_feature_row(X))
        arcep_mat.append(arr)
    nump = np.asarray(arcep_mat)
    np.save('timitTrain',nump)
    return

build_timit_data()