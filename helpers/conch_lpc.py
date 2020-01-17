# This file has been copied (with minor changes) from Michael
# McAuliffe's Conch project, to provide a compatible replacement
# implementation of the lpc() function from the obsolete Python-2-only
# scikits.talkbox library.
#
# Conch repository: https://github.com/mmcauliffe/Conch-sounds
# Source: https://github.com/mmcauliffe/Conch-sounds/blob/master/conch/analysis/formants/lpc.py

# Copyright (c) 2015 Michael McAuliffe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#import librosa
import numpy as np
import scipy as sp
from scipy.signal import lfilter

from scipy.fftpack import fft, ifft
from scipy.signal import gaussian

#from ..helper import nextpow2
#from ..functions import BaseAnalysisFunction

# Source: https://github.com/mmcauliffe/Conch-sounds/blob/master/conch/analysis/helper.py
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    return np.ceil(np.log2(np.abs(x)))

def lpc_ref(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)

    Notes
    ----
    This is just for reference, as it is using the direct inversion of the
    toeplitz matrix, which is really slow"""
    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, 'float32')
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size - 1:signal.size + order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype='float32')


# @jit
def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1 / r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order + 1, 'float32')
    # temporary array
    t = np.empty(order + 1, 'float32')
    # Reflection coefficients
    k = np.empty(order, 'float32')

    a[0] = 1.
    e = r[0]

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k[i - 1] = -acc / e
        a[i] = k[i - 1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i - 1] * np.conj(t[i - j])

        e *= 1 - k[i - 1] * np.conj(k[i - 1])

    return a, e, k


# @jit
def _acorr_last_axis(x, nfft, maxlag):
    a = np.real(ifft(np.abs(fft(x, n=nfft) ** 2)))
    return a[..., :maxlag + 1] / x.shape[-1]


# @jit
def acorr_lpc(x, axis=-1):
    """Compute autocorrelation of x along the given axis.

    This compute the biased autocorrelation estimator (divided by the size of
    input signal)

    Notes
    -----
        The reason why we do not use acorr directly is for speed issue."""
    if not np.isrealobj(x):
        raise ValueError("Complex input not supported yet")

    maxlag = x.shape[axis]
    nfft = int(2 ** nextpow2(2 * maxlag - 1))

    if axis != -1:
        x = np.swapaxes(x, -1, axis)
    a = _acorr_last_axis(x, nfft, maxlag)
    if axis != -1:
        a = np.swapaxes(a, -1, axis)
    return a


# @jit
def lpc(signal, order, axis=-1):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)

    Returns
    -------
    a : array-like
        the solution of the inversion.
    e : array-like
        the prediction error.
    k : array-like
        reflection coefficients.

    Notes
    -----
    This uses Levinson-Durbin recursion for the autocorrelation matrix
    inversion, and fft for the autocorrelation computation.

    For small order, particularly if order << signal size, direct computation
    of the autocorrelation is faster: use levinson and correlate in this case."""
    n = signal.shape[axis]
    if order > n:
        raise ValueError("Input signal must have length >= order")

    r = acorr_lpc(signal, axis)
    return levinson_1d(r, order)


def process_frame(X, window, num_formants, new_sr):
    X = X * window
    A, e, k = lpc(X, num_formants * 2)

    rts = np.roots(A)
    rts = rts[np.where(np.imag(rts) >= 0)]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * (new_sr / (2 * np.pi))
    frq_inds = np.argsort(frqs)
    frqs = frqs[frq_inds]
    bw = -1 / 2 * (new_sr / (2 * np.pi)) * np.log(np.abs(rts[frq_inds]))
    return frqs, bw


def lpc_formants(signal, sr, num_formants, max_freq, time_step,
                 win_len, window_shape='gaussian'):
    output = {}
    new_sr = 2 * max_freq
    alpha = np.exp(-2 * np.pi * 50 * (1 / new_sr))
    proc = lfilter([1., -alpha], 1, signal)
    if sr > new_sr:
        proc = librosa.resample(proc, sr, new_sr)
    nperseg = int(win_len * new_sr)
    nperstep = int(time_step * new_sr)
    if window_shape == 'gaussian':
        window = gaussian(nperseg + 2, 0.45 * (nperseg - 1) / 2)[1:nperseg + 1]
    else:
        window = np.hanning(nperseg + 2)[1:nperseg + 1]
    indices = np.arange(int(nperseg / 2), proc.shape[0] - int(nperseg / 2) + 1, nperstep)
    num_frames = len(indices)
    for i in range(num_frames):
        if nperseg % 2 != 0:
            X = proc[indices[i] - int(nperseg / 2):indices[i] + int(nperseg / 2) + 1]
        else:
            X = proc[indices[i] - int(nperseg / 2):indices[i] + int(nperseg / 2)]
        frqs, bw = process_frame(X, window, num_formants, new_sr)
        formants = []
        for j, f in enumerate(frqs):
            if f < 50:
                continue
            if f > max_freq - 50:
                continue
            formants.append((np.asscalar(f), np.asscalar(bw[j])))
        missing = num_formants - len(formants)
        if missing:
            formants += [(None, None)] * missing
        output[indices[i] / new_sr] = formants
    return output


#class FormantTrackFunction(BaseAnalysisFunction):
#    def __init__(self, num_formants=5, max_frequency=5000,
#                 time_step=0.01, window_length=0.025, window_shape='gaussian'):
#        super(FormantTrackFunction, self).__init__()
#        self.arguments = [num_formants, max_frequency, time_step, window_length, window_shape]
#        self._function = lpc_formants
#        self.requires_file = False
