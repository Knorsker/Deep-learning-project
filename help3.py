import copy
import functools
import hashlib
import math
import pathlib
import tempfile
import typing
import warnings
from collections import namedtuple
from pathlib import Path
import subprocess
import shlex
import ffmpy
import json
import torchaudio
import scipy

import julius
import numpy as np
import soundfile
import torch

class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20
    """Gain factor for converting between amplitude and decibels."""
    CODEC_PRESETS = {
        "8-bit": {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
        "GSM-FR": {"format": "gsm"},
        "MP3": {"format": "mp3", "compression": -9},
        "Vorbis": {"format": "vorbis", "compression": -1},
        "Ogg": {
            "format": "ogg",
            "compression": -1,
        },
        "Amr-nb": {"format": "amr-nb"},
    }
    """Presets for applying codecs via torchaudio."""

    def mix(
        self,
        other,
        snr: typing.Union[torch.Tensor, np.ndarray, float] = 10,
        other_eq: typing.Union[torch.Tensor, np.ndarray] = None,
    ):
        """Mixes noise with signal at specified
        signal-to-noise ratio. Optionally, the
        other signal can be equalized in-place.


        Parameters
        ----------
        other : AudioSignal
            AudioSignal object to mix with.
        snr : typing.Union[torch.Tensor, np.ndarray, float], optional
            Signal to noise ratio, by default 10
        other_eq : typing.Union[torch.Tensor, np.ndarray], optional
            EQ curve to apply to other signal, if any, by default None

        Returns
        -------
        AudioSignal
            In-place modification of AudioSignal.
        """
        snr = ensure_tensor(snr).to(self.device)

        pad_len = max(0, self.signal_length - other.signal_length)
        other.zero_pad(0, pad_len)
        other.truncate_samples(self.signal_length)
        if other_eq is not None:
            other = other.equalizer(other_eq)

        tgt_loudness = self.loudness() - snr
        other = other.normalize(tgt_loudness)

        self.audio_data = self.audio_data + other.audio_data
        return self

    def convolve(self, other, start_at_max: bool = True):
        """Convolves self with other.
        This function uses FFTs to do the convolution.

        Parameters
        ----------
        other : AudioSignal
            Signal to convolve with.
        start_at_max : bool, optional
            Whether to start at the max value of other signal, to
            avoid inducing delays, by default True

        Returns
        -------
        AudioSignal
            Convolved signal, in-place.
        """
        from . import AudioSignal

        pad_len = self.signal_length - other.signal_length

        if pad_len > 0:
            other.zero_pad(0, pad_len)
        else:
            other.truncate_samples(self.signal_length)

        if start_at_max:
            # Use roll to rotate over the max for every item
            # so that the impulse responses don't induce any
            # delay.
            idx = other.audio_data.abs().argmax(axis=-1)
            irs = torch.zeros_like(other.audio_data)
            for i in range(other.batch_size):
                irs[i] = torch.roll(other.audio_data[i], -idx[i].item(), -1)
            other = AudioSignal(irs, other.sample_rate)

        delta = torch.zeros_like(other.audio_data)
        delta[..., 0] = 1

        length = self.signal_length
        delta_fft = torch.fft.rfft(delta, length)
        other_fft = torch.fft.rfft(other.audio_data, length)
        self_fft = torch.fft.rfft(self.audio_data, length)

        convolved_fft = other_fft * self_fft
        convolved_audio = torch.fft.irfft(convolved_fft, length)

        delta_convolved_fft = other_fft * delta_fft
        delta_audio = torch.fft.irfft(delta_convolved_fft, length)

        # Use the delta to rescale the audio exactly as needed.
        delta_max = delta_audio.abs().max(dim=-1, keepdims=True)[0]
        scale = 1 / delta_max.clamp(1e-5)
        convolved_audio = convolved_audio * scale

        self.audio_data = convolved_audio

        return self

    def apply_ir(
        self,
        ir,
        drr: typing.Union[torch.Tensor, np.ndarray, float] = None,
        ir_eq: typing.Union[torch.Tensor, np.ndarray] = None,
        use_original_phase: bool = False,
    ):
        """Applies an impulse response to the signal. If ` is`ir_eq``
        is specified, the impulse response is equalized before
        it is applied, using the given curve.

        Parameters
        ----------
        ir : AudioSignal
            Impulse response to convolve with.
        drr : typing.Union[torch.Tensor, np.ndarray, float], optional
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None
        ir_eq : typing.Union[torch.Tensor, np.ndarray], optional
            Equalization that will be applied to impulse response
            if specified, by default None
        use_original_phase : bool, optional
            Whether to use the original phase, instead of the convolved
            phase, by default False

        Returns
        -------
        AudioSignal
            Signal with impulse response applied to it
        """
        if ir_eq is not None:
            ir = ir.equalizer(ir_eq)
        if drr is not None:
            ir = ir.alter_drr(drr)

        # Save the peak before
        max_spk = self.audio_data.abs().max(dim=-1, keepdims=True).values

        # Augment the impulse response to simulate microphone effects
        # and with varying direct-to-reverberant ratio.
        phase = self.phase
        self.convolve(ir)

        # Use the input phase
        if use_original_phase:
            self.stft()
            self.stft_data = self.magnitude * torch.exp(1j * phase)
            self.istft()

        # Rescale to the input's amplitude
        max_transformed = self.audio_data.abs().max(dim=-1, keepdims=True).values
        scale_factor = max_spk.clamp(1e-8) / max_transformed.clamp(1e-8)
        self = self * scale_factor

        return self

    def ensure_max_of_audio(self, max: float = 1.0):
        """Ensures that ``abs(audio_data) <= max``.

        Parameters
        ----------
        max : float, optional
            Max absolute value of signal, by default 1.0

        Returns
        -------
        AudioSignal
            Signal with values scaled between -max and max.
        """
        peak = self.audio_data.abs().max(dim=-1, keepdims=True)[0]
        peak_gain = torch.ones_like(peak)
        peak_gain[peak > max] = max / peak[peak > max]
        self.audio_data = self.audio_data * peak_gain
        return self

    def normalize(self, db: typing.Union[torch.Tensor, np.ndarray, float] = -24.0):
        """Normalizes the signal's volume to the specified db, in LUFS.
        This is GPU-compatible, making for very fast loudness normalization.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray, float], optional
            Loudness to normalize to, by default -24.0

        Returns
        -------
        AudioSignal
            Normalized audio signal.
        """
        db = ensure_tensor(db).to(self.device)
        ref_db = self.loudness()
        gain = db - ref_db
        gain = torch.exp(gain * self.GAIN_FACTOR)

        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def volume_change(self, db: typing.Union[torch.Tensor, np.ndarray, float]):
        """Change volume of signal by some amount, in dB.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray, float]
            Amount to change volume by.

        Returns
        -------
        AudioSignal
            Signal at new volume.
        """
        db = ensure_tensor(db, ndim=1).to(self.device)
        gain = torch.exp(db * self.GAIN_FACTOR)
        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def _to_2d(self):
        waveform = self.audio_data.reshape(-1, self.signal_length)
        return waveform

    def _to_3d(self, waveform):
        return waveform.reshape(self.batch_size, self.num_channels, -1)

    def pitch_shift(self, n_semitones: int, quick: bool = True):
        """Pitch shift the signal. All items in the batch
        get the same pitch shift.

        Parameters
        ----------
        n_semitones : int
            How many semitones to shift the signal by.
        quick : bool, optional
            Using quick pitch shifting, by default True

        Returns
        -------
        AudioSignal
            Pitch shifted audio signal.
        """
        device = self.device
        effects = [
            ["pitch", str(n_semitones * 100)],
            ["rate", str(self.sample_rate)],
        ]
        if quick:
            effects[0].insert(1, "-q")

        waveform = self._to_2d().cpu()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self.to(device)

    def time_stretch(self, factor: float, quick: bool = True):
        """Time stretch the audio signal.

        Parameters
        ----------
        factor : float
            Factor by which to stretch the AudioSignal. Typically
            between 0.8 and 1.2.
        quick : bool, optional
            Whether to use quick time stretching, by default True

        Returns
        -------
        AudioSignal
            Time-stretched AudioSignal.
        """
        device = self.device
        effects = [
            ["tempo", str(factor)],
            ["rate", str(self.sample_rate)],
        ]
        if quick:
            effects[0].insert(1, "-q")

        waveform = self._to_2d().cpu()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self.to(device)

    def apply_codec(
        self,
        preset: str = None,
        format: str = "wav",
        encoding: str = None,
        bits_per_sample: int = None,
        compression: int = None,
    ):  # pragma: no cover
        """Applies an audio codec to the signal.

        Parameters
        ----------
        preset : str, optional
            One of the keys in ``self.CODEC_PRESETS``, by default None
        format : str, optional
            Format for audio codec, by default "wav"
        encoding : str, optional
            Encoding to use, by default None
        bits_per_sample : int, optional
            How many bits per sample, by default None
        compression : int, optional
            Compression amount of codec, by default None

        Returns
        -------
        AudioSignal
            AudioSignal with codec applied.

        Raises
        ------
        ValueError
            If preset is not in ``self.CODEC_PRESETS``, an error
            is thrown.
        """
        torchaudio_version_070 = "0.7" in torchaudio.__version__
        if torchaudio_version_070:
            return self

        kwargs = {
            "format": format,
            "encoding": encoding,
            "bits_per_sample": bits_per_sample,
            "compression": compression,
        }

        if preset is not None:
            if preset in self.CODEC_PRESETS:
                kwargs = self.CODEC_PRESETS[preset]
            else:
                raise ValueError(
                    f"Unknown preset: {preset}. "
                    f"Known presets: {list(self.CODEC_PRESETS.keys())}"
                )

        waveform = self._to_2d()
        if kwargs["format"] in ["vorbis", "mp3", "ogg", "amr-nb"]:
            # Apply it in a for loop
            augmented = torch.cat(
                [
                    torchaudio.functional.apply_codec(
                        waveform[i][None, :], self.sample_rate, **kwargs
                    )
                    for i in range(waveform.shape[0])
                ],
                dim=0,
            )
        else:
            augmented = torchaudio.functional.apply_codec(
                waveform, self.sample_rate, **kwargs
            )
        augmented = self._to_3d(augmented)

        self.audio_data = augmented
        return self

    def mel_filterbank(self, n_bands: int):
        """Breaks signal into mel bands.

        Parameters
        ----------
        n_bands : int
            Number of mel bands to use.

        Returns
        -------
        torch.Tensor
            Mel-filtered bands, with last axis being the band index.
        """
        filterbank = (
            julius.SplitBands(self.sample_rate, n_bands).float().to(self.device)
        )
        filtered = filterbank(self.audio_data)
        return filtered.permute(1, 2, 3, 0)

    def equalizer(self, db: typing.Union[torch.Tensor, np.ndarray]):
        """Applies a mel-spaced equalizer to the audio signal.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray]
            EQ curve to apply.

        Returns
        -------
        AudioSignal
            AudioSignal with equalization applied.
        """
        db = ensure_tensor(db)
        n_bands = db.shape[-1]
        fbank = self.mel_filterbank(n_bands)

        # If there's a batch dimension, make sure it's the same.
        if db.ndim == 2:
            if db.shape[0] != 1:
                assert db.shape[0] == fbank.shape[0]
        else:
            db = db.unsqueeze(0)

        weights = (10**db).to(self.device).float()
        fbank = fbank * weights[:, None, None, :]
        eq_audio_data = fbank.sum(-1)
        self.audio_data = eq_audio_data
        return self

    def clip_distortion(
        self, clip_percentile: typing.Union[torch.Tensor, np.ndarray, float]
    ):
        """Clips the signal at a given percentile. The higher it is,
        the lower the threshold for clipping.

        Parameters
        ----------
        clip_percentile : typing.Union[torch.Tensor, np.ndarray, float]
            Values are between 0.0 to 1.0. Typical values are 0.1 or below.

        Returns
        -------
        AudioSignal
            Audio signal with clipped audio data.
        """
        clip_percentile = ensure_tensor(clip_percentile, ndim=1)
        min_thresh = torch.quantile(self.audio_data, clip_percentile / 2, dim=-1)
        max_thresh = torch.quantile(self.audio_data, 1 - (clip_percentile / 2), dim=-1)

        nc = self.audio_data.shape[1]
        min_thresh = min_thresh[:, :nc, :]
        max_thresh = max_thresh[:, :nc, :]

        self.audio_data = self.audio_data.clamp(min_thresh, max_thresh)

        return self

    def quantization(
        self, quantization_channels: typing.Union[torch.Tensor, np.ndarray, int]
    ):
        """Applies quantization to the input waveform.

        Parameters
        ----------
        quantization_channels : typing.Union[torch.Tensor, np.ndarray, int]
            Number of evenly spaced quantization channels to quantize
            to.

        Returns
        -------
        AudioSignal
            Quantized AudioSignal.
        """
        quantization_channels = ensure_tensor(quantization_channels, ndim=3)

        x = self.audio_data
        x = (x + 1) / 2
        x = x * quantization_channels
        x = x.floor()
        x = x / quantization_channels
        x = 2 * x - 1

        residual = (self.audio_data - x).detach()
        self.audio_data = self.audio_data - residual
        return self

    def mulaw_quantization(
        self, quantization_channels: typing.Union[torch.Tensor, np.ndarray, int]
    ):
        """Applies mu-law quantization to the input waveform.

        Parameters
        ----------
        quantization_channels : typing.Union[torch.Tensor, np.ndarray, int]
            Number of mu-law spaced quantization channels to quantize
            to.

        Returns
        -------
        AudioSignal
            Quantized AudioSignal.
        """
        mu = quantization_channels - 1.0
        mu = ensure_tensor(mu, ndim=3)

        x = self.audio_data

        # quantize
        x = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        x = ((x + 1) / 2 * mu + 0.5).to(torch.int64)

        # unquantize
        x = (x / mu) * 2 - 1.0
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu

        residual = (self.audio_data - x).detach()
        self.audio_data = self.audio_data - residual
        return self

    def __matmul__(self, other):
        return self.convolve(other)

class Meter(torch.nn.Module):
    """Tensorized version of pyloudnorm.Meter. Works with batched audio tensors.

    Parameters
    ----------
    rate : int
        Sample rate of audio.
    filter_class : str, optional
        Class of weighting filter used.
        K-weighting' (default), 'Fenton/Lee 1'
        'Fenton/Lee 2', 'Dash et al.'
        by default "K-weighting"
    block_size : float, optional
        Gating block size in seconds, by default 0.400
    zeros : int, optional
         Number of zeros to use in FIR approximation of
         IIR filters, by default 512
    use_fir : bool, optional
        Whether to use FIR approximation or exact IIR formulation.
        If computing on GPU, ``use_fir=True`` will be used, as its
        much faster, by default False
    """

    def __init__(
        self,
        rate: int,
        filter_class: str = "K-weighting",
        block_size: float = 0.400,
        zeros: int = 512,
        use_fir: bool = False,
    ):
        super().__init__()

        self.rate = rate
        self.filter_class = filter_class
        self.block_size = block_size
        self.use_fir = use_fir

        G = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.41, 1.41]))
        self.register_buffer("G", G)

        # Compute impulse responses so that filtering is fast via
        # a convolution at runtime, on GPU, unlike lfilter.
        impulse = np.zeros((zeros,))
        impulse[..., 0] = 1.0

        firs = np.zeros((len(self._filters), 1, zeros))
        passband_gain = torch.zeros(len(self._filters))

        for i, (_, filter_stage) in enumerate(self._filters.items()):
            firs[i] = scipy.signal.lfilter(filter_stage.b, filter_stage.a, impulse)
            passband_gain[i] = filter_stage.passband_gain

        firs = torch.from_numpy(firs[..., ::-1].copy()).float()

        self.register_buffer("firs", firs)
        self.register_buffer("passband_gain", passband_gain)

    def apply_filter_gpu(self, data: torch.Tensor):
        """Performs FIR approximation of loudness computation.

        Parameters
        ----------
        data : torch.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        torch.Tensor
            Filtered audio data.
        """
        # Data is of shape (nb, nch, nt)
        # Reshape to (nb*nch, 1, nt)
        nb, nt, nch = data.shape
        data = data.permute(0, 2, 1)
        data = data.reshape(nb * nch, 1, nt)

        # Apply padding
        pad_length = self.firs.shape[-1]

        # Apply filtering in sequence
        for i in range(self.firs.shape[0]):
            data = F.pad(data, (pad_length, pad_length))
            data = julius.fftconv.fft_conv1d(data, self.firs[i, None, ...])
            data = self.passband_gain[i] * data
            data = data[..., 1 : nt + 1]

        data = data.permute(0, 2, 1)
        data = data[:, :nt, :]
        return data

    def apply_filter_cpu(self, data: torch.Tensor):
        """Performs IIR formulation of loudness computation.

        Parameters
        ----------
        data : torch.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        torch.Tensor
            Filtered audio data.
        """
        for _, filter_stage in self._filters.items():
            passband_gain = filter_stage.passband_gain

            a_coeffs = torch.from_numpy(filter_stage.a).float().to(data.device)
            b_coeffs = torch.from_numpy(filter_stage.b).float().to(data.device)

            _data = data.permute(0, 2, 1)
            filtered = torchaudio.functional.lfilter(
                _data, a_coeffs, b_coeffs, clamp=False
            )
            data = passband_gain * filtered.permute(0, 2, 1)
        return data

    def apply_filter(self, data: torch.Tensor):
        """Applies filter on either CPU or GPU, depending
        on if the audio is on GPU or is on CPU, or if
        ``self.use_fir`` is True.

        Parameters
        ----------
        data : torch.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        torch.Tensor
            Filtered audio data.
        """
        if data.is_cuda or self.use_fir:
            data = self.apply_filter_gpu(data)
        else:
            data = self.apply_filter_cpu(data)
        return data

    def forward(self, data: torch.Tensor):
        """Computes integrated loudness of data.

        Parameters
        ----------
        data : torch.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        torch.Tensor
            Filtered audio data.
        """
        return self.integrated_loudness(data)

    def _unfold(self, input_data):
        T_g = self.block_size
        overlap = 0.75  # overlap of 75% of the block duration
        step = 1.0 - overlap  # step size by percentage

        kernel_size = int(T_g * self.rate)
        stride = int(T_g * self.rate * step)
        unfolded = julius.core.unfold(input_data.permute(0, 2, 1), kernel_size, stride)
        unfolded = unfolded.transpose(-1, -2)

        return unfolded

    def integrated_loudness(self, data: torch.Tensor):
        """Computes integrated loudness of data.

        Parameters
        ----------
        data : torch.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        torch.Tensor
            Filtered audio data.
        """
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        else:
            data = data.float()

        input_data = copy.copy(data)
        # Data always has a batch and channel dimension.
        # Is of shape (nb, nt, nch)
        if input_data.ndim < 2:
            input_data = input_data.unsqueeze(-1)
        if input_data.ndim < 3:
            input_data = input_data.unsqueeze(0)

        nb, nt, nch = input_data.shape

        # Apply frequency weighting filters - account
        # for the acoustic respose of the head and auditory system
        input_data = self.apply_filter(input_data)

        G = self.G  # channel gains
        T_g = self.block_size  # 400 ms gating block standard
        Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold

        unfolded = self._unfold(input_data)

        z = (1.0 / (T_g * self.rate)) * unfolded.square().sum(2)
        l = -0.691 + 10.0 * torch.log10((G[None, :nch, None] * z).sum(1, keepdim=True))
        l = l.expand_as(z)

        # find gating block indices above absolute threshold
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        masked = l > Gamma_a
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2)

        # calculate the relative threshold value (see eq. 6)
        Gamma_r = (
            -0.691 + 10.0 * torch.log10((z_avg_gated * G[None, :nch]).sum(-1)) - 10.0
        )
        Gamma_r = Gamma_r[:, None, None]
        Gamma_r = Gamma_r.expand(nb, nch, l.shape[-1])

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        z_avg_gated[l <= Gamma_r] = 0
        masked = (l > Gamma_a) * (l > Gamma_r)
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2)

        # # Cannot use nan_to_num (pytorch 1.8 does not come with GCP-supported cuda version)
        # z_avg_gated = torch.nan_to_num(z_avg_gated)
        z_avg_gated = torch.where(
            z_avg_gated.isnan(), torch.zeros_like(z_avg_gated), z_avg_gated
        )
        z_avg_gated[z_avg_gated == float("inf")] = float(np.finfo(np.float32).max)
        z_avg_gated[z_avg_gated == -float("inf")] = float(np.finfo(np.float32).min)

        LUFS = -0.691 + 10.0 * torch.log10((G[None, :nch] * z_avg_gated).sum(1))
        return LUFS.float()

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        from pyloudnorm import Meter

        meter = Meter(self.rate)
        meter.filter_class = value
        self._filter_class = value
        self._filters = meter._filters

class LoudnessMixin:
    _loudness = None
    MIN_LOUDNESS = -70
    """Minimum loudness possible."""

    def loudness(
        self, filter_class: str = "K-weighting", block_size: float = 0.400, **kwargs
    ):
        """Calculates loudness using an implementation of ITU-R BS.1770-4.
        Allows control over gating block size and frequency weighting filters for
        additional control. Measure the integrated gated loudness of a signal.

        API is derived from PyLoudnorm, but this implementation is ported to PyTorch
        and is tensorized across batches. When on GPU, an FIR approximation of the IIR
        filters is used to compute loudness for speed.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.

        Parameters
        ----------
        filter_class : str, optional
            Class of weighting filter used.
            K-weighting' (default), 'Fenton/Lee 1'
            'Fenton/Lee 2', 'Dash et al.'
            by default "K-weighting"
        block_size : float, optional
            Gating block size in seconds, by default 0.400
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.loudness.Meter`.

        Returns
        -------
        torch.Tensor
            Loudness of audio data.
        """
        if self._loudness is not None:
            return self._loudness.to(self.device)
        original_length = self.signal_length
        if self.signal_duration < 0.5:
            pad_len = int((0.5 - self.signal_duration) * self.sample_rate)
            self.zero_pad(0, pad_len)

        # create BS.1770 meter
        meter = Meter(
            self.sample_rate, filter_class=filter_class, block_size=block_size, **kwargs
        )
        meter = meter.to(self.device)
        # measure loudness
        loudness = meter.integrated_loudness(self.audio_data.permute(0, 2, 1))
        self.truncate_samples(original_length)
        min_loudness = (
            torch.ones_like(loudness, device=loudness.device) * self.MIN_LOUDNESS
        )
        self._loudness = torch.maximum(loudness, min_loudness)

        return self._loudness.to(self.device)

class PlayMixin:
    def embed(self, ext: str = None, display: bool = True, return_html: bool = False):
        """Embeds audio as a playable audio embed in a notebook, or HTML
        document, etc.

        Parameters
        ----------
        ext : str, optional
            Extension to use when saving the audio, by default ".wav"
        display : bool, optional
            This controls whether or not to display the audio when called. This
            is used when the embed is the last line in a Jupyter cell, to prevent
            the audio from being embedded twice, by default True
        return_html : bool, optional
            Whether to return the data wrapped in an HTML audio element, by default False

        Returns
        -------
        str
            Either the element for display, or the HTML string of it.
        """
        if ext is None:
            ext = DEFAULT_EXTENSION
        ext = f".{ext}" if not ext.startswith(".") else ext
        ffmpy, IPython = _check_imports()
        sr = self.sample_rate
        tmpfiles = []

        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(mode="w+", suffix=".wav", delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name)
            if ext != ".wav" and ffmpy:
                tmp_converted = NamedTemporaryFile(mode="w+", suffix=ext, delete=False)
                tmpfiles.append(tmp_wav)
                ff = ffmpy.FFmpeg(
                    inputs={tmp_wav.name: None},
                    outputs={
                        tmp_converted.name: "-write_xing 0 -codec:a libmp3lame -b:a 128k -y -hide_banner -loglevel error"
                    },
                )
                ff.run()
            else:
                tmp_converted = tmp_wav

            audio_element = IPython.display.Audio(data=tmp_converted.name, rate=sr)
            if display:
                IPython.display.display(audio_element)

        if return_html:
            audio_element = (
                f"<audio "
                f"  controls "
                f"  src='{audio_element.src_attr()}'> "
                f"</audio> "
            )
        return audio_element

    def widget(
        self,
        title: str = None,
        ext: str = ".wav",
        add_headers: bool = True,
        player_width: str = "100%",
        margin: str = "10px",
        plot_fn: str = "specshow",
        return_html: bool = False,
        **kwargs,
    ):
        """Creates a playable widget with spectrogram. Inspired (heavily) by
        https://sjvasquez.github.io/blog/melnet/.

        Parameters
        ----------
        title : str, optional
            Title of plot, placed in upper right of top-most axis.
        ext : str, optional
            Extension for embedding, by default ".mp3"
        add_headers : bool, optional
            Whether or not to add headers (use for first embed, False for later embeds), by default True
        player_width : str, optional
            Width of the player, as a string in a CSS rule, by default "100%"
        margin : str, optional
            Margin on all sides of player, by default "10px"
        plot_fn : function, optional
            Plotting function to use (by default self.specshow).
        return_html : bool, optional
            Whether to return the data wrapped in an HTML audio element, by default False
        kwargs : dict, optional
            Keyword arguments to plot_fn (by default self.specshow).

        Returns
        -------
        HTML
            HTML object.
        """
        import matplotlib.pyplot as plt

        def _save_fig_to_tag():
            buffer = io.BytesIO()

            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close()

            buffer.seek(0)
            data_uri = base64.b64encode(buffer.read()).decode("ascii")
            tag = "data:image/png;base64,{0}".format(data_uri)

            return tag

        _, IPython = _check_imports()

        header_html = ""

        if add_headers:
            header_html = headers.replace("PLAYER_WIDTH", str(player_width))
            header_html = header_html.replace("MARGIN", str(margin))
            IPython.display.display(IPython.display.HTML(header_html))

        widget_html = widget
        if isinstance(plot_fn, str):
            plot_fn = getattr(self, plot_fn)
            kwargs["title"] = title
        plot_fn(**kwargs)

        fig = plt.gcf()
        pixels = fig.get_size_inches() * fig.dpi

        tag = _save_fig_to_tag()

        # Make the source image for the levels
        self.specshow()
        format_figure((12, 1.5))
        levels_tag = _save_fig_to_tag()

        player_id = "".join(random.choice(string.ascii_uppercase) for _ in range(10))

        audio_elem = self.embed(ext=ext, display=False)
        widget_html = widget_html.replace("AUDIO_SRC", audio_elem.src_attr())
        widget_html = widget_html.replace("IMAGE_SRC", tag)
        widget_html = widget_html.replace("LEVELS_SRC", levels_tag)
        widget_html = widget_html.replace("PLAYER_ID", player_id)

        # Calculate width/height of figure based on figure size.
        widget_html = widget_html.replace("PADDING_AMOUNT", f"{int(pixels[1])}px")
        widget_html = widget_html.replace("MAX_WIDTH", f"{int(pixels[0])}px")

        IPython.display.display(IPython.display.HTML(widget_html))

        if return_html:
            html = header_html if add_headers else ""
            html += widget_html
            return html

    def play(self):
        """
        Plays an audio signal if ffplay from the ffmpeg suite of tools is installed.
        Otherwise, will fail. The audio signal is written to a temporary file
        and then played with ffplay.
        """
        tmpfiles = []
        with _close_temp_files(tmpfiles):
            tmp_wav = NamedTemporaryFile(suffix=".wav", delete=False)
            tmpfiles.append(tmp_wav)
            self.write(tmp_wav.name)
            print(self)
            subprocess.call(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    tmp_wav.name,
                ]
            )
        return self

class ImpulseResponseMixin:
    """These functions are generally only used with AudioSignals that are derived
    from impulse responses, not other sources like music or speech. These methods
    are used to replicate the data augmentation described in [1].

    1.  Bryan, Nicholas J. "Impulse response data augmentation and deep
        neural networks for blind room acoustic parameter estimation."
        ICASSP 2020-2020 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2020.
    """

    def decompose_ir(self):
        """Decomposes an impulse response into early and late
        field responses.
        """
        # Equations 1 and 2
        # -----------------
        # Breaking up into early
        # response + late field response.

        td = torch.argmax(self.audio_data, dim=-1, keepdim=True)
        t0 = int(self.sample_rate * 0.0025)

        idx = torch.arange(self.audio_data.shape[-1], device=self.device)[None, None, :]
        idx = idx.expand(self.batch_size, -1, -1)
        early_idx = (idx >= td - t0) * (idx <= td + t0)

        early_response = torch.zeros_like(self.audio_data, device=self.device)
        early_response[early_idx] = self.audio_data[early_idx]

        late_idx = ~early_idx
        late_field = torch.zeros_like(self.audio_data, device=self.device)
        late_field[late_idx] = self.audio_data[late_idx]

        # Equation 4
        # ----------
        # Decompose early response into windowed
        # direct path and windowed residual.

        window = torch.zeros_like(self.audio_data, device=self.device)
        for idx in range(self.batch_size):
            window_idx = early_idx[idx, 0].nonzero()
            window[idx, ..., window_idx] = self.get_window(
                "hann", window_idx.shape[-1], self.device
            )
        return early_response, late_field, window

    def measure_drr(self):
        """Measures the direct-to-reverberant ratio of the impulse
        response.

        Returns
        -------
        float
            Direct-to-reverberant ratio
        """
        early_response, late_field, _ = self.decompose_ir()
        num = (early_response**2).sum(dim=-1)
        den = (late_field**2).sum(dim=-1)
        drr = 10 * torch.log10(num / den)
        return drr

    @staticmethod
    def solve_alpha(early_response, late_field, wd, target_drr):
        """Used to solve for the alpha value, which is used
        to alter the drr.
        """
        # Equation 5
        # ----------
        # Apply the good ol' quadratic formula.

        wd_sq = wd**2
        wd_sq_1 = (1 - wd) ** 2
        e_sq = early_response**2
        l_sq = late_field**2
        a = (wd_sq * e_sq).sum(dim=-1)
        b = (2 * (1 - wd) * wd * e_sq).sum(dim=-1)
        c = (wd_sq_1 * e_sq).sum(dim=-1) - torch.pow(10, target_drr / 10) * l_sq.sum(
            dim=-1
        )

        expr = ((b**2) - 4 * a * c).sqrt()
        alpha = torch.maximum(
            (-b - expr) / (2 * a),
            (-b + expr) / (2 * a),
        )
        return alpha

    def alter_drr(self, drr: typing.Union[torch.Tensor, np.ndarray, float]):
        """Alters the direct-to-reverberant ratio of the impulse response.

        Parameters
        ----------
        drr : typing.Union[torch.Tensor, np.ndarray, float]
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None

        Returns
        -------
        AudioSignal
            Altered impulse response.
        """
        drr = util.ensure_tensor(drr, 2, self.batch_size).to(self.device)

        early_response, late_field, window = self.decompose_ir()
        alpha = self.solve_alpha(early_response, late_field, window, drr)
        min_alpha = (
            late_field.abs().max(dim=-1)[0] / early_response.abs().max(dim=-1)[0]
        )
        alpha = torch.maximum(alpha, min_alpha)[..., None]

        aug_ir_data = (
            alpha * window * early_response
            + ((1 - window) * early_response)
            + late_field
        )
        self.audio_data = aug_ir_data
        self.ensure_max_of_audio()
        return self

class DSPMixin:
    _original_batch_size = None
    _original_num_channels = None
    _padded_signal_length = None

    def _preprocess_signal_for_windowing(self, window_duration, hop_duration):
        self._original_batch_size = self.batch_size
        self._original_num_channels = self.num_channels

        window_length = int(window_duration * self.sample_rate)
        hop_length = int(hop_duration * self.sample_rate)

        if window_length % hop_length != 0:
            factor = window_length // hop_length
            window_length = factor * hop_length

        self.zero_pad(hop_length, hop_length)
        self._padded_signal_length = self.signal_length

        return window_length, hop_length

    def windows(
        self, window_duration: float, hop_duration: float, preprocess: bool = True
    ):
        """Generator which yields windows of specified duration from signal with a specified
        hop length.

        Parameters
        ----------
        window_duration : float
            Duration of every window in seconds.
        hop_duration : float
            Hop between windows in seconds.
        preprocess : bool, optional
            Whether to preprocess the signal, so that the first sample is in
            the middle of the first window, by default True

        Yields
        ------
        AudioSignal
            Each window is returned as an AudioSignal.
        """
        if preprocess:
            window_length, hop_length = self._preprocess_signal_for_windowing(
                window_duration, hop_duration
            )

        self.audio_data = self.audio_data.reshape(-1, 1, self.signal_length)

        for b in range(self.batch_size):
            i = 0
            start_idx = i * hop_length
            while True:
                start_idx = i * hop_length
                i += 1
                end_idx = start_idx + window_length
                if end_idx > self.signal_length:
                    break
                yield self[b, ..., start_idx:end_idx]

    def collect_windows(
        self, window_duration: float, hop_duration: float, preprocess: bool = True
    ):
        """Reshapes signal into windows of specified duration from signal with a specified
        hop length. Window are placed along the batch dimension. Use with
        :py:func:`audiotools.core.dsp.DSPMixin.overlap_and_add` to reconstruct the
        original signal.

        Parameters
        ----------
        window_duration : float
            Duration of every window in seconds.
        hop_duration : float
            Hop between windows in seconds.
        preprocess : bool, optional
            Whether to preprocess the signal, so that the first sample is in
            the middle of the first window, by default True

        Returns
        -------
        AudioSignal
            AudioSignal unfolded with shape ``(nb * nch * num_windows, 1, window_length)``
        """
        if preprocess:
            window_length, hop_length = self._preprocess_signal_for_windowing(
                window_duration, hop_duration
            )

        # self.audio_data: (nb, nch, nt).
        unfolded = torch.nn.functional.unfold(
            self.audio_data.reshape(-1, 1, 1, self.signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length),
        )
        # unfolded: (nb * nch, window_length, num_windows).
        # -> (nb * nch * num_windows, 1, window_length)
        unfolded = unfolded.permute(0, 2, 1).reshape(-1, 1, window_length)
        self.audio_data = unfolded
        return self

    def overlap_and_add(self, hop_duration: float):
        """Function which takes a list of windows and overlap adds them into a
        signal the same length as ``audio_signal``.

        Parameters
        ----------
        hop_duration : float
            How much to shift for each window
            (overlap is window_duration - hop_duration) in seconds.

        Returns
        -------
        AudioSignal
            overlap-and-added signal.
        """
        hop_length = int(hop_duration * self.sample_rate)
        window_length = self.signal_length

        nb, nch = self._original_batch_size, self._original_num_channels

        unfolded = self.audio_data.reshape(nb * nch, -1, window_length).permute(0, 2, 1)
        folded = torch.nn.functional.fold(
            unfolded,
            output_size=(1, self._padded_signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length),
        )

        norm = torch.ones_like(unfolded, device=unfolded.device)
        norm = torch.nn.functional.fold(
            norm,
            output_size=(1, self._padded_signal_length),
            kernel_size=(1, window_length),
            stride=(1, hop_length),
        )

        folded = folded / norm

        folded = folded.reshape(nb, nch, -1)
        self.audio_data = folded
        self.trim(hop_length, hop_length)
        return self

    def low_pass(
        self, cutoffs: typing.Union[torch.Tensor, np.ndarray, float], zeros: int = 51
    ):
        """Low-passes the signal in-place. Each item in the batch
        can have a different low-pass cutoff, if the input
        to this signal is an array or tensor. If a float, all
        items are given the same low-pass filter.

        Parameters
        ----------
        cutoffs : typing.Union[torch.Tensor, np.ndarray, float]
            Cutoff in Hz of low-pass filter.
        zeros : int, optional
            Number of taps to use in low-pass filter, by default 51

        Returns
        -------
        AudioSignal
            Low-passed AudioSignal.
        """
        cutoffs = util.ensure_tensor(cutoffs, 2, self.batch_size)
        cutoffs = cutoffs / self.sample_rate
        filtered = torch.empty_like(self.audio_data)

        for i, cutoff in enumerate(cutoffs):
            lp_filter = julius.LowPassFilter(cutoff.cpu(), zeros=zeros).to(self.device)
            filtered[i] = lp_filter(self.audio_data[i])

        self.audio_data = filtered
        self.stft_data = None
        return self

    def high_pass(
        self, cutoffs: typing.Union[torch.Tensor, np.ndarray, float], zeros: int = 51
    ):
        """High-passes the signal in-place. Each item in the batch
        can have a different high-pass cutoff, if the input
        to this signal is an array or tensor. If a float, all
        items are given the same high-pass filter.

        Parameters
        ----------
        cutoffs : typing.Union[torch.Tensor, np.ndarray, float]
            Cutoff in Hz of high-pass filter.
        zeros : int, optional
            Number of taps to use in high-pass filter, by default 51

        Returns
        -------
        AudioSignal
            High-passed AudioSignal.
        """
        cutoffs = util.ensure_tensor(cutoffs, 2, self.batch_size)
        cutoffs = cutoffs / self.sample_rate
        filtered = torch.empty_like(self.audio_data)

        for i, cutoff in enumerate(cutoffs):
            hp_filter = julius.HighPassFilter(cutoff.cpu(), zeros=zeros).to(self.device)
            filtered[i] = hp_filter(self.audio_data[i])

        self.audio_data = filtered
        self.stft_data = None
        return self

    def mask_frequencies(
        self,
        fmin_hz: typing.Union[torch.Tensor, np.ndarray, float],
        fmax_hz: typing.Union[torch.Tensor, np.ndarray, float],
        val: float = 0.0,
    ):
        """Masks frequencies between ``fmin_hz`` and ``fmax_hz``, and fills them
        with the value specified by ``val``. Useful for implementing SpecAug.
        The min and max can be different for every item in the batch.

        Parameters
        ----------
        fmin_hz : typing.Union[torch.Tensor, np.ndarray, float]
            Lower end of band to mask out.
        fmax_hz : typing.Union[torch.Tensor, np.ndarray, float]
            Upper end of band to mask out.
        val : float, optional
            Value to fill in, by default 0.0

        Returns
        -------
        AudioSignal
            Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
            masked audio data.
        """
        # SpecAug
        mag, phase = self.magnitude, self.phase
        fmin_hz = util.ensure_tensor(fmin_hz, ndim=mag.ndim)
        fmax_hz = util.ensure_tensor(fmax_hz, ndim=mag.ndim)
        assert torch.all(fmin_hz < fmax_hz)

        # build mask
        nbins = mag.shape[-2]
        bins_hz = torch.linspace(0, self.sample_rate / 2, nbins, device=self.device)
        bins_hz = bins_hz[None, None, :, None].repeat(
            self.batch_size, 1, 1, mag.shape[-1]
        )
        mask = (fmin_hz <= bins_hz) & (bins_hz < fmax_hz)
        mask = mask.to(self.device)

        mag = mag.masked_fill(mask, val)
        phase = phase.masked_fill(mask, val)
        self.stft_data = mag * torch.exp(1j * phase)
        return self

    def mask_timesteps(
        self,
        tmin_s: typing.Union[torch.Tensor, np.ndarray, float],
        tmax_s: typing.Union[torch.Tensor, np.ndarray, float],
        val: float = 0.0,
    ):
        """Masks timesteps between ``tmin_s`` and ``tmax_s``, and fills them
        with the value specified by ``val``. Useful for implementing SpecAug.
        The min and max can be different for every item in the batch.

        Parameters
        ----------
        tmin_s : typing.Union[torch.Tensor, np.ndarray, float]
            Lower end of timesteps to mask out.
        tmax_s : typing.Union[torch.Tensor, np.ndarray, float]
            Upper end of timesteps to mask out.
        val : float, optional
            Value to fill in, by default 0.0

        Returns
        -------
        AudioSignal
            Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
            masked audio data.
        """
        # SpecAug
        mag, phase = self.magnitude, self.phase
        tmin_s = util.ensure_tensor(tmin_s, ndim=mag.ndim)
        tmax_s = util.ensure_tensor(tmax_s, ndim=mag.ndim)

        assert torch.all(tmin_s < tmax_s)

        # build mask
        nt = mag.shape[-1]
        bins_t = torch.linspace(0, self.signal_duration, nt, device=self.device)
        bins_t = bins_t[None, None, None, :].repeat(
            self.batch_size, 1, mag.shape[-2], 1
        )
        mask = (tmin_s <= bins_t) & (bins_t < tmax_s)

        mag = mag.masked_fill(mask, val)
        phase = phase.masked_fill(mask, val)
        self.stft_data = mag * torch.exp(1j * phase)
        return self

    def mask_low_magnitudes(
        self, db_cutoff: typing.Union[torch.Tensor, np.ndarray, float], val: float = 0.0
    ):
        """Mask away magnitudes below a specified threshold, which
        can be different for every item in the batch.

        Parameters
        ----------
        db_cutoff : typing.Union[torch.Tensor, np.ndarray, float]
            Decibel value for which things below it will be masked away.
        val : float, optional
            Value to fill in for masked portions, by default 0.0

        Returns
        -------
        AudioSignal
            Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
            masked audio data.
        """
        mag = self.magnitude
        log_mag = self.log_magnitude()

        db_cutoff = util.ensure_tensor(db_cutoff, ndim=mag.ndim)
        mask = log_mag < db_cutoff
        mag = mag.masked_fill(mask, val)

        self.magnitude = mag
        return self

    def shift_phase(self, shift: typing.Union[torch.Tensor, np.ndarray, float]):
        """Shifts the phase by a constant value.

        Parameters
        ----------
        shift : typing.Union[torch.Tensor, np.ndarray, float]
            What to shift the phase by.

        Returns
        -------
        AudioSignal
            Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
            masked audio data.
        """
        shift = util.ensure_tensor(shift, ndim=self.phase.ndim)
        self.phase = self.phase + shift
        return self

    def corrupt_phase(self, scale: typing.Union[torch.Tensor, np.ndarray, float]):
        """Corrupts the phase randomly by some scaled value.

        Parameters
        ----------
        scale : typing.Union[torch.Tensor, np.ndarray, float]
            Standard deviation of noise to add to the phase.

        Returns
        -------
        AudioSignal
            Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
            masked audio data.
        """
        scale = util.ensure_tensor(scale, ndim=self.phase.ndim)
        self.phase = self.phase + scale * torch.randn_like(self.phase)
        return self

    def preemphasis(self, coef: float = 0.85):
        """Applies pre-emphasis to audio signal.

        Parameters
        ----------
        coef : float, optional
            How much pre-emphasis to apply, lower values do less. 0 does nothing.
            by default 0.85

        Returns
        -------
        AudioSignal
            Pre-emphasized signal.
        """
        kernel = torch.tensor([1, -coef, 0]).view(1, 1, -1).to(self.device)
        x = self.audio_data.reshape(-1, 1, self.signal_length)
        x = torch.nn.functional.conv1d(x, kernel, padding=1)
        self.audio_data = x.reshape(*self.audio_data.shape)
        return self

class DisplayMixin:
    @format_figure
    def specshow(
        self,
        preemphasis: bool = False,
        x_axis: str = "time",
        y_axis: str = "linear",
        n_mels: int = 128,
        **kwargs,
    ):
        """Displays a spectrogram, using ``librosa.display.specshow``.

        Parameters
        ----------
        preemphasis : bool, optional
            Whether or not to apply preemphasis, which makes high
            frequency detail easier to see, by default False
        x_axis : str, optional
            How to label the x axis, by default "time"
        y_axis : str, optional
            How to label the y axis, by default "linear"
        n_mels : int, optional
            If displaying a mel spectrogram with ``y_axis = "mel"``,
            this controls the number of mels, by default 128.
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.util.format_figure`.
        """
        import librosa
        import librosa.display

        # Always re-compute the STFT data before showing it, in case
        # it changed.
        signal = self.clone()
        signal.stft_data = None

        if preemphasis:
            signal.preemphasis()

        ref = signal.magnitude.max()
        log_mag = signal.log_magnitude(ref_value=ref)

        if y_axis == "mel":
            log_mag = 20 * signal.mel_spectrogram(n_mels).clamp(1e-5).log10()
            log_mag -= log_mag.max()

        librosa.display.specshow(
            log_mag.numpy()[0].mean(axis=0),
            x_axis=x_axis,
            y_axis=y_axis,
            sr=signal.sample_rate,
            **kwargs,
        )

    @format_figure
    def waveplot(self, x_axis: str = "time", **kwargs):
        """Displays a waveform plot, using ``librosa.display.waveshow``.

        Parameters
        ----------
        x_axis : str, optional
            How to label the x axis, by default "time"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.util.format_figure`.
        """
        import librosa
        import librosa.display

        audio_data = self.audio_data[0].mean(dim=0)
        audio_data = audio_data.cpu().numpy()

        plot_fn = "waveshow" if hasattr(librosa.display, "waveshow") else "waveplot"
        wave_plot_fn = getattr(librosa.display, plot_fn)
        wave_plot_fn(audio_data, x_axis=x_axis, sr=self.sample_rate, **kwargs)

    @format_figure
    def wavespec(self, x_axis: str = "time", **kwargs):
        """Displays a waveform plot, using ``librosa.display.waveshow``.

        Parameters
        ----------
        x_axis : str, optional
            How to label the x axis, by default "time"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.display.DisplayMixin.specshow`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(6, 1)
        plt.subplot(gs[0, :])
        self.waveplot(x_axis=x_axis)
        plt.subplot(gs[1:, :])
        self.specshow(x_axis=x_axis, **kwargs)

    def write_audio_to_tb(
        self,
        tag: str,
        writer,
        step: int = None,
        plot_fn: typing.Union[typing.Callable, str] = "specshow",
        **kwargs,
    ):
        """Writes a signal and its spectrogram to Tensorboard. Will show up
        under the Audio and Images tab in Tensorboard.

        Parameters
        ----------
        tag : str
            Tag to write signal to (e.g. ``clean/sample_0.wav``). The image will be
            written to the corresponding ``.png`` file (e.g. ``clean/sample_0.png``).
        writer : SummaryWriter
            A SummaryWriter object from PyTorch library.
        step : int, optional
            The step to write the signal to, by default None
        plot_fn : typing.Union[typing.Callable, str], optional
            How to create the image. Set to ``None`` to avoid plotting, by default "specshow"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.display.DisplayMixin.specshow` or
            whatever ``plot_fn`` is set to.
        """
        import matplotlib.pyplot as plt

        audio_data = self.audio_data[0, 0].detach().cpu()
        sample_rate = self.sample_rate
        writer.add_audio(tag, audio_data, step, sample_rate)

        if plot_fn is not None:
            if isinstance(plot_fn, str):
                plot_fn = getattr(self, plot_fn)
            fig = plt.figure()
            plt.clf()
            plot_fn(**kwargs)
            writer.add_figure(tag.replace("wav", "png"), fig, step)

    def save_image(
        self,
        image_path: str,
        plot_fn: typing.Union[typing.Callable, str] = "specshow",
        **kwargs,
    ):
        """Save AudioSignal spectrogram (or whatever ``plot_fn`` is set to) to
        a specified file.

        Parameters
        ----------
        image_path : str
            Where to save the file to.
        plot_fn : typing.Union[typing.Callable, str], optional
            How to create the image. Set to ``None`` to avoid plotting, by default "specshow"
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.display.DisplayMixin.specshow` or
            whatever ``plot_fn`` is set to.
        """
        import matplotlib.pyplot as plt

        if isinstance(plot_fn, str):
            plot_fn = getattr(self, plot_fn)

        plt.clf()
        plot_fn(**kwargs)
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()

def r128stats(filepath: str, quiet: bool):
    """Takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter.

    Parameters
    ----------
    filepath : str
        Path to compute loudness stats on.
    quiet : bool
        Whether to show FFMPEG output during computation.

    Returns
    -------
    dict
        Dictionary containing loudness stats.
    """
    ffargs = [
        "ffmpeg",
        "-nostats",
        "-i",
        filepath,
        "-filter_complex",
        "ebur128",
        "-f",
        "null",
        "-",
    ]
    if quiet:
        ffargs += ["-hide_banner"]
    proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE, universal_newlines=True)
    stats = proc.communicate()[1]
    summary_index = stats.rfind("Summary:")

    summary_list = stats[summary_index:].split()
    i_lufs = float(summary_list[summary_list.index("I:") + 1])
    i_thresh = float(summary_list[summary_list.index("I:") + 4])
    lra = float(summary_list[summary_list.index("LRA:") + 1])
    lra_thresh = float(summary_list[summary_list.index("LRA:") + 4])
    lra_low = float(summary_list[summary_list.index("low:") + 1])
    lra_high = float(summary_list[summary_list.index("high:") + 1])
    stats_dict = {
        "I": i_lufs,
        "I Threshold": i_thresh,
        "LRA": lra,
        "LRA Threshold": lra_thresh,
        "LRA Low": lra_low,
        "LRA High": lra_high,
    }

    return stats_dict

def ffprobe_offset(path):
    ff = ffmpy.FFprobe(
        inputs={path: None},
        global_options="-show_entries format=start_time:stream=duration,start_time,codec_type,start_pts,time_base -of json -v quiet",
    )
    streams = json.loads(ff.run(stdout=subprocess.PIPE)[0])["streams"]
    seconds_offset = 0.0
    # Get the offset of the first audio stream we find
    # and return its start time, if it has one.
    for stream in streams:
        if stream["codec_type"] == "audio":
            seconds_offset = stream.get("start_time", 0.0)
            break
    return float(seconds_offset)

class FFMPEGMixin:
    _loudness = None

    def ffmpeg_loudness(self, quiet: bool = True):
        """Computes loudness of audio file using FFMPEG.

        Parameters
        ----------
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        torch.Tensor
            Loudness of every item in the batch, computed via
            FFMPEG.
        """
        loudness = []

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            for i in range(self.batch_size):
                self[i].write(f.name)
                loudness_stats = r128stats(f.name, quiet=quiet)
                loudness.append(loudness_stats["I"])

        self._loudness = torch.from_numpy(np.array(loudness)).float()
        return self.loudness()

    def ffmpeg_resample(self, sample_rate: int, quiet: bool = True):
        """Resamples AudioSignal using FFMPEG. More memory-efficient
        than using julius.resample for long audio files.

        Parameters
        ----------
        sample_rate : int
            Sample rate to resample to.
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        AudioSignal
            Resampled AudioSignal.
        """
        from audiotools import AudioSignal

        if sample_rate == self.sample_rate:
            return self

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.write(f.name)
            f_out = f.name.replace("wav", "rs.wav")
            command = f"ffmpeg -i {f.name} -ar {sample_rate} {f_out}"
            if quiet:
                command += " -hide_banner -loglevel error"
            subprocess.check_call(shlex.split(command))
            resampled = AudioSignal(f_out)
            Path.unlink(Path(f_out))
        return resampled

    @classmethod
    def load_from_file_with_ffmpeg(cls, audio_path: str, quiet: bool = True, **kwargs):
        """Loads AudioSignal object after decoding it to a wav file using FFMPEG.
        Useful for loading audio that isn't covered by librosa's loading mechanism. Also
        useful for loading mp3 files, without any offset.

        Parameters
        ----------
        audio_path : str
            Path to load AudioSignal from.
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        AudioSignal
            AudioSignal loaded from file with FFMPEG.
        """
        audio_path = str(audio_path)
        with tempfile.TemporaryDirectory() as d:
            wav_file = str(Path(d) / "extracted.wav")
            padded_wav = str(Path(d) / "padded.wav")

            global_options = "-y"
            if quiet:
                global_options += " -loglevel error"

            ff = ffmpy.FFmpeg(
                inputs={audio_path: None},
                outputs={wav_file: None},
                global_options=global_options,
            )
            ff.run()

            # We pad the file using the start time offset
            # in case it's an audio stream starting at some
            # offset in a video container.
            pad = ffprobe_offset(audio_path)
            # Don't pad files with discrepancies less than
            # 0.027s - it's likely due to codec latency.
            # The amount of latency introduced by mp3 is
            # 1152, which is 0.0261 44khz. So we
            # set the threshold here slightly above that.
            # Source: https://lame.sourceforge.io/tech-FAQ.txt.
            if pad < 0.027:
                pad = 0.0
            ff = ffmpy.FFmpeg(
                inputs={wav_file: None},
                outputs={padded_wav: f"-af 'adelay={pad*1000}:all=true'"},
                global_options=global_options,
            )
            ff.run()

            signal = cls(padded_wav, **kwargs)

        return signal

class WhisperMixin:
    is_initialized = False

    def setup_whisper(
        self,
        pretrained_model_name_or_path: str = "openai/whisper-base.en",
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        from transformers import WhisperForConditionalGeneration
        from transformers import WhisperProcessor

        self.whisper_device = device
        self.whisper_processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path
        ).to(self.whisper_device)
        self.is_initialized = True

    def get_whisper_features(self) -> torch.Tensor:
        """Preprocess audio signal as per the whisper model's training config.

        Returns
        -------
        torch.Tensor
            The prepinput features of the audio signal. Shape: (1, channels, seq_len)
        """
        import torch

        if not self.is_initialized:
            self.setup_whisper()

        signal = self.to(self.device)
        raw_speech = list(
            (
                signal.clone()
                .resample(self.whisper_processor.feature_extractor.sampling_rate)
                .audio_data[:, 0, :]
                .numpy()
            )
        )

        with torch.inference_mode():
            input_features = self.whisper_processor(
                raw_speech,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).input_features

        return input_features

    def get_whisper_transcript(self) -> str:
        """Get the transcript of the audio signal using the whisper model.

        Returns
        -------
        str
            The transcript of the audio signal, including special tokens such as <|startoftranscript|> and <|endoftext|>.
        """

        if not self.is_initialized:
            self.setup_whisper()

        input_features = self.get_whisper_features()

        with torch.inference_mode():
            input_features = input_features.to(self.whisper_device)
            generated_ids = self.whisper_model.generate(inputs=input_features)

        transcription = self.whisper_processor.batch_decode(generated_ids)
        return transcription[0]

    def get_whisper_embeddings(self) -> torch.Tensor:
        """Get the last hidden state embeddings of the audio signal using the whisper model.

        Returns
        -------
        torch.Tensor
            The Whisper embeddings of the audio signal. Shape: (1, seq_len, hidden_size)
        """
        import torch

        if not self.is_initialized:
            self.setup_whisper()

        input_features = self.get_whisper_features()
        encoder = self.whisper_model.get_encoder()

        with torch.inference_mode():
            input_features = input_features.to(self.whisper_device)
            embeddings = encoder(input_features)

        return embeddings.last_hidden_state

def random_state(seed: typing.Union[int, np.random.RandomState]):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : typing.Union[int, np.random.RandomState] or None
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState
        Random state object.

    Raises
    ------
    ValueError
        If seed is not valid, an error is thrown.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
        )

# @dataclass
class Info:
    """Shim for torchaudio.info API changes."""

    sample_rate: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate

def ensure_tensor(
    x: typing.Union[np.ndarray, torch.Tensor, float, int],
    ndim: int = None,
    batch_size: int = None,
):
    """Ensures that the input ``x`` is a tensor of specified
    dimensions and batch size.

    Parameters
    ----------
    x : typing.Union[np.ndarray, torch.Tensor, float, int]
        Data that will become a tensor on its way out.
    ndim : int, optional
        How many dimensions should be in the output, by default None
    batch_size : int, optional
        The batch size of the output, by default None

    Returns
    -------
    torch.Tensor
        Modified version of ``x`` as a tensor.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if ndim is not None:
        assert x.ndim <= ndim
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
    if batch_size is not None:
        if x.shape[0] != batch_size:
            shape = list(x.shape)
            shape[0] = batch_size
            x = x.expand(*shape)
    return x

def _get_value(other):
    from . import AudioSignal

    if isinstance(other, AudioSignal):
        return other.audio_data
    return other


class AudioSignal(
    EffectMixin,
    LoudnessMixin,
    PlayMixin,
    ImpulseResponseMixin,
    DSPMixin,
    DisplayMixin,
    FFMPEGMixin,
    WhisperMixin,
):
    """This is the core object of this library. Audio is always
    loaded into an AudioSignal, which then enables all the features
    of this library, including audio augmentations, I/O, playback,
    and more.

    The structure of this object is that the base functionality
    is defined in ``core/audio_signal.py``, while extensions to
    that functionality are defined in the other ``core/*.py``
    files. For example, all the display-based functionality
    (e.g. plot spectrograms, waveforms, write to tensorboard)
    are in ``core/display.py``.

    Parameters
    ----------
    audio_path_or_array : typing.Union[torch.Tensor, str, Path, np.ndarray]
        Object to create AudioSignal from. Can be a tensor, numpy array,
        or a path to a file. The file is always reshaped to
    sample_rate : int, optional
        Sample rate of the audio. If different from underlying file, resampling is
        performed. If passing in an array or tensor, this must be defined,
        by default None
    stft_params : STFTParams, optional
        Parameters of STFT to use. , by default None
    offset : float, optional
        Offset in seconds to read from file, by default 0
    duration : float, optional
        Duration in seconds to read from file, by default None
    device : str, optional
        Device to load audio onto, by default None

    Examples
    --------
    Loading an AudioSignal from an array, at a sample rate of
    44100.

    >>> signal = AudioSignal(torch.randn(5*44100), 44100)

    Note, the signal is reshaped to have a batch size, and one
    audio channel:

    >>> print(signal.shape)
    (1, 1, 44100)

    You can treat AudioSignals like tensors, and many of the same
    functions you might use on tensors are defined for AudioSignals
    as well:

    >>> signal.to("cuda")
    >>> signal.cuda()
    >>> signal.clone()
    >>> signal.detach()

    Indexing AudioSignals returns an AudioSignal:

    >>> signal[..., 3*44100:4*44100]

    The above signal is 1 second long, and is also an AudioSignal.
    """

    def __init__(
        self,
        audio_path_or_array: typing.Union[torch.Tensor, str, Path, np.ndarray],
        sample_rate: int = None,
        stft_params: STFTParams = None,
        offset: float = 0,
        duration: float = None,
        device: str = None,
    ):
        audio_path = None
        audio_array = None

        if isinstance(audio_path_or_array, str):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, pathlib.Path):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, np.ndarray):
            audio_array = audio_path_or_array
        elif torch.is_tensor(audio_path_or_array):
            audio_array = audio_path_or_array
        else:
            raise ValueError(
                "audio_path_or_array must be either a Path, "
                "string, numpy array, or torch Tensor!"
            )

        self.path_to_file = None

        self.audio_data = None
        self.sources = None  # List of AudioSignal objects.
        self.stft_data = None
        if audio_path is not None:
            self.load_from_file(
                audio_path, offset=offset, duration=duration, device=device
            )
        elif audio_array is not None:
            assert sample_rate is not None, "Must set sample rate!"
            self.load_from_array(audio_array, sample_rate, device=device)

        self.window = None
        self.stft_params = stft_params

        self.metadata = {
            "offset": offset,
            "duration": duration,
        }

    @property
    def path_to_input_file(
        self,
    ):
        """
        Path to input file, if it exists.
        Alias to ``path_to_file`` for backwards compatibility
        """
        return self.path_to_file

    @classmethod
    def excerpt(
        cls,
        audio_path: typing.Union[str, Path],
        offset: float = None,
        duration: float = None,
        state: typing.Union[np.random.RandomState, int] = None,
        **kwargs,
    ):
        """Randomly draw an excerpt of ``duration`` seconds from an
        audio file specified at ``audio_path``, between ``offset`` seconds
        and end of file. ``state`` can be used to seed the random draw.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to audio file to grab excerpt from.
        offset : float, optional
            Lower bound for the start time, in seconds drawn from
            the file, by default None.
        duration : float, optional
            Duration of excerpt, in seconds, by default None
        state : typing.Union[np.random.RandomState, int], optional
            RandomState or seed of random state, by default None

        Returns
        -------
        AudioSignal
            AudioSignal containing excerpt.

        Examples
        --------
        >>> signal = AudioSignal.excerpt("path/to/audio", duration=5)
        """
        info = info(audio_path)
        total_duration = info.duration

        state = random_state(state)
        lower_bound = 0 if offset is None else offset
        upper_bound = max(total_duration - duration, 0)
        offset = state.uniform(lower_bound, upper_bound)

        signal = cls(audio_path, offset=offset, duration=duration, **kwargs)
        signal.metadata["offset"] = offset
        signal.metadata["duration"] = duration

        return signal

    @classmethod
    def salient_excerpt(
        cls,
        audio_path: typing.Union[str, Path],
        loudness_cutoff: float = None,
        num_tries: int = 8,
        state: typing.Union[np.random.RandomState, int] = None,
        **kwargs,
    ):
        """Similar to AudioSignal.excerpt, except it extracts excerpts only
        if they are above a specified loudness threshold, which is computed via
        a fast LUFS routine.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to audio file to grab excerpt from.
        loudness_cutoff : float, optional
            Loudness threshold in dB. Typical values are ``-40, -60``,
            etc, by default None
        num_tries : int, optional
            Number of tries to grab an excerpt above the threshold
            before giving up, by default 8.
        state : typing.Union[np.random.RandomState, int], optional
            RandomState or seed of random state, by default None
        kwargs : dict
            Keyword arguments to AudioSignal.excerpt

        Returns
        -------
        AudioSignal
            AudioSignal containing excerpt.


        .. warning::
            if ``num_tries`` is set to None, ``salient_excerpt`` may try forever, which can
            result in an infinite loop if ``audio_path`` does not have
            any loud enough excerpts.

        Examples
        --------
        >>> signal = AudioSignal.salient_excerpt(
                "path/to/audio",
                loudness_cutoff=-40,
                duration=5
            )
        """
        state = random_state(state)
        if loudness_cutoff is None:
            excerpt = cls.excerpt(audio_path, state=state, **kwargs)
        else:
            loudness = -np.inf
            num_try = 0
            while loudness <= loudness_cutoff:
                excerpt = cls.excerpt(audio_path, state=state, **kwargs)
                loudness = excerpt.loudness()
                num_try += 1
                if num_tries is not None and num_try >= num_tries:
                    break
        return excerpt

    @classmethod
    def zeros(
        cls,
        duration: float,
        sample_rate: int,
        num_channels: int = 1,
        batch_size: int = 1,
        **kwargs,
    ):
        """Helper function create an AudioSignal of all zeros.

        Parameters
        ----------
        duration : float
            Duration of AudioSignal
        sample_rate : int
            Sample rate of AudioSignal
        num_channels : int, optional
            Number of channels, by default 1
        batch_size : int, optional
            Batch size, by default 1

        Returns
        -------
        AudioSignal
            AudioSignal containing all zeros.

        Examples
        --------
        Generate 5 seconds of all zeros at a sample rate of 44100.

        >>> signal = AudioSignal.zeros(5.0, 44100)
        """
        n_samples = int(duration * sample_rate)
        return cls(
            torch.zeros(batch_size, num_channels, n_samples), sample_rate, **kwargs
        )

    @classmethod
    def wave(
        cls,
        frequency: float,
        duration: float,
        sample_rate: int,
        num_channels: int = 1,
        shape: str = "sine",
        **kwargs,
    ):
        """
        Generate a waveform of a given frequency and shape.

        Parameters
        ----------
        frequency : float
            Frequency of the waveform
        duration : float
            Duration of the waveform
        sample_rate : int
            Sample rate of the waveform
        num_channels : int, optional
            Number of channels, by default 1
        shape : str, optional
            Shape of the waveform, by default "saw"
            One of "sawtooth", "square", "sine", "triangle"
        kwargs : dict
            Keyword arguments to AudioSignal
        """
        n_samples = int(duration * sample_rate)
        t = torch.linspace(0, duration, n_samples)
        if shape == "sawtooth":
            from scipy.signal import sawtooth

            wave_data = sawtooth(2 * np.pi * frequency * t, 0.5)
        elif shape == "square":
            from scipy.signal import square

            wave_data = square(2 * np.pi * frequency * t)
        elif shape == "sine":
            wave_data = np.sin(2 * np.pi * frequency * t)
        elif shape == "triangle":
            from scipy.signal import sawtooth

            # frequency is doubled by the abs call, so omit the 2 in 2pi
            wave_data = sawtooth(np.pi * frequency * t, 0.5)
            wave_data = -np.abs(wave_data) * 2 + 1
        else:
            raise ValueError(f"Invalid shape {shape}")

        wave_data = torch.tensor(wave_data, dtype=torch.float32)
        wave_data = wave_data.unsqueeze(0).unsqueeze(0).repeat(1, num_channels, 1)
        return cls(wave_data, sample_rate, **kwargs)

    @classmethod
    def batch(
        cls,
        audio_signals: list,
        pad_signals: bool = False,
        truncate_signals: bool = False,
        resample: bool = False,
        dim: int = 0,
    ):
        """Creates a batched AudioSignal from a list of AudioSignals.

        Parameters
        ----------
        audio_signals : list[AudioSignal]
            List of AudioSignal objects
        pad_signals : bool, optional
            Whether to pad signals to length of the maximum length
            AudioSignal in the list, by default False
        truncate_signals : bool, optional
            Whether to truncate signals to length of shortest length
            AudioSignal in the list, by default False
        resample : bool, optional
            Whether to resample AudioSignal to the sample rate of
            the first AudioSignal in the list, by default False
        dim : int, optional
            Dimension along which to batch the signals.

        Returns
        -------
        AudioSignal
            Batched AudioSignal.

        Raises
        ------
        RuntimeError
            If not all AudioSignals are the same sample rate, and
            ``resample=False``, an error is raised.
        RuntimeError
            If not all AudioSignals are the same the length, and
            both ``pad_signals=False`` and ``truncate_signals=False``,
            an error is raised.

        Examples
        --------
        Batching a bunch of random signals:

        >>> signal_list = [AudioSignal(torch.randn(44100), 44100) for _ in range(10)]
        >>> signal = AudioSignal.batch(signal_list)
        >>> print(signal.shape)
        (10, 1, 44100)

        """
        signal_lengths = [x.signal_length for x in audio_signals]
        sample_rates = [x.sample_rate for x in audio_signals]

        if len(set(sample_rates)) != 1:
            if resample:
                for x in audio_signals:
                    x.resample(sample_rates[0])
            else:
                raise RuntimeError(
                    f"Not all signals had the same sample rate! Got {sample_rates}. "
                    f"All signals must have the same sample rate, or resample must be True. "
                )

        if len(set(signal_lengths)) != 1:
            if pad_signals:
                max_length = max(signal_lengths)
                for x in audio_signals:
                    pad_len = max_length - x.signal_length
                    x.zero_pad(0, pad_len)
            elif truncate_signals:
                min_length = min(signal_lengths)
                for x in audio_signals:
                    x.truncate_samples(min_length)
            else:
                raise RuntimeError(
                    f"Not all signals had the same length! Got {signal_lengths}. "
                    f"All signals must be the same length, or pad_signals/truncate_signals "
                    f"must be True. "
                )
        # Concatenate along the specified dimension (default 0)
        audio_data = torch.cat([x.audio_data for x in audio_signals], dim=dim)
        audio_paths = [x.path_to_file for x in audio_signals]

        batched_signal = cls(
            audio_data,
            sample_rate=audio_signals[0].sample_rate,
        )
        batched_signal.path_to_file = audio_paths
        return batched_signal

    # I/O
    def load_from_file(
        self,
        audio_path: typing.Union[str, Path],
        offset: float,
        duration: float,
        device: str = "cpu",
    ):
        """Loads data from file. Used internally when AudioSignal
        is instantiated with a path to a file.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to file
        offset : float
            Offset in seconds
        duration : float
            Duration in seconds
        device : str, optional
            Device to put AudioSignal on, by default "cpu"

        Returns
        -------
        AudioSignal
            AudioSignal loaded from file
        """
        import librosa

        data, sample_rate = librosa.load(
            audio_path,
            offset=offset,
            duration=duration,
            sr=None,
            mono=False,
        )
        data = ensure_tensor(data)
        if data.shape[-1] == 0:
            raise RuntimeError(
                f"Audio file {audio_path} with offset {offset} and duration {duration} is empty!"
            )

        if data.ndim < 2:
            data = data.unsqueeze(0)
        if data.ndim < 3:
            data = data.unsqueeze(0)
        self.audio_data = data

        self.original_signal_length = self.signal_length

        self.sample_rate = sample_rate
        self.path_to_file = audio_path
        return self.to(device)

    def load_from_array(
        self,
        audio_array: typing.Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        device: str = "cpu",
    ):
        """Loads data from array, reshaping it to be exactly 3
        dimensions. Used internally when AudioSignal is called
        with a tensor or an array.

        Parameters
        ----------
        audio_array : typing.Union[torch.Tensor, np.ndarray]
            Array/tensor of audio of samples.
        sample_rate : int
            Sample rate of audio
        device : str, optional
            Device to move audio onto, by default "cpu"

        Returns
        -------
        AudioSignal
            AudioSignal loaded from array
        """
        audio_data = ensure_tensor(audio_array)

        if audio_data.dtype == torch.double:
            audio_data = audio_data.float()

        if audio_data.ndim < 2:
            audio_data = audio_data.unsqueeze(0)
        if audio_data.ndim < 3:
            audio_data = audio_data.unsqueeze(0)
        self.audio_data = audio_data

        self.original_signal_length = self.signal_length

        self.sample_rate = sample_rate
        return self.to(device)

    def write(self, audio_path: typing.Union[str, Path]):
        """Writes audio to a file. Only writes the audio
        that is in the very first item of the batch. To write other items
        in the batch, index the signal along the batch dimension
        before writing. After writing, the signal's ``path_to_file``
        attribute is updated to the new path.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to write audio to.

        Returns
        -------
        AudioSignal
            Returns original AudioSignal, so you can use this in a fluent
            interface.

        Examples
        --------
        Creating and writing a signal to disk:

        >>> signal = AudioSignal(torch.randn(10, 1, 44100), 44100)
        >>> signal.write("/tmp/out.wav")

        Writing a different element of the batch:

        >>> signal[5].write("/tmp/out.wav")

        Using this in a fluent interface:

        >>> signal.write("/tmp/original.wav").low_pass(4000).write("/tmp/lowpass.wav")

        """
        if self.audio_data[0].abs().max() > 1:
            warnings.warn("Audio amplitude > 1 clipped when saving")
        soundfile.write(str(audio_path), self.audio_data[0].numpy().T, self.sample_rate)

        self.path_to_file = audio_path
        return self

    def deepcopy(self):
        """Copies the signal and all of its attributes.

        Returns
        -------
        AudioSignal
            Deep copy of the audio signal.
        """
        return copy.deepcopy(self)

    def copy(self):
        """Shallow copy of signal.

        Returns
        -------
        AudioSignal
            Shallow copy of the audio signal.
        """
        return copy.copy(self)

    def clone(self):
        """Clones all tensors contained in the AudioSignal,
        and returns a copy of the signal with everything
        cloned. Useful when using AudioSignal within autograd
        computation graphs.

        Relevant attributes are the stft data, the audio data,
        and the loudness of the file.

        Returns
        -------
        AudioSignal
            Clone of AudioSignal.
        """
        clone = type(self)(
            self.audio_data.clone(),
            self.sample_rate,
            stft_params=self.stft_params,
        )
        if self.stft_data is not None:
            clone.stft_data = self.stft_data.clone()
        if self._loudness is not None:
            clone._loudness = self._loudness.clone()
        clone.path_to_file = copy.deepcopy(self.path_to_file)
        clone.metadata = copy.deepcopy(self.metadata)
        return clone

    def detach(self):
        """Detaches tensors contained in AudioSignal.

        Relevant attributes are the stft data, the audio data,
        and the loudness of the file.

        Returns
        -------
        AudioSignal
            Same signal, but with all tensors detached.
        """
        if self._loudness is not None:
            self._loudness = self._loudness.detach()
        if self.stft_data is not None:
            self.stft_data = self.stft_data.detach()

        self.audio_data = self.audio_data.detach()
        return self

    def hash(self):
        """Writes the audio data to a temporary file, and then
        hashes it using hashlib. Useful for creating a file
        name based on the audio content.

        Returns
        -------
        str
            Hash of audio data.

        Examples
        --------
        Creating a signal, and writing it to a unique file name:

        >>> signal = AudioSignal(torch.randn(44100), 44100)
        >>> hash = signal.hash()
        >>> signal.write(f"{hash}.wav")

        """
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.write(f.name)
            h = hashlib.sha256()
            b = bytearray(128 * 1024)
            mv = memoryview(b)
            with open(f.name, "rb", buffering=0) as f:
                for n in iter(lambda: f.readinto(mv), 0):
                    h.update(mv[:n])
            file_hash = h.hexdigest()
        return file_hash

    # Signal operations
    def to_mono(self):
        """Converts audio data to mono audio, by taking the mean
        along the channels dimension.

        Returns
        -------
        AudioSignal
            AudioSignal with mean of channels.
        """
        self.audio_data = self.audio_data.mean(1, keepdim=True)
        return self

    def resample(self, sample_rate: int):
        """Resamples the audio, using sinc interpolation. This works on both
        cpu and gpu, and is much faster on gpu.

        Parameters
        ----------
        sample_rate : int
            Sample rate to resample to.

        Returns
        -------
        AudioSignal
            Resampled AudioSignal
        """
        if sample_rate == self.sample_rate:
            return self
        self.audio_data = julius.resample_frac(
            self.audio_data, self.sample_rate, sample_rate
        )
        self.sample_rate = sample_rate
        return self

    # Tensor operations
    def to(self, device: str):
        """Moves all tensors contained in signal to the specified device.

        Parameters
        ----------
        device : str
            Device to move AudioSignal onto. Typical values are
            "cuda", "cpu", or "cuda:n" to specify the nth gpu.

        Returns
        -------
        AudioSignal
            AudioSignal with all tensors moved to specified device.
        """
        if self._loudness is not None:
            self._loudness = self._loudness.to(device)
        if self.stft_data is not None:
            self.stft_data = self.stft_data.to(device)
        if self.audio_data is not None:
            self.audio_data = self.audio_data.to(device)
        return self

    def float(self):
        """Calls ``.float()`` on ``self.audio_data``.

        Returns
        -------
        AudioSignal
        """
        self.audio_data = self.audio_data.float()
        return self

    def cpu(self):
        """Moves AudioSignal to cpu.

        Returns
        -------
        AudioSignal
        """
        return self.to("cpu")

    def cuda(self):  # pragma: no cover
        """Moves AudioSignal to cuda.

        Returns
        -------
        AudioSignal
        """
        return self.to("cuda")

    def numpy(self):
        """Detaches ``self.audio_data``, moves to cpu, and converts to numpy.

        Returns
        -------
        np.ndarray
            Audio data as a numpy array.
        """
        return self.audio_data.detach().cpu().numpy()

    def zero_pad(self, before: int, after: int):
        """Zero pads the audio_data tensor before and after.

        Parameters
        ----------
        before : int
            How many zeros to prepend to audio.
        after : int
            How many zeros to append to audio.

        Returns
        -------
        AudioSignal
            AudioSignal with padding applied.
        """
        self.audio_data = torch.nn.functional.pad(self.audio_data, (before, after))
        return self

    def zero_pad_to(self, length: int, mode: str = "after"):
        """Pad with zeros to a specified length, either before or after
        the audio data.

        Parameters
        ----------
        length : int
            Length to pad to
        mode : str, optional
            Whether to prepend or append zeros to signal, by default "after"

        Returns
        -------
        AudioSignal
            AudioSignal with padding applied.
        """
        if mode == "before":
            self.zero_pad(max(length - self.signal_length, 0), 0)
        elif mode == "after":
            self.zero_pad(0, max(length - self.signal_length, 0))
        return self

    def trim(self, before: int, after: int):
        """Trims the audio_data tensor before and after.

        Parameters
        ----------
        before : int
            How many samples to trim from beginning.
        after : int
            How many samples to trim from end.

        Returns
        -------
        AudioSignal
            AudioSignal with trimming applied.
        """
        if after == 0:
            self.audio_data = self.audio_data[..., before:]
        else:
            self.audio_data = self.audio_data[..., before:-after]
        return self

    def truncate_samples(self, length_in_samples: int):
        """Truncate signal to specified length.

        Parameters
        ----------
        length_in_samples : int
            Truncate to this many samples.

        Returns
        -------
        AudioSignal
            AudioSignal with truncation applied.
        """
        self.audio_data = self.audio_data[..., :length_in_samples]
        return self

    @property
    def device(self):
        """Get device that AudioSignal is on.

        Returns
        -------
        torch.device
            Device that AudioSignal is on.
        """
        if self.audio_data is not None:
            device = self.audio_data.device
        elif self.stft_data is not None:
            device = self.stft_data.device
        return device

    # Properties
    @property
    def audio_data(self):
        """Returns the audio data tensor in the object.

        Audio data is always of the shape
        (batch_size, num_channels, num_samples). If value has less
        than 3 dims (e.g. is (num_channels, num_samples)), then it will
        be reshaped to (1, num_channels, num_samples) - a batch size of 1.

        Parameters
        ----------
        data : typing.Union[torch.Tensor, np.ndarray]
            Audio data to set.

        Returns
        -------
        torch.Tensor
            Audio samples.
        """
        return self._audio_data

    @audio_data.setter
    def audio_data(self, data: typing.Union[torch.Tensor, np.ndarray]):
        if data is not None:
            assert torch.is_tensor(data), "audio_data should be torch.Tensor"
            assert data.ndim == 3, "audio_data should be 3-dim (B, C, T)"
        self._audio_data = data
        # Old loudness value not guaranteed to be right, reset it.
        self._loudness = None
        return

    # alias for audio_data
    samples = audio_data

    @property
    def stft_data(self):
        """Returns the STFT data inside the signal. Shape is
        (batch, channels, frequencies, time).

        Returns
        -------
        torch.Tensor
            Complex spectrogram data.
        """
        return self._stft_data

    @stft_data.setter
    def stft_data(self, data: typing.Union[torch.Tensor, np.ndarray]):
        if data is not None:
            assert torch.is_tensor(data) and torch.is_complex(data)
            if self.stft_data is not None and self.stft_data.shape != data.shape:
                warnings.warn("stft_data changed shape")
        self._stft_data = data
        return

    @property
    def batch_size(self):
        """Batch size of audio signal.

        Returns
        -------
        int
            Batch size of signal.
        """
        return self.audio_data.shape[0]

    @property
    def signal_length(self):
        """Length of audio signal.

        Returns
        -------
        int
            Length of signal in samples.
        """
        return self.audio_data.shape[-1]

    # alias for signal_length
    length = signal_length

    @property
    def shape(self):
        """Shape of audio data.

        Returns
        -------
        tuple
            Shape of audio data.
        """
        return self.audio_data.shape

    @property
    def signal_duration(self):
        """Length of audio signal in seconds.

        Returns
        -------
        float
            Length of signal in seconds.
        """
        return self.signal_length / self.sample_rate

    # alias for signal_duration
    duration = signal_duration

    @property
    def num_channels(self):
        """Number of audio channels.

        Returns
        -------
        int
            Number of audio channels.
        """
        return self.audio_data.shape[1]

    # STFT
    @staticmethod
    @functools.lru_cache(None)
    def get_window(window_type: str, window_length: int, device: str):
        """Wrapper around scipy.signal.get_window so one can also get the
        popular sqrt-hann window. This function caches for efficiency
        using functools.lru\_cache.

        Parameters
        ----------
        window_type : str
            Type of window to get
        window_length : int
            Length of the window
        device : str
            Device to put window onto.

        Returns
        -------
        torch.Tensor
            Window returned by scipy.signal.get_window, as a tensor.
        """
        from scipy import signal

        if window_type == "average":
            window = np.ones(window_length) / window_length
        elif window_type == "sqrt_hann":
            window = np.sqrt(signal.get_window("hann", window_length))
        else:
            window = signal.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(device).float()
        return window

    @property
    def stft_params(self):
        """Returns STFTParams object, which can be re-used to other
        AudioSignals.

        This property can be set as well. If values are not defined in STFTParams,
        they are inferred automatically from the signal properties. The default is to use
        32ms windows, with 8ms hop length, and the square root of the hann window.

        Returns
        -------
        STFTParams
            STFT parameters for the AudioSignal.

        Examples
        --------
        >>> stft_params = STFTParams(128, 32)
        >>> signal1 = AudioSignal(torch.randn(44100), 44100, stft_params=stft_params)
        >>> signal2 = AudioSignal(torch.randn(44100), 44100, stft_params=signal1.stft_params)
        >>> signal1.stft_params = STFTParams() # Defaults
        """
        return self._stft_params

    @stft_params.setter
    def stft_params(self, value: STFTParams):
        default_win_len = int(2 ** (np.ceil(np.log2(0.032 * self.sample_rate))))
        default_hop_len = default_win_len // 4
        default_win_type = "hann"
        default_match_stride = False
        default_padding_type = "reflect"

        default_stft_params = STFTParams(
            window_length=default_win_len,
            hop_length=default_hop_len,
            window_type=default_win_type,
            match_stride=default_match_stride,
            padding_type=default_padding_type,
        )._asdict()

        value = value._asdict() if value else default_stft_params

        for key in default_stft_params:
            if value[key] is None:
                value[key] = default_stft_params[key]

        self._stft_params = STFTParams(**value)
        self.stft_data = None

    def compute_stft_padding(
        self, window_length: int, hop_length: int, match_stride: bool
    ):
        """Compute how the STFT should be padded, based on match\_stride.

        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_length : int
            Hop length of STFT.
        match_stride : bool
            Whether or not to match stride, making the STFT have the same alignment as
            convolutional layers.

        Returns
        -------
        tuple
            Amount to pad on either side of audio.
        """
        length = self.signal_length

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(length / hop_length) * hop_length - length
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        return right_pad, pad

    def stft(
        self,
        window_length: int = None,
        hop_length: int = None,
        window_type: str = None,
        match_stride: bool = None,
        padding_type: str = None,
    ):
        """Computes the short-time Fourier transform of the audio data,
        with specified STFT parameters.

        Parameters
        ----------
        window_length : int, optional
            Window length of STFT, by default ``0.032 * self.sample_rate``.
        hop_length : int, optional
            Hop length of STFT, by default ``window_length // 4``.
        window_type : str, optional
            Type of window to use, by default ``sqrt\_hann``.
        match_stride : bool, optional
            Whether to match the stride of convolutional layers, by default False
        padding_type : str, optional
            Type of padding to use, by default 'reflect'

        Returns
        -------
        torch.Tensor
            STFT of audio data.

        Examples
        --------
        Compute the STFT of an AudioSignal:

        >>> signal = AudioSignal(torch.randn(44100), 44100)
        >>> signal.stft()

        Vary the window and hop length:

        >>> stft_params = [STFTParams(128, 32), STFTParams(512, 128)]
        >>> for stft_param in stft_params:
        >>>     signal.stft_params = stft_params
        >>>     signal.stft()

        """
        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length if hop_length is None else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type if window_type is None else window_type
        )
        match_stride = (
            self.stft_params.match_stride if match_stride is None else match_stride
        )
        padding_type = (
            self.stft_params.padding_type if padding_type is None else padding_type
        )

        window = self.get_window(window_type, window_length, self.audio_data.device)
        window = window.to(self.audio_data.device)

        audio_data = self.audio_data
        right_pad, pad = self.compute_stft_padding(
            window_length, hop_length, match_stride
        )
        audio_data = torch.nn.functional.pad(
            audio_data, (pad, pad + right_pad), padding_type
        )
        stft_data = torch.stft(
            audio_data.reshape(-1, audio_data.shape[-1]),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft_data.shape
        stft_data = stft_data.reshape(self.batch_size, self.num_channels, nf, nt)

        if match_stride:
            # Drop first two and last two frames, which are added
            # because of padding. Now num_frames * hop_length = num_samples.
            stft_data = stft_data[..., 2:-2]
        self.stft_data = stft_data

        return stft_data

    def istft(
        self,
        window_length: int = None,
        hop_length: int = None,
        window_type: str = None,
        match_stride: bool = None,
        length: int = None,
    ):
        """Computes inverse STFT and sets it to audio\_data.

        Parameters
        ----------
        window_length : int, optional
            Window length of STFT, by default ``0.032 * self.sample_rate``.
        hop_length : int, optional
            Hop length of STFT, by default ``window_length // 4``.
        window_type : str, optional
            Type of window to use, by default ``sqrt\_hann``.
        match_stride : bool, optional
            Whether to match the stride of convolutional layers, by default False
        length : int, optional
            Original length of signal, by default None

        Returns
        -------
        AudioSignal
            AudioSignal with istft applied.

        Raises
        ------
        RuntimeError
            Raises an error if stft was not called prior to istft on the signal,
            or if stft_data is not set.
        """
        if self.stft_data is None:
            raise RuntimeError("Cannot do inverse STFT without self.stft_data!")

        window_length = (
            self.stft_params.window_length
            if window_length is None
            else int(window_length)
        )
        hop_length = (
            self.stft_params.hop_length if hop_length is None else int(hop_length)
        )
        window_type = (
            self.stft_params.window_type if window_type is None else window_type
        )
        match_stride = (
            self.stft_params.match_stride if match_stride is None else match_stride
        )

        window = self.get_window(window_type, window_length, self.stft_data.device)

        nb, nch, nf, nt = self.stft_data.shape
        stft_data = self.stft_data.reshape(nb * nch, nf, nt)
        right_pad, pad = self.compute_stft_padding(
            window_length, hop_length, match_stride
        )

        if length is None:
            length = self.original_signal_length
            length = length + 2 * pad + right_pad

        if match_stride:
            # Zero-pad the STFT on either side, putting back the frames that were
            # dropped in stft().
            stft_data = torch.nn.functional.pad(stft_data, (2, 2))

        audio_data = torch.istft(
            stft_data,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            length=length,
            center=True,
        )
        audio_data = audio_data.reshape(nb, nch, -1)
        if match_stride:
            audio_data = audio_data[..., pad : -(pad + right_pad)]
        self.audio_data = audio_data

        return self

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(
        sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None
    ):
        """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

        Parameters
        ----------
        sr : int
            Sample rate of audio
        n_fft : int
            Number of FFT bins
        n_mels : int
            Number of mels
        fmin : float, optional
            Lowest frequency, in Hz, by default 0.0
        fmax : float, optional
            Highest frequency, by default None

        Returns
        -------
        np.ndarray [shape=(n_mels, 1 + n_fft/2)]
            Mel transform matrix
        """
        from librosa.filters import mel as librosa_mel_fn

        return librosa_mel_fn(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

    def mel_spectrogram(
        self, n_mels: int = 80, mel_fmin: float = 0.0, mel_fmax: float = None, **kwargs
    ):
        """Computes a Mel spectrogram.

        Parameters
        ----------
        n_mels : int, optional
            Number of mels, by default 80
        mel_fmin : float, optional
            Lowest frequency, in Hz, by default 0.0
        mel_fmax : float, optional
            Highest frequency, by default None
        kwargs : dict, optional
            Keyword arguments to self.stft().

        Returns
        -------
        torch.Tensor [shape=(batch, channels, mels, time)]
            Mel spectrogram.
        """
        stft = self.stft(**kwargs)
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            sr=self.sample_rate,
            n_fft=2 * (nf - 1),
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).to(self.device)

        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)
        return mel_spectrogram

    @staticmethod
    @functools.lru_cache(None)
    def get_dct(n_mfcc: int, n_mels: int, norm: str = "ortho", device: str = None):
        """Create a discrete cosine transform (DCT) transformation matrix with shape (``n_mels``, ``n_mfcc``),
        it can be normalized depending on norm. For more information about dct:
        http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

        Parameters
        ----------
        n_mfcc : int
            Number of mfccs
        n_mels : int
            Number of mels
        norm   : str
            Use "ortho" to get a orthogonal matrix or None, by default "ortho"
        device : str, optional
            Device to load the transformation matrix on, by default None

        Returns
        -------
        torch.Tensor [shape=(n_mels, n_mfcc)] T
            The dct transformation matrix.
        """
        from torchaudio.functional import create_dct

        return create_dct(n_mfcc, n_mels, norm).to(device)

    def mfcc(
        self, n_mfcc: int = 40, n_mels: int = 80, log_offset: float = 1e-6, **kwargs
    ):
        """Computes mel-frequency cepstral coefficients (MFCCs).

        Parameters
        ----------
        n_mfcc : int, optional
            Number of mels, by default 40
        n_mels : int, optional
            Number of mels, by default 80
        log_offset: float, optional
            Small value to prevent numerical issues when trying to compute log(0), by default 1e-6
        kwargs : dict, optional
            Keyword arguments to self.mel_spectrogram(), note that some of them will be used for self.stft()

        Returns
        -------
        torch.Tensor [shape=(batch, channels, mfccs, time)]
            MFCCs.
        """

        mel_spectrogram = self.mel_spectrogram(n_mels, **kwargs)
        mel_spectrogram = torch.log(mel_spectrogram + log_offset)
        dct_mat = self.get_dct(n_mfcc, n_mels, "ortho", self.device)

        mfcc = mel_spectrogram.transpose(-1, -2) @ dct_mat
        mfcc = mfcc.transpose(-1, -2)
        return mfcc

    @property
    def magnitude(self):
        """Computes and returns the absolute value of the STFT, which
        is the magnitude. This value can also be set to some tensor.
        When set, ``self.stft_data`` is manipulated so that its magnitude
        matches what this is set to, and modulated by the phase.

        Returns
        -------
        torch.Tensor
            Magnitude of STFT.

        Examples
        --------
        >>> signal = AudioSignal(torch.randn(44100), 44100)
        >>> magnitude = signal.magnitude # Computes stft if not computed
        >>> magnitude[magnitude < magnitude.mean()] = 0
        >>> signal.magnitude = magnitude
        >>> signal.istft()
        """
        if self.stft_data is None:
            self.stft()
        return torch.abs(self.stft_data)

    @magnitude.setter
    def magnitude(self, value):
        self.stft_data = value * torch.exp(1j * self.phase)
        return

    def log_magnitude(
        self, ref_value: float = 1.0, amin: float = 1e-5, top_db: float = 80.0
    ):
        """Computes the log-magnitude of the spectrogram.

        Parameters
        ----------
        ref_value : float, optional
            The magnitude is scaled relative to ``ref``: ``20 * log10(S / ref)``.
            Zeros in the output correspond to positions where ``S == ref``,
            by default 1.0
        amin : float, optional
            Minimum threshold for ``S`` and ``ref``, by default 1e-5
        top_db : float, optional
            Threshold the output at ``top_db`` below the peak:
            ``max(10 * log10(S/ref)) - top_db``, by default -80.0

        Returns
        -------
        torch.Tensor
            Log-magnitude spectrogram
        """
        magnitude = self.magnitude

        amin = amin**2
        log_spec = 10.0 * torch.log10(magnitude.pow(2).clamp(min=amin))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            log_spec = torch.maximum(log_spec, log_spec.max() - top_db)
        return log_spec

    @property
    def phase(self):
        """Computes and returns the phase of the STFT.
        This value can also be set to some tensor.
        When set, ``self.stft_data`` is manipulated so that its phase
        matches what this is set to, we original magnitudeith th.

        Returns
        -------
        torch.Tensor
            Phase of STFT.

        Examples
        --------
        >>> signal = AudioSignal(torch.randn(44100), 44100)
        >>> phase = signal.phase # Computes stft if not computed
        >>> phase[phase < phase.mean()] = 0
        >>> signal.phase = phase
        >>> signal.istft()
        """
        if self.stft_data is None:
            self.stft()
        return torch.angle(self.stft_data)

    @phase.setter
    def phase(self, value):
        self.stft_data = self.magnitude * torch.exp(1j * value)
        return

    # Operator overloading
    def __add__(self, other):
        new_signal = self.clone()
        new_signal.audio_data += _get_value(other)
        return new_signal

    def __iadd__(self, other):
        self.audio_data += _get_value(other)
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        new_signal = self.clone()
        new_signal.audio_data -= _get_value(other)
        return new_signal

    def __isub__(self, other):
        self.audio_data -= _get_value(other)
        return self

    def __mul__(self, other):
        new_signal = self.clone()
        new_signal.audio_data *= _get_value(other)
        return new_signal

    def __imul__(self, other):
        self.audio_data *= _get_value(other)
        return self

    def __rmul__(self, other):
        return self * other

    # Representation
    def _info(self):
        dur = f"{self.signal_duration:0.3f}" if self.signal_duration else "[unknown]"
        info = {
            "duration": f"{dur} seconds",
            "batch_size": self.batch_size,
            "path": self.path_to_file if self.path_to_file else "path unknown",
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels if self.num_channels else "[unknown]",
            "audio_data.shape": self.audio_data.shape,
            "stft_params": self.stft_params,
            "device": self.device,
        }

        return info

    def markdown(self):
        """Produces a markdown representation of AudioSignal, in a markdown table.

        Returns
        -------
        str
            Markdown representation of AudioSignal.

        Examples
        --------
        >>> signal = AudioSignal(torch.randn(44100), 44100)
        >>> print(signal.markdown())
        | Key | Value
        |---|---
        | duration | 1.000 seconds |
        | batch_size | 1 |
        | path | path unknown |
        | sample_rate | 44100 |
        | num_channels | 1 |
        | audio_data.shape | torch.Size([1, 1, 44100]) |
        | stft_params | STFTParams(window_length=2048, hop_length=512, window_type='sqrt_hann', match_stride=False) |
        | device | cpu |
        """
        info = self._info()

        FORMAT = "| Key | Value \n" "|---|--- \n"
        for k, v in info.items():
            row = f"| {k} | {v} |\n"
            FORMAT += row
        return FORMAT

    def __str__(self):
        info = self._info()

        desc = ""
        for k, v in info.items():
            desc += f"{k}: {v}\n"
        return desc

    def __rich__(self):
        from rich.table import Table

        info = self._info()

        table = Table(title=f"{self.__class__.__name__}")
        table.add_column("Key", style="green")
        table.add_column("Value", style="cyan")

        for k, v in info.items():
            table.add_row(k, str(v))
        return table

    # Comparison
    def __eq__(self, other):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                if not torch.allclose(v, other.__dict__[k], atol=1e-6):
                    max_error = (v - other.__dict__[k]).abs().max()
                    print(f"Max abs error for {k}: {max_error}")
                    return False
        return True

    # Indexing
    def __getitem__(self, key):
        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            audio_data = self.audio_data
            _loudness = self._loudness
            stft_data = self.stft_data

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            # Indexing only on the batch dimension.
            # Then let's copy over relevant stuff.
            # Future work: make this work for time-indexing
            # as well, using the hop length.
            audio_data = self.audio_data[key]
            _loudness = self._loudness[key] if self._loudness is not None else None
            stft_data = self.stft_data[key] if self.stft_data is not None else None

        sources = None

        copy = type(self)(audio_data, self.sample_rate, stft_params=self.stft_params)
        copy._loudness = _loudness
        copy._stft_data = stft_data
        copy.sources = sources

        return copy

    def __setitem__(self, key, value):
        if not isinstance(value, type(self)):
            self.audio_data[key] = value
            return

        if torch.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            self.audio_data = value.audio_data
            self._loudness = value._loudness
            self.stft_data = value.stft_data
            return

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
            torch.is_tensor(key) and key.ndim <= 1
        ):
            if self.audio_data is not None and value.audio_data is not None:
                self.audio_data[key] = value.audio_data
            if self._loudness is not None and value._loudness is not None:
                self._loudness[key] = value._loudness
            if self.stft_data is not None and value.stft_data is not None:
                self.stft_data[key] = value.stft_data
            return

    def __ne__(self, other):
        return not self == other