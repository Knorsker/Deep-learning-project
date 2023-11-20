import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from typing import Union
import math
from pathlib import Path
# from audiotools import AudioSignal
import tqdm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

SUPPORTED_VERSIONS = ["1.0.0"]
class DACFile:
    codes: torch.Tensor

    # Metadata
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

    def save(self, path):
        artifacts = {
            "codes": self.codes.numpy().astype(np.uint16),
            "metadata": {
                "input_db": self.input_db.numpy().astype(np.float32),
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "padding": self.padding,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        with open(path, "wb") as f:
            np.save(f, artifacts)
        return path

    @classmethod
    def load(cls, path):
        artifacts = np.load(path, allow_pickle=True)[()]
        codes = torch.from_numpy(artifacts["codes"].astype(int))
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])

# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

class CodecMixin:
    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        assert isinstance(value, bool)

        layers = [
            l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L

    # @torch.no_grad()
    # def compress(
    #     self,
    #     audio_path_or_signal: Union[str, Path, AudioSignal],
    #     win_duration: float = 1.0,
    #     verbose: bool = False,
    #     normalize_db: float = -16,
    #     n_quantizers: int = None,
    # ) -> DACFile:
    #     """Processes an audio signal from a file or AudioSignal object into
    #     discrete codes. This function processes the signal in short windows,
    #     using constant GPU memory.

    #     Parameters
    #     ----------
    #     audio_path_or_signal : Union[str, Path, AudioSignal]
    #         audio signal to reconstruct
    #     win_duration : float, optional
    #         window duration in seconds, by default 5.0
    #     verbose : bool, optional
    #         by default False
    #     normalize_db : float, optional
    #         normalize db, by default -16

    #     Returns
    #     -------
    #     DACFile
    #         Object containing compressed codes and metadata
    #         required for decompression
    #     """
    #     audio_signal = audio_path_or_signal
    #     if isinstance(audio_signal, (str, Path)):
    #         audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))

    #     self.eval()
    #     original_padding = self.padding
    #     original_device = audio_signal.device

    #     audio_signal = audio_signal.clone()
    #     original_sr = audio_signal.sample_rate

    #     resample_fn = audio_signal.resample
    #     loudness_fn = audio_signal.loudness

    #     # If audio is > 10 minutes long, use the ffmpeg versions
    #     if audio_signal.signal_duration >= 10 * 60 * 60:
    #         resample_fn = audio_signal.ffmpeg_resample
    #         loudness_fn = audio_signal.ffmpeg_loudness

    #     original_length = audio_signal.signal_length
    #     resample_fn(self.sample_rate)
    #     input_db = loudness_fn()

    #     if normalize_db is not None:
    #         audio_signal.normalize(normalize_db)
    #     audio_signal.ensure_max_of_audio()

    #     nb, nac, nt = audio_signal.audio_data.shape
    #     audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
    #     win_duration = (
    #         audio_signal.signal_duration if win_duration is None else win_duration
    #     )

    #     if audio_signal.signal_duration <= win_duration:
    #         # Unchunked compression (used if signal length < win duration)
    #         self.padding = True
    #         n_samples = nt
    #         hop = nt
    #     else:
    #         # Chunked inference
    #         self.padding = False
    #         # Zero-pad signal on either side by the delay
    #         audio_signal.zero_pad(self.delay, self.delay)
    #         n_samples = int(win_duration * self.sample_rate)
    #         # Round n_samples to nearest hop length multiple
    #         n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
    #         hop = self.get_output_length(n_samples)

    #     codes = []
    #     range_fn = range if not verbose else tqdm.trange

    #     for i in range_fn(0, nt, hop):
    #         x = audio_signal[..., i : i + n_samples]
    #         x = x.zero_pad(0, max(0, n_samples - x.shape[-1]))

    #         audio_data = x.audio_data.to(self.device)
    #         audio_data = self.preprocess(audio_data, self.sample_rate)
    #         _, c, _, _, _ = self.encode(audio_data, n_quantizers)
    #         codes.append(c.to(original_device))
    #         chunk_length = c.shape[-1]

    #     codes = torch.cat(codes, dim=-1)

    #     dac_file = DACFile(
    #         codes=codes,
    #         chunk_length=chunk_length,
    #         original_length=original_length,
    #         input_db=input_db,
    #         channels=nac,
    #         sample_rate=original_sr,
    #         padding=self.padding,
    #         dac_version=SUPPORTED_VERSIONS[-1],
    #     )

    #     if n_quantizers is not None:
    #         codes = codes[:, :n_quantizers, :]

    #     self.padding = original_padding
    #     return dac_file

    # @torch.no_grad()
    # def decompress(
    #     self,
    #     obj: Union[str, Path, DACFile],
    #     verbose: bool = False,
    # ) -> AudioSignal:
    #     """Reconstruct audio from a given .dac file

    #     Parameters
    #     ----------
    #     obj : Union[str, Path, DACFile]
    #         .dac file location or corresponding DACFile object.
    #     verbose : bool, optional
    #         Prints progress if True, by default False

    #     Returns
    #     -------
    #     AudioSignal
    #         Object with the reconstructed audio
    #     """
    #     self.eval()
    #     if isinstance(obj, (str, Path)):
    #         obj = DACFile.load(obj)

    #     original_padding = self.padding
    #     self.padding = obj.padding

    #     range_fn = range if not verbose else tqdm.trange
    #     codes = obj.codes
    #     original_device = codes.device
    #     chunk_length = obj.chunk_length
    #     recons = []

    #     for i in range_fn(0, codes.shape[-1], chunk_length):
    #         c = codes[..., i : i + chunk_length].to(self.device)
    #         z = self.quantizer.from_codes(c)[0]
    #         r = self.decode(z)
    #         recons.append(r.to(original_device))

    #     recons = torch.cat(recons, dim=-1)
    #     recons = AudioSignal(recons, self.sample_rate)

    #     resample_fn = recons.resample
    #     loudness_fn = recons.loudness

    #     # If audio is > 10 minutes long, use the ffmpeg versions
    #     if recons.signal_duration >= 10 * 60 * 60:
    #         resample_fn = recons.ffmpeg_resample
    #         loudness_fn = recons.ffmpeg_loudness

    #     recons.normalize(obj.input_db)
    #     resample_fn(obj.sample_rate)
    #     recons = recons[..., : obj.original_length]
    #     loudness_fn()
    #     recons.audio_data = recons.audio_data.reshape(
    #         -1, obj.channels, obj.original_length
    #     )

    #     self.padding = original_padding
    #     return recons

