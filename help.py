from torch import nn
import torch
import argbind
from typing import List, Union
import numpy as np
import math
from help2 import Snake1d, WNConv1d, ResidualVectorQuantize, WNConvTranspose1d, CodecMixin
from audiotools.ml import BaseModel
from pathlib import Path

__MODEL_LATEST_TAGS__ = {
    ("44khz", "8kbps"): "0.0.1",
    ("24khz", "8kbps"): "0.0.4",
    ("16khz", "8kbps"): "0.0.5",
    ("44khz", "16kbps"): "1.0.0",
}
__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "44khz",
        "1.0.0",
        "16kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}



@argbind.bind(group="download", positional=True, without_prefix=True)
def download(
    model_type: str = "44khz", model_bitrate: str = "8kbps", tag: str = "latest"
):
    """
    Function that downloads the weights file from URL if a local cache is not found.

    Parameters
    ----------
    model_type : str
        The type of model to download. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz".
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
        Only 44khz model supports 16kbps.
    tag : str
        The tag of the model to download. Defaults to "latest".

    Returns
    -------
    Path
        Directory path required to load model via audiotools.
    """
    # model_type = model_type.lower()
    model_type = str(model_type).lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[(model_type, model_bitrate)]

    download_link = __MODEL_URLS__.get((model_type, tag, model_bitrate), None)

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    local_path = (
        Path.home()
        / ".cache"
        / "descript"
        / "dac"
        / f"weights_{model_type}_{model_bitrate}_{tag}.pth"
    )
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        local_path.write_bytes(response.content)

    return local_path


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

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
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

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
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


def load_model(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
    load_path: str = None,
):
    if not load_path:
        load_path = download(
            model_type=model_type, model_bitrate=model_bitrate, tag=tag
        )
    generator = DAC.load(load_path)
    return generator




