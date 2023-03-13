import torchaudio
import torch
from .config import signal_config
import scipy.stats as stats
import numpy as np


def wav2fbank(num_mel_bins):
    """
    Returns a function to convert a waveform to a log Mel-scaled spectrogram using either Kaldi or TorchAudio.

    Args:
        None

    Returns:
        function: A function to convert a waveform to a log Mel-scaled spectrogram.
    """
    def kaldi_fbank(waveform):
        """
        Converts a waveform to a log Mel-scaled spectrogram using Kaldi.

        Args:
            waveform (torch.Tensor): A 1D Tensor representing a waveform.

        Returns:
            torch.Tensor: A 2D Tensor representing a log Mel-scaled spectrogram.
        """
        # Compute the log Mel-scaled spectrogram using Kaldi
        return torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=signal_config['sample_rate'],
            frame_length=signal_config['frame_length'],
            frame_shift=signal_config['frame_shift'],
            htk_compat=True,
            window_type=signal_config['window_type'],
            num_mel_bins=num_mel_bins
        )

    def torchaudio_fbank(waveform):
        """
        Converts a waveform to a log Mel-scaled spectrogram using TorchAudio.

        Args:
            waveform (torch.Tensor): A 1D Tensor representing a waveform.

        Returns:
            torch.Tensor: A 2D Tensor representing a log Mel-scaled spectrogram.
        """
        # Compute the log Mel-scaled spectrogram using TorchAudio
        sample_rate = signal_config['sample_rate']
        frame_length = signal_config['frame_length']
        frame_shift = signal_config['frame_shift']
        window_type = signal_config['window_type']
        sample_num_per_ms = int(sample_rate / 1000)
        frame_length_samples = int(sample_num_per_ms * frame_length)
        frame_shift_samples = int(sample_num_per_ms * frame_shift)

        # Set the window function based on the window_type parameter
        if window_type == 'hanning':
            window_fn = torch.hann_window
        elif window_type == 'hamming':
            window_fn = torch.hamming_window
        elif window_type == 'blackman':
            window_fn = torch.blackman_window

        # Create the MelSpectrogram transform object
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frame_length_samples,
            hop_length=frame_shift_samples,
            center=False,
            f_min=20,
            power=2.0,
            n_mels=num_mel_bins,
            window_fn=window_fn
        )

        # Compute the log Mel-scaled spectrogram
        return torch.log10(mel_spectrogram(waveform) + 1e-6)

    # Choose the appropriate function based on the compliance parameter in the signal_config dictionary
    if signal_config['compliance'] == 'kaldi':
        return kaldi_fbank
    elif signal_config['compliance'] == 'torchaudio':
        return torchaudio_fbank


def generate_gap_audio(ms):
    """
    Generates a gap audio signal.

    Args:
        ms (int): The length of the gap in milliseconds.

    Returns:
        numpy.ndarray: A NumPy array containing the generated audio data.
    """
    # Set the maximum amplitude for the audio signal
    max_amp = 0.01  # This value should be specified in the signal_config dictionary

    fs = signal_config["sample_rate"]

    # Generate a truncated normal distribution with a mean of 0 and standard deviation of 1
    # This distribution is truncated to be between -1 and 1
    # The scale parameter is set to the minimum of 2**16 and 2**max_amp
    # This ensures that the audio data will fit within the range of a 16-bit integer
    noise = stats.truncnorm(-1, 1, scale=min(2**16, 2**max_amp)).rvs(ms * fs // 1000)

    # Convert the audio data to a 16-bit integer and return it
    noise = noise.astype(np.int16)
    return noise