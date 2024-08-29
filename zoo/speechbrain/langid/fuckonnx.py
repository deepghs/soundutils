import torch
from speechbrain.processing.features import STFT
from speechbrain.utils.filter_analysis import FilterProperties


class _F_STFT(torch.nn.Module):
    # fuck original STFT module

    def __init__(self, origin: STFT):
        super().__init__()
        self.origin = origin

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : torch.Tensor
            A batch of audio signals to transform.

        Returns
        -------
        stft : torch.Tensor
        """
        # Managing multi-channel stft
        stft = torch.stft(
            x,
            self.origin.n_fft,
            self.origin.hop_length,
            self.origin.win_length,
            self.origin.window,
            self.origin.center,
            self.origin.pad_mode,
            self.origin.normalized_stft,
            self.origin.onesided,
            return_complex=True,
        )

        stft = torch.view_as_real(stft)

        # Retrieving the original dimensionality (batch,time, channels)
        stft = stft.transpose(2, 1)

        return stft

        # print('Fucking input x shape', x.shape)
        # return self.origin.forward(x)

    def get_filter_properties(self) -> FilterProperties:
        return self.origin.get_filter_properties()
