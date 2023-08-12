import random
import numpy as np
import torch
import torchaudio


class XGBLoader:
    def __init__(
        self, annotation: dict, new_s_freq: int = 256, window_size: int = 20
    ) -> None:
        self.idx = 0
        self.new_s_freq = new_s_freq
        self.window_size = window_size
        self.annotation = annotation

        default_channel_nums = 22
        sample_freq = annotation["s_freq"]  # sampleing freqeuncy
        montage = annotation["montage"]
        # resample EEG to a fixed sampling frequency.
        resampler = torchaudio.transforms.Resample(sample_freq, new_s_freq)

        with np.load(annotation["npz_filepath"]) as npz_file:
            raw_eeg = npz_file["arr_0"]

        raw_eeg = torch.from_numpy(raw_eeg).to(torch.float32)
        # resample
        raw_eeg_resample = resampler(raw_eeg)

        if montage not in ["01_tcp_ar", "02_tcp_le"]:
            zero_eeg = torch.zeros(default_channel_nums, raw_eeg_resample.shape[-1])
            zero_eeg[0 : raw_eeg_resample.shape[0], ...] = raw_eeg_resample

            raw_eeg_resample = zero_eeg

        self.x, self.y = self.create_window(
            raw_eeg_resample, self.annotation["channel_annot"]
        )

    def _psd_extract(self, x: torch.Tensor):
        x_fft = torch.fft.fft(x, norm="ortho", dim=-1)
        # Compute the power specturm of the signal
        power_spectrum = torch.abs(x_fft) ** 2
        # Normalize the power spectrum by the number of samples in the signal
        power_spectrum /= power_spectrum.shape[-1]
        return power_spectrum

    def create_window(self, eeg_sample: np.ndarray, channel_annot: dict):
        """Resample and generate class label tensor for an eeg reading.
        Args:
        eeg_sample (nd.array): Raw EEG data
        channel_annot (dict): corresponding annoations
        s_fre (int): sampling frequency.
        window_size: window_size in seconds.
        """
        context_length = (
            self.window_size * self.new_s_freq
        )  # change window_size(seconds) to sequence_lenth
        sample_length = eeg_sample.shape[-1]  # total length of the raw eeg signal

        # pad the `eeg_sample` to the nearest integer factor of `window_size`
        padding_size = int(
            context_length
            * torch.ceil(torch.tensor(sample_length / context_length)).item()
        )
        padded_zero = torch.zeros(eeg_sample.shape[0], padding_size)
        padded_zero[..., 0:sample_length] = eeg_sample
        padded_zero = padded_zero.view(-1, padded_zero.shape[0], context_length)
        # class labels
        target = torch.zeros(padded_zero.shape[0], padded_zero.shape[1])

        for idx in range(target.shape[0]):
            channel_labels_tensor = torch.zeros(target.shape[1])
            channel_labels = []

            for i, labels in enumerate(channel_annot.values()):
                for label in labels:
                    start_time, stop_time, c = label

                    sample_start_time = idx * self.window_size
                    sample_stop_time = (idx + 1) * self.window_size
                    if (
                        sample_start_time >= start_time
                        and sample_stop_time <= stop_time
                    ):
                        channel_labels.append(0 if c == "bckg" else 1)

            channel_labels_tensor[0 : len(channel_labels)] = torch.tensor(
                channel_labels, dtype=torch.float32
            )

            target[idx, ...] = channel_labels_tensor
        # target = target.unsqueeze(-1)
        return padded_zero, target

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx > len(self.x) - 1:
            raise StopIteration

        x = self.x[self.idx]
        # x = self._psd_extract(x)
        y = self.y[self.idx]

        """ if torch.all(y):
            idx = random.randint(0, len(y) - 1)
            x[idx, ...] = torch.zeros(x.shape[-1])
            y[idx] = 0.0 """
        self.idx += 1
        return x, y
