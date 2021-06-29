from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import re
import random
import librosa
from model_davenet import LoadAudio


class Youtube_DataLoader(Dataset):
    """Youtube dataset loader."""

    def __init__(
            self,
            csv,
            features_path,
            features_path_audio,
            caption,
            we,
            min_time=10.0,
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            we_dim=300,
            max_words=30,
            min_words=0,
            n_pair=1,
            num_audio_frames=1024,
            random_audio_windows=False,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.features_path = features_path
        self.features_path_audio = features_path_audio if features_path_audio != "" \
                                   else features_path
        self.caption = caption
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.feature_framerate_3D = feature_framerate_3D
        self.we_dim = we_dim
        self.max_words = max_words
        self.min_words = min_words
        self.num_audio_frames = num_audio_frames
        self.we = we
        self.n_pair = n_pair
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = {'2d': features_path}
        if features_path != '':
            self.feature_path['3d'] = features_path
        self.random_audio_windows = random_audio_windows

    def __len__(self):
        return len(self.csv)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _zero_pad_audio(self, audio, max_frames):
        n_frames = audio.shape[1]
        if n_frames >= max_frames:
            return audio[:, 0:max_frames], int(max_frames)
        else:
            p = max_frames - n_frames
            audio_padded = np.pad(audio, ((0, 0),(0, p)), 'constant', constant_values=(0, 0))
            return audio_padded, n_frames

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def _get_audio_and_text(self, caption, n_pair_max, mel_spec):
        n_caption = len(caption['start'])
        k = n_pair_max
        starts = np.zeros(k)
        ends = np.zeros(k)
        text = th.zeros(k, self.max_words, self.we_dim)
        audio = [0 for i in range(k)]
        nframes = np.zeros(k)
        r_ind = np.random.choice(range(n_caption), k, replace=True)

        for i in range(k):
            ind = r_ind[i]
            audio[i], nframes[i], starts[i], ends[i], text[i] = self._get_single_audio_text(caption, ind, mel_spec)

        audio = th.cat([i.unsqueeze(0) for i in audio], dim=0)
        return audio, nframes, starts, ends, text

    def _get_single_audio_text(self, caption, ind, mel_spec):
        start, end = ind, ind
        words = self._tokenize_text(caption['text'][ind])
        diff = caption['end'][end] - caption['start'][start]
        # Extend the video clip if shorter than the minimum desired clip duration
        while diff < self.min_time:
            if start > 0 and end < len(caption['end']) - 1:
                next_words = self._tokenize_text(caption['text'][end + 1])
                prev_words = self._tokenize_text(caption['text'][start - 1])
                d1 = caption['end'][end + 1] - caption['start'][start]
                d2 = caption['end'][end] - caption['start'][start - 1]
                # Use the closest neighboring video clip
                if d2 <= d1:
                    start -= 1
                    words.extend(prev_words)    
                else:
                    end += 1
                    words.extend(next_words)
            # If no video clips after it, use the clip before it
            elif start > 0:
                words.extend(self._tokenize_text(caption['text'][start - 1]))
                start -= 1
             # If no video clips before it, use the clip after it.
            elif end < len(caption['end']) - 1:
                words.extend(self._tokenize_text(caption['text'][end + 1])) 
                end += 1
            # If there's no clips before or after
            else:
                break
            diff = caption['end'][end] - caption['start'][start]
        
        frames = librosa.core.time_to_frames([caption['start'][start], caption['end'][end]], sr=16000, hop_length=160, n_fft=400)
        padded_mel_spec, nframes = self._zero_pad_audio(mel_spec[:, frames[0]: frames[1]], self.num_audio_frames)
        return th.from_numpy(padded_mel_spec), nframes, caption['start'][start], caption['end'][end], self._words_to_we(words)

    def _get_audio_random(self, n_pair_max, mel_spec):
        k = n_pair_max
        starts = np.zeros(k)
        ends = np.zeros(k)
        audio = [0 for i in range(k)]
        nframes = np.zeros(k)
        video_duration_seconds = int(librosa.core.frames_to_time(mel_spec.shape[1], sr=16000, hop_length=160, n_fft=400))
        num_audio_seconds = int(librosa.core.frames_to_time(self.num_audio_frames, sr=16000, hop_length=160, n_fft=400))
        # Sample clips that end before the end of the video
        # If the video is shorter than the desired window, use the entire video
        start_seconds = np.random.choice(range(max(1, video_duration_seconds - (num_audio_seconds + 1))), k, replace=True)

        for i in range(k):
            start_frame = max(0, librosa.core.time_to_frames(start_seconds[i], sr=16000, hop_length=160, n_fft=400))
            audio_window = mel_spec[:, start_frame : start_frame + self.num_audio_frames]
            # Pad in the case that the audio wasn't long enough
            padded_mel_spec, nframes_spec = self._zero_pad_audio(audio_window, self.num_audio_frames)   
            end_second = start_seconds[i] + num_audio_seconds
            audio[i], nframes[i], starts[i], ends[i] = th.from_numpy(padded_mel_spec), nframes_spec, start_seconds[i], end_second

        audio = th.cat([i.unsqueeze(0) for i in audio], dim=0)
        return audio, nframes, starts, ends

    def _get_video(self, vid_path, s, e, video_id):
        feature_path = {}
        video = {}
        output = {}
        for k in self.feature_path:
            feature_path[k] = os.path.join(self.feature_path[k], vid_path, video_id + "_{}.npz".format(k))
            np_arr = np.load(feature_path[k])['features']
            video[k] = th.from_numpy(np_arr).float()
            output[k] = th.zeros(len(s), video[k].shape[-1])
            for i in range(len(s)):
                start = int(s[i] * self.fps[k])
                end = int(e[i] * self.fps[k]) + 1
                slice = video[k][start:end]
                if len(slice) < 1:
                    print("missing visual feats; video_id: {}, start: {}, end: {}".format(
                        feature_path[k], start, end))
                else:
                    output[k][i] = F.normalize(th.max(slice, dim=0)[0], dim=0)

        return th.cat([output[k] for k in output], dim=1)

    def __getitem__(self, idx):
        vid_path = self.csv['path'].values[idx].replace("None/", "")
        video_id = vid_path.split("/")[-1]
        audio_path = os.path.join(self.features_path_audio, vid_path, video_id + "_spec.npz")
        mel_spec = np.load(audio_path)['arr_0']
        if self.random_audio_windows:
            audio, nframes, starts, ends = self._get_audio_random(self.n_pair, mel_spec)
        else:
            audio, nframes, starts, ends, text = self._get_audio_and_text(self.caption[video_id], self.n_pair, mel_spec)
        video = self._get_video(vid_path, starts, ends, video_id)
        if self.random_audio_windows: 
            return {'video': video, 'audio':th.HalfTensor(audio), 'nframes': th.IntTensor(nframes), 'video_id': video_id}
        else:
            return {'video': video, 'audio':th.HalfTensor(audio), 'nframes': th.IntTensor(nframes), 'video_id': video_id,
                    'text': text}  