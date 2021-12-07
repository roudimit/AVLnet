from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
from torch.utils.data.dataloader import default_collate
import audio_to_spectrograms
import os

class SMiT_DataLoader(Dataset):
    """S-MiT dataset loader, modified from YouCook dataset loader."""

    def __init__(
            self,
            data_path,
            we,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=20, # captions should be no longer than 20s
            tri_modal=False,    
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data_path, 'rb'))
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.num_frames_multiplier = num_frames_multiplier
        self.tri_modal = tri_modal

    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

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

    def __getitem__(self, idx):
        # load 2d and 3d features (features are max-pooled over the time dimension)
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
        video = th.cat((feat_2d, feat_3d))

        # load audio and zero pad/truncate if necessary
        """
        caption_audio_file = self.data[idx]['spoken_caption_path']
        audio_to_spectrograms.extract_audio(caption_audio_file, 'temp1.wav', sys.stdout)
        audio_to_spectrograms.stereo_to_mono_downsample('temp1.wav', 'temp2.wav', 48000)
        feats, frames = audio_to_spectrograms.LoadAudio('temp2.wav')
        caption_audio_feats = feats
        os.remove('temp1.wav')
        os.remove('temp2.wav')
        """
        caption_audio_file = '/nobackup/users/lynberry/caption_audio/'+self.data[idx]['id']+'.wav'
        caption_audio_feats = th.zeros((40,1024*self.num_frames_multiplier),dtype=th.float)
        nframes = 0
        try:
            feats, frames = audio_to_spectrograms.LoadAudio(caption_audio_file, use_raw_length=True)
            caption_audio_feats = feats
        
            target_length = 1024 * self.num_frames_multiplier
            nframes = caption_audio_feats.shape[1]
            assert nframes == frames
            p = target_length - nframes
            if p > 0:
                caption_audio_feats = np.pad(caption_audio_feats, ((0,0),(0,p)), 'constant', constant_values=(0,0))
            elif p < 0:
                caption_audio_feats = caption_audio_feats[:,0:p]
            caption_audio_feats = th.FloatTensor(caption_audio_feats)
        except:
            ... #print("failed on file", caption_audio_file)

        natural_audio_feats = th.zeros((40,3072),dtype=th.float) # should this be an empty tensor? Yes. what shape?
        if self.data[idx]['has_audio']:
            try:
                natural_audio_file = '/nobackup/users/lynberry/natural_audio/'+self.data[idx]['id']+'.wav'
                feats, frames = audio_to_spectrograms.LoadAudio(natural_audio_file, use_raw_length=True)
                natural_audio_feats = feats
    
                target_length = 1024 * 3 # All clips are 3s long, so no self.num_frames_multiplier
                nframes = natural_audio_feats.shape[1]
                assert nframes == frames
                p = target_length - nframes
                if p > 0:
                    natural_audio_feats = np.pad(natural_audio_feats, ((0,0),(0,p)), 'constant', constant_values=(0,0))
                elif p < 0:
                    natural_audio_feats = natural_audio_feats[:,0:p]
                natural_audio_feats = th.FloatTensor(natural_audio_feats)
            except:
                ... #print("failed on file", natural_audio_file)
            if natural_audio_feats.shape[1] != 3072:
                print("file", natural_audio_file, "has feats shape", natural_audio_feats.shape)
                print("p was", p, "with target length", target_length, "and nframes", nframes)

        caption = ''
        if self.tri_modal:
            caption = self._words_to_we(self._tokenize_text(self.data[idx]['text_caption'])) 
        
        task, start, end, vid_id = 0, 0, 0, ''
        if 'task' in self.data[idx]: # TODO: this will always fail atm
            task = int(self.data[idx]['task'])
            start = int(self.data[idx]['start'])
            end = int(self.data[idx]['end'])
            vid_id = self.data[idx]['video_id']
        
        text_sim = np.array(1)
        if 'text_sim' in self.data[idx]: # TODO: this will always fail atm
            text_sim = self.data[idx]['text_sim']

        assert caption_audio_feats.shape[1] == 20480
        assert natural_audio_feats.shape[1] == 3072
        # print("caption_audio_feats.shape", caption_audio_feats.shape)
        # print("natural_audio_feats.shape", natural_audio_feats.shape)
        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': caption_audio_feats, 'natural_audio': natural_audio_feats,
                'nframes': nframes, 'task': task, 'start': start, 'end': end, 
                'vid_id': vid_id, 'text_sim': text_sim}
        