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
import time
import random
import ffmpeg
import h5py
import io
from PIL import Image
from torchvision.transforms import ToTensor
import soundfile as sf
# from torch import torchaudio

class SMiT_DataLoader(Dataset):
    """S-MiT dataset loader, modified from YouCook dataset loader."""

    def __init__(
            self,
            data_path,
            we,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=2, # captions should be no longer than 20s
            tri_modal=False,   
            load_images=False,
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
        self.load_images = load_images
        if 'train' in data_path:
            hdf5_path = '/nobackup/users/lynberry/train_frames.hdf5'
        elif 'val' in data_path:
            hdf5_path = '/nobackup/users/lynberry/val_frames.hdf5'
        elif 'test' in data_path:
            hdf5_path = '/nobackup/users/lynberry/test_frames.hdf5'
        else:
            print("Can't find frames file for data path", data_path)
        #self.frame_file = h5py.File(hdf5_path, 'r')
        self.toTensor = ToTensor()
        self.audio_conf = {'num_mel_bins':128, 'target_length':1024, 'freqm':0, 'timem':0, 'mixup':0, 'dataset':'audioset', 'mode':'train', 'mean':-1.9999702, 'std':4.110817, 'noise':False}

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
        start_time = time.time()
        if self.load_images:
            # Load 8 frames from hdf5 file
            #vid_grp = self.frame_file[self.data[idx]['id']]
            imgs = []
            for img in self.data[idx]['frames']: # vid_grp.keys():
                imgs.append(self.toTensor(Image.open(img)))
            video = th.stack(imgs)
            video_unpooled = None
        else:
            #This section assumes video features are already extracted by a frozen model
            # load 2d and 3d features (features are max-pooled over the time dimension)
            feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
            feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
            video = th.cat((feat_2d, feat_3d))
            unpooled2d = F.normalize(th.from_numpy(self.data[idx]['2d_full']).float(), dim=0)
            unpooled3d = F.normalize(th.from_numpy(self.data[idx]['3d_full']).float(), dim=0)
            tokens2d = unpooled2d.shape[0]
            tokens3d = unpooled3d.shape[0]
            if unpooled2d.shape[0] < 4:
                unpooled2d = F.pad(unpooled2d, (0,0,0,4-unpooled2d.shape[0]), 'constant', 0)
            elif unpooled2d.shape[0] > 4:
                unpooled2d = unpooled2d[:4]
            if unpooled3d.shape[0] < 6:
                unpooled3d = F.pad(unpooled3d, (0,0,0,6-unpooled3d.shape[0]), 'constant', 0)
            elif unpooled3d.shape[0] > 6:
                unpooled3d = unpooled3d[:6]
            video_unpooled = th.cat((unpooled2d, unpooled3d))
            assert(video_unpooled.shape[0] == 10)
        video_time = time.time()

        caption = ''
        if self.tri_modal:
            caption = self._words_to_we(self._tokenize_text(self.data[idx]['text_caption'])) 
        
        task, start, end, vid_id = 0, 0, 0, ''
        if 'task' in self.data[idx]: # will always be false, not using for S-MiT
            task = int(self.data[idx]['task'])
            start = int(self.data[idx]['start'])
            end = int(self.data[idx]['end'])
            vid_id = self.data[idx]['video_id']
        
        text_sim = np.array(1)
        if 'text_sim' in self.data[idx]: # will always be false, not using for S-MiT
            text_sim = self.data[idx]['text_sim']

        caption_audio_feats = self.data[idx]['caption_audio_feats']
        natural_audio_feats = self.data[idx]['natural_audio_feats']

        nframes = len(self.data[idx]['nframes'])

        # Extract random 10s from captions
        if nframes > 1024:
            start_index = random.randint(0,min(nframes,2048)-1024)
        else:
            start_index = 0
        caption_audio_feats = caption_audio_feats[:,:,start_index:start_index+1024]

        """
        # Get natural audio spectrogram
        if self.data[idx]['has_audio']:
            waveform, samplerate = sf.read(file=self.data[idx]['natural_audio_bytes'], dtype='float32')
            waveform = waveform - waveform.mean()
            print('waveform', type(waveform), waveform)
            print('samplerate', type(samplerate), samplerate)

            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=samplerate, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            target_length = self.audio_conf.get('target_length')
            n_frames = fbank.shape[0]
            p = target_length - n_frames
    
            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]

        # end of wav2fbank method, return to main get_item

        else:
            natural_audio_feats = self.data[idx]['natural_audio_feats']
            print('No audio, feats shape and sum:', natural_audio_feats.shape, natural_audio_feats.sum())
        """

        total_time = time.time()

        # print("Time to process videos", video_time - start_time)
        # print("Time to load all data", total_time - start_time)

        if video_unpooled != None:
            return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': caption_audio_feats, 'natural_audio': natural_audio_feats,
                'nframes': nframes, 'task': task, 'start': start, 'end': end, 
                'vid_id': vid_id, 'text_sim': text_sim,
                'has_audio':self.data[idx]['has_audio'], 'video_unpooled':video_unpooled}
        else:
            return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': caption_audio_feats, 'natural_audio': natural_audio_feats,
                'nframes': nframes, 'task': task, 'start': start, 'end': end, 
                'vid_id': vid_id, 'text_sim': text_sim, 'has_audio':self.data[idx]['has_audio']}
