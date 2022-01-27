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
        end_time = time.time()
        # load 2d and 3d features (features are max-pooled over the time dimension)
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
        # print('feat_2d shape', feat_2d.shape)
        # print('feat 3d shape', feat_3d.shape)
        video = th.cat((feat_2d, feat_3d))
        # print('video shape', video.shape)
        vid_time = time.time()
        unpooled2d = F.normalize(th.from_numpy(self.data[idx]['2d_full']).float(), dim=0)
        unpooled3d = F.normalize(th.from_numpy(self.data[idx]['3d_full']).float(), dim=0)
        # print('2d unpooled shape', unpooled2d.shape)
        # print('3d unpooled shape', unpooled3d.shape)
        tokens2d = unpooled2d.shape[0]
        tokens3d = unpooled3d.shape[0]
        if unpooled2d.shape[0] < 4:
            # print('Adding', 3-unpooled2d.shape[0], '2d tokens')
            unpooled2d = F.pad(unpooled2d, (0,0,0,4-unpooled2d.shape[0]), 'constant', 0)
        elif unpooled2d.shape[0] > 4:
            # print('Truncating', unpooled2d.shape[0]-4, '2d tokens')
            unpooled2d = unpooled2d[:4]
        if unpooled3d.shape[0] < 6:
            # print('Adding', 5-unpooled3d.shape[0], '3d tokens')
            unpooled3d = F.pad(unpooled3d, (0,0,0,6-unpooled3d.shape[0]), 'constant', 0)
        elif unpooled3d.shape[0] > 6:
            # print('Truncating', unpooled3d.shape[0]-6, '3d tokens')
            unpooled3d = unpooled3d[:6]
        video_unpooled = th.cat((unpooled2d, unpooled3d))
        if video_unpooled.shape[0] != 10:
            print('2d shape', unpooled2d.shape)
            print('3d shape', unpooled3d.shape)
        # print('video_unpooled shape', video_unpooled.shape)
        unpooled_time = time.time()
        # load audio and zero pad/truncate if necessary
        """
        caption_audio_file = self.data[idx]['spoken_caption_path']
        audio_to_spectrograms.extract_audio(caption_audio_file, 'temp1.wav', sys.stdout)
        audio_to_spectrograms.stereo_to_mono_downsample('temp1.wav', 'temp2.wav', 48000)
        feats, frames = audio_to_spectrograms.LoadAudio('temp2.wav')
        caption_audio_feats = feats
        os.remove('temp1.wav')
        os.remove('temp2.wav')
        
        Commenting out to load audio features directly from pickle file
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
        caption_time = time.time()
        natural_audio_feats = th.zeros((40,3072),dtype=th.float) # should this be an empty tensor? Yes. what shape?
        if self.data[idx]['has_audio']:
            try:
                natural_audio_file = '/nobackup/users/lynberry/natural_audio/'+self.data[idx]['id']+'.wav'
                feats, frames = audio_to_spectrograms.LoadAudio(natural_audio_file, use_raw_length=True)
                natural_audio_feats = feats
    
                target_length = 1024 * 1 # All clips are 3s long, so no self.num_frames_multiplier
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
        nat_time = time.time()
        """

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

        # assert caption_audio_feats.shape[1] == 2048
        # assert natural_audio_feats.shape[1] == 1024
        # print("caption_audio_feats.shape", caption_audio_feats.shape)
        # print("natural_audio_feats.shape", natural_audio_feats.shape)
        # done_time = time.time()
        # print("Times taken...")
        # print("\tVideo processing:\t", vid_time-end_time)
        # print("\tUnpooled video processing:\t", unpooled_time-vid_time)
        # print("\tCaption processing:\t", caption_time-unpooled_time)
        # print("\tNatural audio processing:\t", nat_time-caption_time)
        # print("\tTotal time:\t", done_time-end_time)
        
        caption_audio_feats = self.data[idx]['caption_audio_feats']
        natural_audio_feats = self.data[idx]['natural_audio_feats']

        # print('caption shape before selecting random 10s', caption_audio_feats.shape)
        # print('natural shape', natural_audio_feats.shape)

        #try:
        nframes = len(self.data[idx]['nframes'])
        #except:
        #    nframes = 0

        # Adding 10s extraction from captions
        if nframes > 1024:
            start_index = random.randint(0,min(nframes,2048)-1024)
        else:
            start_index = 0

        caption_audio_feats = caption_audio_feats[:,:,start_index:start_index+1024]

        if int(caption_audio_feats.shape[2]) != 1024:
            print('wrong length!')
            print('num frames', nframes)
            print('start_index', start_index)
            print('stop_index', stop_index)

        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': caption_audio_feats, 'natural_audio': natural_audio_feats,
                'nframes': nframes, 'task': task, 'start': start, 'end': end, 
                'vid_id': vid_id, 'text_sim': text_sim, 'video_unpooled': video_unpooled, 
                'has_audio':self.data[idx]['has_audio'], 'tokens2d':tokens2d, 'tokens3d':tokens3d}
        
