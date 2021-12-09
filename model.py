from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F
from model_davenet import load_DAVEnet

class Net(nn.Module):
    def __init__(
            self,
            embd_dim=1024,
            video_dim=2048,
            we_dim=300,
            tri_modal=False,
            tri_modal_fuse=False,
            natural_audio=False,
            two_level=False,
    ):
        super(Net, self).__init__()
        self.DAVEnet = load_DAVEnet()
        self.DAVEnet_projection = nn.Linear(1024, embd_dim)
        self.GU_audio = Gated_Embedding_Unit(1024, 1024)
        if natural_audio:
            # self.GU_video = Gated_Embedding_Unit(video_dim+embd_dim, embd_dim)
            self.nat_DAVEnet = load_DAVEnet()
            self.nat_DAVEnet_projection = nn.Linear(1024, embd_dim)
            self.nat_GU_audio = Gated_Embedding_Unit(1024, 1024)
            if two_level:
                self.GU_visual = Gated_Embedding_Unit(video_dim, embd_dim)
                self.GU_video = Gated_Embedding_Unit(2*embd_dim, embd_dim)
            else:
                self.GU_video = Gated_Embedding_Unit(video_dim+embd_dim, embd_dim)
        else:
            self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)
        if tri_modal and not tri_modal_fuse:
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
            self.GU_text_captions = Gated_Embedding_Unit(embd_dim, embd_dim)
        elif tri_modal_fuse:
            self.DAVEnet_projection = nn.Linear(1024, embd_dim // 2)
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim // 2)
            self.GU_audio_text = Fused_Gated_Unit(embd_dim // 2, embd_dim)
        self.tri_modal = tri_modal
        self.tri_modal_fuse = tri_modal_fuse
        self.natural_audio = natural_audio
        self.two_level = two_level

    def save_checkpoint(self, path):
        th.save(self.state_dict(), path)
    
    def load_checkpoint(self, path):
        try:
            self.load_state_dict(th.load(path, map_location='cpu'))
        except Exception as e:
            print(e)
            print("IGNORING ERROR, LOADING MODEL USING STRICT=FALSE")
            self.load_state_dict(th.load(path, map_location='cpu'), strict=False)
        print("Loaded model checkpoint from {}".format(path))

    def forward(self, video, audio_input, nframes, text=None, natural_audio_input=None):
        if natural_audio_input != None:
            natural_audio = self.nat_DAVEnet(natural_audio_input)
            # pooling
            pooling_ratio = round(natural_audio_input.size(-1) / natural_audio.size(-1))
            nframes = nframes.float()
            nframes.div_(pooling_ratio)
            nframes = nframes.long()
            audioPoolfunc = th.nn.AdaptiveAvgPool2d((1,1))
            natural_audio_outputs = natural_audio.unsqueeze(2)
            pooled_natural_audio_outputs_list = []
            for idx in range(natural_audio.shape[0]):
                nF = max(1, nframes[idx])
                pooled_natural_audio_outputs_list.append(audioPoolfunc(natural_audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
            natural_audio = th.cat(pooled_natural_audio_outputs_list).squeeze(3).squeeze(2)
            # done pooling
            natural_audio = self.nat_GU_audio(natural_audio)
            natural_audio = self.nat_DAVEnet_projection(natural_audio)
            if self.two_level:
                visual = self.GU_visual(video)
                video = self.GU_video(th.cat(visual, natural_audio),dim=1))
            else:
                video = self.GU_video(th.cat((video, natural_audio),dim=1))
        else:
            video = self.GU_video(video)
        audio = self.DAVEnet(audio_input)
        #if not self.training: # controlled by net.train() / net.eval() (use for downstream tasks) 
        # Mean-pool audio embeddings and disregard embeddings from input 0 padding
        pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
        nframes = nframes.float()
        nframes.div_(pooling_ratio)
        nframes = nframes.long()
        audioPoolfunc = th.nn.AdaptiveAvgPool2d((1, 1))
        audio_outputs = audio.unsqueeze(2)
        pooled_audio_outputs_list = []
        for idx in range(audio.shape[0]):
            nF = max(1, nframes[idx])
            pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
        audio = th.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
        # else:
        #   audio = audio.mean(dim=2) # this averages features from 0 padding too

        if self.tri_modal_fuse:
            text = self.text_pooling_caption(text)
            audio = self.DAVEnet_projection(audio)
            audio_text = self.GU_audio_text(audio, text)
            return audio_text, video

        # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training        
        audio = self.GU_audio(audio) 
        audio = self.DAVEnet_projection(audio)
        if self.tri_modal and not self.tri_modal_fuse:
            text = self.GU_text_captions(self.text_pooling_caption(text))
            return audio, video, text
        return audio, video

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x

class Fused_Gated_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Fused_Gated_Unit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x

class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)

class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return th.max(x, dim=1)[0]  
