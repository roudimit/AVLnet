from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torchvision import transforms

from model_davenet import load_DAVEnet
from ast_models import ASTModel

import sys
sys.path.append('../')
from video_feature_extractor.model import GlobalAvgPool
from video_feature_extractor.model import load_extractor2d
from moments_models.models import load_extractor3d

# from pytorch_memlab import profile_every

class Net(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=2048,
            we_dim=300,
            tri_modal=False,
            tri_modal_fuse=False,
            natural_audio=False,
            two_level=False,
            use_ast=False,
            extra_terms=False,
            load_images=False,
            k_2d=0,
            k_3d=0,
    ):
        super(Net, self).__init__()

        # Load visual feature extractors
        if load_images:
            self.extractor2d = load_extractor2d(k_2d)
            self.extractor3d = load_extractor3d(k_3d)
        
            self.normalize_video = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            self.pool_extracted_features = GlobalAvgPool()

        self.DAVEnet = load_DAVEnet()
        self.DAVEnet_projection = nn.Linear(1024, embd_dim)
        self.GU_audio = TwoLayerProjection(1024, 1024)# Gated_Embedding_Unit(1024, 1024)
        if natural_audio:
            self.nat_DAVEnet = load_DAVEnet()
            self.nat_DAVEnet_projection = nn.Linear(1024, embd_dim)
            self.nat_GU_audio = TwoLayerProjection(1024, 1024)# Gated_Embedding_Unit(1024, 1024)
            if two_level:
                self.GU_visual = TwoLayerProjection(video_dim, embd_dim)# Gated_Embedding_Unit(video_dim, embd_dim)
                self.GU_video = TwoLayerProjection(2*embd_dim, embd_dim)# Gated_Embedding_Unit(2*embd_dim, embd_dim)
            else:
                self.GU_video = TwoLayerProjection(video_dim+embd_dim, embd_dim)# Gated_Embedding_Unit(video_dim+embd_dim, embd_dim)
        else:
            self.GU_video = TwoLayerProjection(2*video_dim, embd_dim)# Gated_Embedding_Unit(video_dim, embd_dim)
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
        self.extra_terms = extra_terms
        self.use_ast = use_ast
        self.load_images = load_images
        self.embd_dim = embd_dim
        self.video_dim = video_dim
        if use_ast:
            self.GU_query = TwoLayerProjection(embd_dim, video_dim)# Gated_Embedding_Unit(video_dim, embd_dim)
            self.AST = ASTModel(label_dim=video_dim, input_fdim=128, input_tdim=1024, audioset_pretrain=True)
            self.mha = nn.MultiheadAttention(embed_dim=video_dim, num_heads=2)#, batch_first=True)
            self.GU_video = TwoLayerProjection(video_dim, embd_dim)
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

    # @profile_every(1)
    def forward(self, video, audio_input, nframes, text=None, natural_audio_input=None, video_unpooled=None, has_audio=None, tokens2d=None, tokens3d=None):
        if self.load_images:
            #with th.no_grad():
            normalized_frames = self.normalize_video(video)
 
            # Sample only 3/8 frames for 2d extraction
            # print('normalized_frames shape', normalized_frames.shape)
            normalized_frames_short = normalized_frames[:,(0,3,6),:,:,:]
            # print('normalized_frames_short shape', normalized_frames_short.shape)

            normalized_frames_2d = normalized_frames_short.view(-1, normalized_frames_short.shape[2], normalized_frames_short.shape[3], normalized_frames_short.shape[4])
            # print('normalized_frames_2d shape', normalized_frames_2d.shape)
            
            features_2d = self.extractor2d(normalized_frames_2d)
            features_2d = features_2d.view(-1, 3, features_2d.shape[1])
            normalized_frames_3d = normalized_frames.permute(0,2,1,3,4)
            features_3d = self.extractor3d(normalized_frames_3d).squeeze()
            pooled_2d = th.mean(features_2d, dim=1)
            video = th.cat((pooled_2d, features_3d),dim=1)

        if natural_audio_input != None:
            if self.use_ast:
                natural_audio = self.AST(natural_audio_input).unsqueeze(0).float()
                video_unpooled = video_unpooled.permute(1,0,2)
                video_tokens = th.cat((natural_audio,video_unpooled))
                video_tokens = video_tokens
                query = self.GU_query(video).unsqueeze(0)
                attn_out, _ = self.mha(query, video_tokens, video_tokens)
                video = self.GU_video(attn_out.squeeze())
            else:
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
                    video = self.GU_video(th.cat((visual, natural_audio),dim=1))
                else:
                    video = self.GU_video(th.cat((video, natural_audio),dim=1))
        else:
            # print('Given video shape', video.shape)
            # print('with embd_dim', self.embd_dim, 'and video_dim', self.video_dim)
            video = self.GU_video(video)


        audio = self.DAVEnet(audio_input) 
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

        if self.tri_modal_fuse:
            text = self.text_pooling_caption(text)
            audio = self.DAVEnet_projection(audio)
            audio_text = self.GU_audio_text(audio, text)
            return audio_text, video

        # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training        
        audio = self.GU_audio(audio) 
        audio = self.DAVEnet_projection(audio).squeeze()
        if self.tri_modal and not self.tri_modal_fuse:
            text = self.GU_text_captions(self.text_pooling_caption(text))
            return audio, video, text
        if self.extra_terms:
            return audio, video, query.squeeze(), natural_audio.squeeze()

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

class TwoLayerProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TwoLayerProjection, self).__init__()
        self.fc1 = nn.Linear(in_dim, 2*in_dim)
        self.glu1 = nn.GLU()
        self.fc2 = nn.Linear(in_dim, 2*out_dim)
        self.glu2 = nn.GLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.glu1(x)
        x = self.fc2(x)
        x = self.glu2(x)
        return x
