This repo has the implementation of our Interspeech 2021 papers: [AVLnet: Learning Audio-Visual Language Representations from Instructional Videos](https://arxiv.org/abs/2006.09199) [1] and [Cascaded Multilingual Audio-Visual Learning from Videos](https://arxiv.org/abs/2111.04823) [2]. Our website [avlnet.csail.mit.edu](avlnet.csail.mit.edu) has an audio-video retrieval demo.

AVLnet (Audio-Video Language Network) is trained on the audio-video pairs from the HowTo100M dataset, and can be used for video clip retrieval using raw speech audio and natural sounds, without needing to transcribe speech to text. AVLnet-Text integrates a text branch and is trained on audio, video, and text from the HowTo100M dataset. It can be used for text to video retrieval on standard video and language datasets. We propose two versions of the model, AVLnet-Text-Tri which keeps the three branches separate so that any two modalities can be compared, and AVLnet-Text-Fused which fuses the audio and text branches due to the complementary information in audio and text.

To learn multilingual representations, we propose a cascaded approach that applies the AVLnet model trained on English videos to videos in Japanese. We collected a dataset of instructional cooking videos
in Japanese, named YouCook-Japanese. Applying our cascaded approach, we show an improvement in retrieval performance of nearly 10x on YouCook-Japanese compared to training on the Japanese videos solely.

## Instructions

In this repo, we provide everything necessary to evaluate and fine-tune our models already trained on HowTo100M (pretrained weights provided). The instructions for training on HowTo100M are in training.md. 

Currently, we provide:
- Code, model weights, and data to evaluate AVLnet on YouCook2, MSR-VTT, CrossTask, and YouCook-Japanese [2].
- Code, model weights, and data to evaluate AVLnet-Text on YouCook2 and MSR-VTT. 
- Code to train both AVLnet and AVLnet-Text on HowTo100M.
## Requirements
We recommend installing the following packages in a fresh anaconda environment. Note that the evaluation code will run without Librosa and Apex. Our training code will also run without Apex, but we have only tested it using Apex with mixed-precision.  
- Python 3
- PyTorch (tested with 1.3 and 1.4)
- NumPy 
- SciPy
- Gensim
- TQDM
- Librosa
- NVIDIA Apex (for mixed-precision training on HowTo100M) (https://github.com/NVIDIA/apex)
## Download the Model Weights and Data
- Download the model weights and datafiles [here](https://www.dropbox.com/sh/bd75sz4m734xs0z/AADbN9Ujhn6FZX12ulpNWyR_a?dl=0) using the following code.
```
wget https://www.dropbox.com/sh/bd75sz4m734xs0z/AADGydRa_0QClNmGXEtBOoKca/AVLnet_release_models.tar.gz?dl=0
tar -xvf 'AVLnet_release_models.tar.gz?dl=0'
mkdir model
mv AVLnet_release model
wget https://www.dropbox.com/sh/bd75sz4m734xs0z/AADUY_-IqGWx9NiiXb6ae304a/AVLnet_release_data.tar.gz?dl=0
tar -xvf 'AVLnet_release_data.tar.gz?dl=0'
wget https://www.dropbox.com/s/4tqokt8pp53gjjp/YouCook_Japanese.tar.gz?dl=0
tar -xvf 'YouCook_Japanese.tar.gz?dl=0' && mv YouCook_Japanese data
```
- The datafiles are pickle files that contain the audio and video features. The audio features are spectrograms and the video features are 2d features from a ResNet-152 and 3d features from a ResNext-101. The code only requires a single, max-pooled visual feature vector from each clip to work. However, we've also included the visual features with full temporal resolution for YouCook2 and MSR-VTT.
- Please check the Appendix in [1] for the full details about the data splits we used. 
- For YouCook2, `youcook_train_audio.pkl` and `youcook_val_audio.pkl` contain the training and validation splits. For the validation split, there are 11 clips that do not have the video features with full temporal resolution. We also provide the Sounds-241 and Speech-241 splits (`youcook_val_sounds241.pkl`, `youcook_val_speech241.pkl`), where clips in Sounds-241 did not have any speech detected by an ASR system, and all clips in Speech-241 had at least one word detected. We used this splits to test how well AVLnet could perform retrieval on speech versus sounds.  These splits **should not be used as a benchmark to compare with AVLnet**, but rather as a diagnostic tool to study new models.
- For MSR-VTT, `msrvtt_train.pkl` contains the data from the split of 7k training clips proposed by Miech et al. in the [HowTo100M paper.](https://arxiv.org/pdf/1906.03327.pdf) There are only 6,783 training clips with audio. The test set `msrvtt_jsfusion_test.pkl` contains the test split of 1k clips. Only 968 clips had audio, so we count the 32 test clips without audio as mistakes in our retrieval calculations for a fair comparison. Finally, we also provide the data from the "1k-A" training split with 9k clips in `msrvtt_1k-A_train.pkl`. Our results our reported with the 7k split, but we include this split too in case others want to try it (you can use it with the `--msrvtt_train_path` flag). Luo et al. explain the different splits well in [CLIP4Clip](https://arxiv.org/pdf/2104.08860.pdf).
- For YouCook-Japanese, we also provide the extracted features. The list of videos is provided in `youcook_japanese_v1.json` and the clip info (start and end times) are in the corresponding pickle files.
## General Code Notes
- To evaluate language retrieval instead of video clip retrieval, add the following flag: `--eval_lang_retrieval=1`
- We generally used between 1 to 4 V100 GPUs with 32 GB each. If your GPUs don't have enough memory for the fine-tuning, you can reduce `--batch_size`, `--*_num_frames_multiplier` (check args.py), however your performance may differ from ours and you may need to re-tune hyperparameters. 
- The error `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR` can be resolved by fine-tuning with more GPUs.
- We have tested the code on different machines and the results can vary slightly.
- We use the dataloader for youcook (youcook.py) for the CrossTask and YouCook-Japanese data too.
## AVLnet Training and Evaluation on YouCook2, MSR-VTT, CrossTask, and YouCook-Japanese

### YouCook2
Evaluate and fine-tune the HowTo100M-trained model on YouCook2:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=256 --epochs=5 --lr_decay=1.0 --embd_dim=4096  --pretrain_path=model/AVLnet_release/AVLnet_release.pth
```

Train from scratch on YouCook2:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=64 --epochs=15 --lr=1e-4 --lr_decay=1.0 --embd_dim=4096
```


### MSR-VTT
Evaluate and fine-tune the HowTo100M-trained model on MSR-VTT:
```
python train.py --msrvtt=1 --eval_msrvtt=1 --num_thread_reader=8 --batch_size=256 --epochs=5 --lr_decay=1.0 --embd_dim=4096  --pretrain_path=model/AVLnet_release/AVLnet_release.pth
```

Train from scratch on MSR-VTT:
```
python train.py --msrvtt=1 --eval_msrvtt=1 --num_thread_reader=8 --batch_size=64 --epochs=15 --lr_decay=1.0
```

### CrossTask
Evaluate and fine-tune the HowTo100M-trained model on CrossTask:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=256 --epochs=5 --lr_decay=1.0 --embd_dim=4096  --pretrain_path=model/AVLnet_release/AVLnet_release.pth --youcook_train_path=data/crosstask_clips_train.pkl --youcook_val_path=data/crosstask_clips_val.pkl
```

Train from scratch on CrossTask:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=64 --epochs=15 --lr=1e-4 --lr_decay=1.0 --embd_dim=4096 --youcook_train_path=data/crosstask_clips_train.pkl --youcook_val_path=data/crosstask_clips_val.pkl
```

### YouCook-Japanese
Evaluate and fine-tune the HowTo100M-trained model on YouCook-Japanese:
(Note: the validation set is provided as `youcook_japanese_val.pkl` and should be used for hyperparameter tuning)
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=256 --epochs=5 --lr_decay=1.0 --embd_dim=4096 --pretrain_path=model/AVLnet_release/AVLnet_release.pth --youcook_train_path=data/YouCook_Japanese/youcook_japanese_train.pkl --youcook_val_path=data/YouCook_Japanese/youcook_japanese_eval.pkl  
```

Train from scratch on YouCook-Japanese:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=64 --epochs=15 --lr_decay=1.0 --embd_dim=4096 --lr=1e-4 --youcook_train_path=data/YouCook_Japanese/youcook_japanese_train.pkl --youcook_val_path=data/YouCook_Japanese/youcook_japanese_eval.pkl  
```

## AVLnet-Text Evaluation and Fine-tuning
**Please see our paper for the difference between AVLnet-Text-Tri and AVLnet-Text-Fused.**
AVLnet-Text-Tri performs **T->A+V** retrieval and AVLnet-Text-Fused performs **T+A->V** retrieval.

### AVLnet-Text-Tri
Note the `--fuse_videoaudio_additive=1` flag (check args.py for details). 

Evaluate and fine-tune the HowTo100M-trained model on YouCook2:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=256 --epochs=3 --tri_modal=1 --fuse_videoaudio_additive=1 --lr=1e-4 --lr_decay=0.9 --embd_dim=6144 --pretrain_path=model/AVLnet_release/AVLnet_Text_Tri_release.pth 
```

Evaluate and fine-tune the HowTo100M-trained model on MSR-VTT:
```
python train.py --msrvtt=1 --eval_msrvtt=1 --num_thread_reader=8 --batch_size=256 --epochs=15 --tri_modal=1 --fuse_videoaudio_additive=1 --lr=1e-4 --lr_decay=1.0 --embd_dim=6144 --pretrain_path=model/AVLnet_release/AVLnet_Text_Tri_release.pth 
```

### AVLnet-Text-Fused

Evaluate and fine-tune the HowTo100M-trained model on YouCook2:
```
python train.py --youcook=1 --eval_youcook=1 --num_thread_reader=8 --batch_size=256 --epochs=5 --lr_decay=1.0 --embd_dim=4096  --pretrain_path=model/AVLnet_release/AVLnet_Text_Fused_release.pth --lr=1e-5 --tri_modal_fuse=1 --tri_modal=1
```

Evaluate and fine-tune the HowTo100M-trained model on MSR-VTT:
```
python train.py --msrvtt=1 --eval_msrvtt=1 --num_thread_reader=8 --batch_size=256 --epochs=5 --lr_decay=1.0 --embd_dim=4096  --pretrain_path=model/AVLnet_release/AVLnet_Text_Fused_release.pth --lr=1e-5 --tri_modal_fuse=1 --tri_modal=1
```

## Use the model on your own videos
- If you want to use smaller clips within your videos, you'll have to split them up at some point. You can do that either before you extract the features, or after. We recommend the former.
- You can use the functions in `audio_to_spectrograms.py` to extract audio, and then spectrograms, from videos.
- You can use [this video feature extractor](https://github.com/roudimit/video_feature_extractor) to extract the visual features. We used the default settings. The model weights for the ResNext-101 are provided in the AVLnet model weights folder.
- We recommend extracting all of the features first, creating a pickle file to match the one for YouCook2, and then modifying the fine-tuning commands for YouCook2.
- Specifically, the pickle file should be a Python list, where each element is a dictionary and represents a video clip. The keys of each dictionary should be `'2d'` which maps to a single feature vector representing the pooled 2d features for the entire clips (same for the `'3d'` key), and `'audio'` which maps to the spectrogram for the clip. Please review `youcook_train_audio.pkl` and `youcook_val_audio.pkl` for more details.

## References

[1] Andrew Rouditchenko*, Angie Boggust*, David Harwath, Brian Chen, Dhiraj Joshi, Samuel Thomas, Kartik Audhkhasi, Hilde Kuehne, Rameswar Panda, Rogerio Feris, Brian Kingsbury, Michael Picheny, Antonio Torralba, James Glass. [AVLnet: Learning Audio-Visual Language Representations from Instructional Videos](https://arxiv.org/abs/2006.09199). Interspeech 2021.

[2] Andrew Rouditchenko, Angie Boggust, David Harwath, Samuel Thomas, Hilde Kuehne, Brian Chen, Rameswar Panda, Rogerio Feris, Brian Kingsbury, Michael Picheny, James Glass. [Cascaded Multilingual Audio-Visual Learning from Videos](https://arxiv.org/abs/2111.04823). Interspeech 2021.

AVLnet - Bibtex:
```bibtex
@article{rouditchenko2020avlnet,
  title={Avlnet: Learning audio-visual language representations from instructional videos},
  author={Rouditchenko, Andrew and Boggust, Angie and Harwath, David and Chen, Brian and Joshi, Dhiraj and Thomas, Samuel and Audhkhasi, Kartik and Kuehne, Hilde and Panda, Rameswar and Feris, Rogerio and others},
  journal={arXiv preprint arXiv:2006.09199},
  year={2020}
}
```

Cascaded Multilingual - Bibtex:
```bibtex
@article{rouditchenko2021cascaded,
  title={Cascaded Multilingual Audio-Visual Learning from Videos},
  author={Rouditchenko, Andrew and Boggust, Angie and Harwath, David and Thomas, Samuel and Kuehne, Hilde and Chen, Brian and Panda, Rameswar and Feris, Rogerio and Kingsbury, Brian and Picheny, Michael and others},
  journal={Proc. Interspeech 2021},
  pages={3006--3010},
  year={2021}
}
```

## Contact
If You find any problems or have any questions, please open an issue and I will try to respond as soon as possible.
## Acknowledgments and Licenses
The main structure of our code is adopted from Antoine Miech's original HowTo100M training code (https://github.com/antoine77340/howto100m). All code derived from there is licensed under Apache License 2.0 (Antoine Miech).

The code in `model_davenet.py` is partly derived from https://github.com/dharwath/DAVEnet-pytorch/ and https://github.com/wnhsu/ResDAVEnet-VQ and is licensed under BSD-3 (David Harwath and Wei-Ning Hsu).

All other code is licensed under BSD-3 (Andrew Rouditchenko).

All license clauses are in the LICENSE file.