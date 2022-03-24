from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import random
import math
import os
import time
import pickle
import numpy as np
from args import get_args
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from gensim.models.keyedvectors import KeyedVectors

import torch as th
th.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
import torch.optim as optim

from youtube_dataloader import Youtube_DataLoader
from youcook_dataloader import Youcook_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from smit_dataloader import SMiT_DataLoader
from model import Net
from loss import MMS_loss
from loss import AMM_loss
from metrics import compute_metrics, print_computed_metrics, AverageMeter

# from pytorch_memlab import LineProfiler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='/nobackup/users/lynberry/logs')

args = get_args()
if args.verbose:
    print(args)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

caption = None
if not(args.youcook) and not(args.msrvtt):
    if not args.random_audio_windows:
        print('Loading HowTo100M captions: {}'.format(args.caption_path))
        caption = pickle.load(open(args.caption_path, 'rb'))
        print('done')

we = None
if args.tri_modal or not args.random_audio_windows:
    print('Loading word vectors: {}'.format(args.word2vec_path))
    we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    print('done')

if args.youcook:
    dataset = Youcook_DataLoader(
        data=args.youcook_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.youcook_num_frames_multiplier,
        tri_modal=args.tri_modal,
    )
elif args.msrvtt:
    dataset = MSRVTT_DataLoader(
        data_path=args.msrvtt_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=True,
        tri_modal=args.tri_modal,
    )
elif args.smit:
    dataset = SMiT_DataLoader(
        data_path=args.smit_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.smit_num_frames_multiplier,
        tri_modal=args.tri_modal,
        load_images=args.load_images,
    )
else:
    dataset = Youtube_DataLoader(
        csv=args.train_csv,
        features_path=args.features_path,
        features_path_audio=args.features_path_audio,
        caption=caption,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        we=we,
        we_dim=args.we_dim,
        n_pair=args.n_pair,
        num_audio_frames=args.howto_audio_frames,
        random_audio_windows=args.random_audio_windows,  
    )

dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_thread_reader,
    shuffle=True,
    batch_sampler=None,
    drop_last=True,
)
if args.eval_youcook:
    dataset_val = Youcook_DataLoader(
        data=args.youcook_val_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.youcook_num_frames_multiplier,
        tri_modal=args.tri_modal,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_msrvtt:
    msrvtt_testset = MSRVTT_DataLoader(
        data_path=args.msrvtt_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=False,
        tri_modal=args.tri_modal,   
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
if args.eval_smit:
    smit_valset = SMiT_DataLoader(
        data_path=args.smit_val_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.smit_num_frames_multiplier,
        tri_modal=args.tri_modal,
        load_images=args.load_images,
    )
    dataloader_smit = DataLoader(
        smit_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.test_smit:
    smit_testset = SMiT_DataLoader(
        data_path=args.smit_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.smit_num_frames_multiplier,
        tri_modal=args.tri_modal,
        load_images=args.load_images,
    )
    dataloader_test_smit = DataLoader(
        smit_testset,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
#with LineProfiler(Net) as prof:
net = Net(
    embd_dim=args.embd_dim,
    video_dim=args.feature_dim,
    we_dim=args.we_dim,
    tri_modal=args.tri_modal,
    tri_modal_fuse=args.tri_modal_fuse,
    natural_audio=args.natural_audio,
    two_level=args.two_level,
    use_ast=args.ast,
    extra_terms=args.extra_terms,
    load_images=args.load_images,
    k_2d=args.train_top_k_2d,
    k_3d=args.train_top_k_3d,
)
# print('Should display here')
# print(prof.display())
# print('done')

# Optimizers + Loss
if args.loss == 0:
    loss_op = MMS_loss(args.one_way)
elif args.loss == 1:
    loss_op = AMM_loss(args.one_way)
net.cuda()
loss_op.cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
if args.apex_level == 0:
    apex = False
elif args.apex_level == 1:
    from apex import amp, optimizers
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    apex = True
net = th.nn.DataParallel(net)
net.train()

if args.pretrain_path != '' and args.apex_level == 1:
    amp_checkpoint_path = os.path.join(os.path.dirname(args.pretrain_path), 'amp_checkpoint.pt')
    checkpoint = th.load(amp_checkpoint_path, map_location='cpu')
    net.module.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    print("Loaded AMP checkpoint")
elif args.pretrain_path != '' and args.apex_level == 0:
    net.module.load_checkpoint(args.pretrain_path) # made non-strict

if args.verbose:
    print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun, apex=False, use_natural_audio=False):
    model.cuda()
    video = data['video'].cuda()
    try:
        video_unpooled = data['video_unpooled'].cuda()
    except:
        video_unpooled = None
    audio = data['audio'].cuda()
    if use_natural_audio:
        natural_audio = data['natural_audio'].cuda()
        natural_audio = natural_audio.view(-1, natural_audio.shape[-2], natural_audio.shape[-1])
    else:
        natural_audio = None
    nframes = data['nframes'].cuda()
    # video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    nframes = nframes.view(-1)
    opt.zero_grad()
    with th.set_grad_enabled(True):
        if args.tri_modal:
            text = data['text'].cuda()
            text = text.view(-1, text.shape[-2], text.shape[-1])
            if args.tri_modal_fuse: # AVLnet-Text audio-text fusion model
                audio_text, video = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                sim_audiotext_video = th.matmul(audio_text, video.t())
                loss = loss_fun(sim_audiotext_video)
            else: # AVLnet-Text independent audio and text branches
                audio, video, text = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                if args.fuse_videoaudio_additive: # only used for fine-tuning
                    audio_video = audio + video
                    sim_text_audiovideo = th.matmul(text, audio_video.t())
                    loss = loss_fun(sim_text_audiovideo)
                else:
                    sim_audio_video = th.matmul(audio, video.t())
                    sim_audio_text = th.matmul(audio, text.t())
                    sim_text_video = th.matmul(text, video.t())
                    loss = loss_fun(sim_audio_video) + loss_fun(sim_audio_text) + loss_fun(sim_text_video)
        else:
            if args.extra_terms:
                audio, video, visual_only, audio_only = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                sim_matrix_main = th.matmul(audio, video.t())
                sim_matrix_visual = th.matmul(audio, visual_only.t())
                sim_matrix_audio = th.matmul(audio, audio_only.t())
                loss = .5 * loss_fun(sim_matrix_main) + .25 * loss_fun(sim_matrix_visual) + .25 * loss_fun(sim_matrix_audio, data['has_audio'])
            else:
                audio, video = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])            
                sim_matrix = th.matmul(audio, video.t())
                loss = loss_fun(sim_matrix)
    # print('Immediately before loss.backward(), memory summary is...')
    # print(th.cuda.memory_summary())
 
    if apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    opt.step()
    return loss.item()

def Eval_retrieval(model, eval_dataloader, dataset_name, use_natural_audio=False):
    model.eval()
    # model_eval = model.module.to('cpu')
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        mmin1 = th.empty(0,args.embd_dim) # matmul input 1
        mmin2 = th.empty(0,args.embd_dim) # matmul input 2
        for data in eval_dataloader:
            th.cuda.empty_cache()
            video = data['video'].cuda()#.cpu().detach()
            try:
                video_unpooled = data['video_unpooled'].cuda()#.cpu().detach()
            except:
                video_unpooled = None 
            audio = data['audio'].cuda()#.cpu().detach()
            if use_natural_audio:
                natural_audio = data['natural_audio'].cuda()#.cpu().detach()
                natural_audio = natural_audio.view(-1, natural_audio.shape[-2], natural_audio.shape[-1])
            else:
                natural_audio = None
            nframes = data['nframes'].cuda()#.cpu().detach()
            if args.tri_modal:
                text = data['text'].cuda()#.cpu().detach()
                if args.tri_modal_fuse: # AVLnet-Text
                    audio_text, video = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=['tokens2d'], tokens3d=['tokens3d'])
                    in1 = audio_text
                    in2 = video
                    # ABOUT TO MATMUL AUDIO_TEXT AND VIDEO
                    m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                    # DONE WITH MATMUL, RESULT IS M
                elif args.fuse_videoaudio_additive: # eval T->V+A for AVLnet-Text indep. model
                    audio, video, text = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    audio_video = audio + video
                    in1 = text
                    in2 = audio_video
                    # ABOUT TO MATMUL TEXT AND AUDIO_VIDEO
                    m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
                    # DONE WITH MATMUL, RESULT IS M
            else:
                if args.extra_terms:
                    audio, video, visual_only, audio_only = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                else:
                    audio, video = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                in1 = audio
                in2 = video
                # ABOUT TO MATMUL AUDIO AND VIDEO
                m = th.matmul(audio, video.t()).cpu().detach().numpy()
                # DONE WITH MATMUL, RESULT IS M
                #print('shape of inputs', in1.shape, in2.shape)
            mmin1 = th.cat((mmin1, in1.cpu().detach()), dim=0)
            mmin2 = th.cat((mmin2, in2.cpu().detach()), dim=0)
        # No longer in for loop
        m = th.matmul(mmin1, mmin2.t()).numpy() # May need to detach before
        metrics, ind = compute_metrics(m, args.eval_lang_retrieval, args.eval_msrvtt)
        print_computed_metrics(metrics)

def TrainOneBatchMP(model, opt, data, loss_fun, apex=False, use_natural_audio=False):
    scaler = GradScaler()

    model.cuda()
    video = data['video'].cuda()
    try:
        video_unpooled = data['video_unpooled'].cuda()
    except:
        video_unpooled = None
    audio = data['audio'].cuda()
    if use_natural_audio:
        natural_audio = data['natural_audio'].cuda()
        natural_audio = natural_audio.view(-1, natural_audio.shape[-2], natural_audio.shape[-1])
    else:
        natural_audio = None
    nframes = data['nframes'].cuda()
    # video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    nframes = nframes.view(-1)
    opt.zero_grad()
    with th.set_grad_enabled(True):
        with autocast():
            if args.tri_modal:
                text = data['text'].cuda()
                text = text.view(-1, text.shape[-2], text.shape[-1])
                if args.tri_modal_fuse: # AVLnet-Text audio-text fusion model
                    audio_text, video = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    sim_audiotext_video = th.matmul(audio_text, video.t())
                    loss = loss_fun(sim_audiotext_video)
                else: # AVLnet-Text independent audio and text branches
                    audio, video, text = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    if args.fuse_videoaudio_additive: # only used for fine-tuning
                        audio_video = audio + video
                        sim_text_audiovideo = th.matmul(text, audio_video.t())
                        loss = loss_fun(sim_text_audiovideo)
                    else:
                        sim_audio_video = th.matmul(audio, video.t())
                        sim_audio_text = th.matmul(audio, text.t())
                        sim_text_video = th.matmul(text, video.t())
                        loss = loss_fun(sim_audio_video) + loss_fun(sim_audio_text) + loss_fun(sim_text_video)
            else:
                if args.extra_terms:
                    audio, video, visual_only, audio_only = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    sim_matrix_main = th.matmul(audio, video.t())
                    sim_matrix_visual = th.matmul(audio, visual_only.t())
                    sim_matrix_audio = th.matmul(audio, audio_only.t())
                    loss = .5 * loss_fun(sim_matrix_main) + .25 * loss_fun(sim_matrix_visual) + .25 * loss_fun(sim_matrix_audio, data['has_audio'])
                else:
                    audio, video = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])            
                    sim_matrix = th.matmul(audio, video.t())
                    loss = loss_fun(sim_matrix)
    # print('Immediately before loss.backward(), memory summary is...')
    # print(th.cuda.memory_summary())
 
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    return loss.item()


def Eval_retrieval_stable(model, eval_dataloader, dataset_name, use_natural_audio=False):
    model.eval()
    model_eval = model.module.to('cpu')
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for data in eval_dataloader:
            th.cuda.empty_cache()
            video = data['video'].cpu().detach()# .cuda()
            try:
                video_unpooled = data['video_unpooled'].cpu().detach()# .cuda()
            except:
                video_unpooled = None
            audio = data['audio'].cpu().detach()# .cuda()
            if use_natural_audio:
                natural_audio = data['natural_audio'].cpu().detach()# .cuda()
                natural_audio = natural_audio.view(-1, natural_audio.shape[-2], natural_audio.shape[-1])
            else:
                natural_audio = None
            nframes = data['nframes'].cpu().detach()# .cuda()
            if args.tri_modal:
                text = data['text'].cpu().detach()# .cuda()
                if args.tri_modal_fuse: # AVLnet-Text
                    audio_text, video = model_eval(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=['tokens2d'], tokens3d=['tokens3d'])
                    m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                elif args.fuse_videoaudio_additive: # eval T->V+A for AVLnet-Text indep. model
                    audio, video, text = model_eval(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    audio_video = audio + video
                    m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
            else:
                audio, video = model_eval(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                m = th.matmul(audio, video.t()).cpu().detach().numpy()
            metrics = compute_metrics(m, args.eval_lang_retrieval, args.eval_msrvtt)
            print_computed_metrics(metrics)

def test_smit_retrieval(model, test_dataloader, dataset_name, use_natural_audio=False, out_best=None, out_worst=None):
    model.eval()
    print('Evaluating on test set, averaged over 5 random batches of 1000')
    with th.no_grad():
        overall_metrics = {'R1':0, 'R5':0, 'R10':0, 'MR':0}
        for i in range(5):
            data = next(iter(test_dataloader))
            th.cuda.empty_cache()
            video = data['video'].cuda()
            try:
                video_unpooled = data['video_unpooled'].cuda()
            except:
                video_unpooled = None
            audio = data['audio'].cuda()
            if use_natural_audio:
                natural_audio = data['natural_audio'].cuda()
                natural_audio = natural_audio.view(-1, natural_audio.shape[-2], natural_audio.shape[-1])
            else:
                natural_audio = None
            nframes = data['nframes'].cuda()
            if args.tri_modal:
                text = data['text'].cuda()
                if args.tri_modal_fuse:
                    audio_text, video = model(video, audio, nframes, text=text, natural_audio_input=nautral_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                elif args.fuse_videoaudio_additive:
                    audio, video, text = model(video, audio, nframes, text=text, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                    audio_video = audio + video
                    m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
            else:
                audio, video = model(video, audio, nframes, natural_audio_input=natural_audio, video_unpooled=video_unpooled, has_audio=data['has_audio'])#, tokens2d=data['tokens2d'], tokens3d=data['tokens3d'])
                m = th.matmul(audio, video.t()).cpu().detach().numpy()
            metrics, ind = compute_metrics(m, args.eval_lang_retrieval, args.eval_msrvtt)
            best_ids = np.flatnonzero(ind == ind.min())
            worst_ids = np.flatnonzero(ind >900)
            print(np.array(data['video_id'])[best_ids], file=out_best)
            print(np.array(data['video_id'])[worst_ids], file=out_worst)
            for k in metrics.keys():
                overall_metrics[k] += metrics[k]
        for k in overall_metrics.keys():
            overall_metrics[k] /= 5.0
        print_computed_metrics(overall_metrics)

batch_time = AverageMeter()
data_time = AverageMeter()
out_best = open('video_IDs_correctly_matched.txt', 'w')
out_worst = open('video_IDs_extremely_incorrect.txt', 'w')
for epoch in range(args.epochs):
    th.cuda.empty_cache()
    print(th.cuda.memory_summary())
    running_loss = 0.0
    if args.test_smit:
        test_smit_retrieval(net, dataloader_test_smit, 'S-MiT', args.natural_audio, out_best, out_worst)
    if args.eval_youcook:
        Eval_retrieval(net, dataloader_val, 'YouCook2')
    if args.eval_msrvtt:
        Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
    if args.eval_smit:
        Eval_retrieval(net, dataloader_smit, 'S-MiT', args.natural_audio)
    if args.verbose:
        print('Epoch: %d' % epoch)
    end_time = time.time()
    for i_batch, sample_batch in enumerate(tqdm(dataloader)):
        data_load_time = time.time() - end_time
        data_time.update(data_load_time)
        if args.use_amp:
            batch_loss = TrainOneBatchMP(net, optimizer, sample_batch, loss_op, apex, args.natural_audio)
        else:
            batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss_op, apex, args.natural_audio)
        # writer.add_scalar('Loss/attention_MMS_2heads_train_all', batch_loss, i_batch+1 + (epoch+4) * math.ceil(dataset_size / args.batch_size))
        process_batch_time = time.time() - end_time
        batch_time.update(process_batch_time)
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose:
            print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
            (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
            running_loss / args.n_display))
            # writer.add_scalar('Loss/with_audio_frozen_features', running_loss / args.n_display, i_batch+1 + (epoch+30) * math.ceil(dataset_size/args.batch_size))
            print('Batch load time avg: %.4f, Batch process time avg: %.4f' %
            (data_time.avg, batch_time.avg))
            running_loss = 0.0
            # reset the load meters
            batch_time = AverageMeter()
            data_time = AverageMeter()
        end_time = time.time()
    try:
        save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5]\
                else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
    except:
        save_epoch = epoch + 1 # If using overwritten checkpoint
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay
    if args.checkpoint_dir != '' and save_epoch % args.checkpoint_interval == 0:
        path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(save_epoch))
        net.module.save_checkpoint(os.path.join(args.checkpoint_dir, 'fully_trainable_take_two.pth'))
        if args.apex_level == 1:
            amp_checkpoint = {'net': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()}
            th.save(amp_checkpoint, os.path.join(args.checkpoint_dir, 'amp_checkpoint.pt'))

if args.eval_youcook:
    Eval_retrieval(net, dataloader_val, 'YouCook2')
if args.eval_msrvtt:
    Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
if args.eval_smit:
    Eval_retrieval(net, dataloader_smit, 'S-MiT', args.natural_audio)
