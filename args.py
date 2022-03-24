import argparse

def get_args(description='Youtube-Text-Video'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--train_csv',
        type=str,
        default='data/HowTo100M_1166_videopaths.txt',
        help='train csv')
    parser.add_argument(
        '--features_path',
        type=str,
        default='parsed_videos/',
        help='path for visual features (2D, 3D) visual features')
    parser.add_argument(
        '--features_path_audio',
        type=str,
        default='',
        help='path for audio files (defaults to --features_path)')
    parser.add_argument(
        '--caption_path',
        type=str,
        default='data/caption.pickle',
        help='HowTo100M caption pickle file path')
    parser.add_argument(
        '--word2vec_path',
        type=str,
        default='data/GoogleNews-vectors-negative300.bin',
        help='word embedding path')
    parser.add_argument(
        '--pretrain_path',
        type=str,
        default='',
        help='pre train model path')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='',
        help='checkpoint model folder')
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--text_thresh', type=int, default=0,  
            help='If not 0, use text sim matching and set the threshold (nec. for youcook2)') 
    parser.add_argument('--eval_lang_retrieval', type=int, default=0,
                    help='if 1, eval language retrieval instead of video retrieval') 
    parser.add_argument('--tri_modal', type=int, default=0,
                            help='use vision, speech, and text')
    parser.add_argument('--tri_modal_fuse', type=int, default=0,
                            help='use speech and text features (tri_modal must be 1)')
    parser.add_argument('--fuse_videoaudio_additive', type=int, default=0,
                            help='eval T->A+V with tri-modal modal \
                                  set tri_modal=1, tri_modal_fuse=0')
    parser.add_argument('--natural_audio', type=int, default=0, help='use natural audio in video encoding')
    parser.add_argument('--two_level', type=int, default=0, help='use two-level fusion to integrate natural audio')
    parser.add_argument('--loss', type=int, default=0,
                                help='0 for Masked Margin Softmax (MMS) loss, 1 for Adaptive Mean Margin (AMM) loss')
    parser.add_argument('--one_way', type=int, default=0,
                                help='Compute loss only one direction (speech --> video)?')
    parser.add_argument('--extra_terms', type=int, default=0,
                                help='Add extra terms to the loss for just video and just audio')
    parser.add_argument('--ast', type=int, default=0, help='use AST to process natural audio')
    parser.add_argument('--apex_level', type=int, default=0,
                                help='Apex (mixed precision) level: chose 0 for none, 1 for O1.')
    parser.add_argument('--random_audio_windows', type=int, default=1,
                                help='1 to use random audio windows, 0 to use HowTo100M ASR clips')
    parser.add_argument('--howto_audio_frames', type=int, default=1024,
                            help='number of frames to use for loading howto100m audio')
    parser.add_argument('--youcook_num_frames_multiplier', type=int, default=5,
                                help='use 1024 * x audio frames for youcook2')
    parser.add_argument('--msrvtt_num_frames_multiplier', type=int, default=3,
                                help='use 1024 * x audio frames for msrvtt')
    parser.add_argument('--num_thread_reader', type=int, default=1,
                                help='')
    parser.add_argument('--embd_dim', type=int, default=4096,
                                help='embedding dim')
    parser.add_argument('--lr', type=float, default=0.0001,
                                help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                                help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256,
                                help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500,
                                help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                                help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=200,
                                help='Information display frequence')
    parser.add_argument('--feature_dim', type=int, default=2048,
                                help='video feature dimension')
    parser.add_argument('--we_dim', type=int, default=300,
                                help='word embedding dimension')
    parser.add_argument('--seed', type=int, default=1,
                                help='random seed')
    parser.add_argument('--verbose', type=int, default=1,
                                help='')
    parser.add_argument('--max_words', type=int, default=20,
                                help='')
    parser.add_argument('--min_words', type=int, default=0,
                                help='')
    parser.add_argument('--feature_framerate', type=int, default=1,
                                help='')
    parser.add_argument('--min_time', type=float, default=5.0,
                                help='Gather small clips')
    parser.add_argument('--n_pair', type=int, default=1,
                                help='Number of video clips to use per video') 
    parser.add_argument('--youcook', type=int, default=0,
                                help='Train on YouCook2 data')
    parser.add_argument('--msrvtt', type=int, default=0,
                                help='Train on MSRVTT data')
    parser.add_argument('--eval_msrvtt', type=int, default=0,
                                help='Evaluate on MSRVTT data')
    parser.add_argument('--eval_youcook', type=int, default=0,
                                help='Evaluate on YouCook2 data')
    parser.add_argument('--eval_smit', type=int, default=0, help='Evaluate on S-MiT data')
    parser.add_argument('--test_smit', type=int, default=0, help='Evaluate on S-MiT test set')
    parser.add_argument('--sentence_dim', type=int, default=-1,
                                help='sentence dimension')
    parser.add_argument(
        '--youcook_train_path',
        type=str,
        default='data/youcook_train_audio.pkl',
        help='')
    parser.add_argument(
        '--youcook_val_path',
        type=str,
        default='data/youcook_val_audio.pkl',
        help='')
    parser.add_argument(
        '--msrvtt_test_path',
        type=str,
        default='data/msrvtt_jsfusion_test.pkl',
        help='')
    parser.add_argument(
        '--msrvtt_train_path',
        type=str,
        default='data/msrvtt_train.pkl',
        help='')
    parser.add_argument('--smit', type=int, default=0, help='Train on S-MiT data')
    parser.add_argument('--smit_num_frames_multiplier', type=int, default=20, help='use 1024 * x audio frames for S-MiT captions')
    parser.add_argument('--smit_train_path', type=str, default='data/smit_train.pkl')
    parser.add_argument('--smit_val_path', type=str, default='data/smit_val.pkl')
    parser.add_argument('--smit_test_path', type=str, default='data/smit_test.pkl')
    parser.add_argument('--load_images', type=int, default=0, help='If 0, use pre-computed video features; if 1, compute features from images')
    parser.add_argument('--use_amp', type=int, default=0, help='Whether to use Automatic Mixed Precision during training')
    parser.add_argument('--train_top_k_2d', type=int, default=0, help='Number of layers to make trainable in 2d feature extractor')
    parser.add_argument('--train_top_k_3d', type=int, default=0, help='Number of layers to make trainable in 3d feature extractor')
    args = parser.parse_args()
    return args
