import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # for training on our DeepHuman dataset
        g_ours = parser.add_argument_group('DeepHuman')
        g_ours.add_argument('--meshDirSearch', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data")
        g_ours.add_argument('--trainingDataRatio', type=float, default="0.8")
        g_ours.add_argument('--datasetType', type=str, default='all', help="all, mini, adjusted")
        g_ours.add_argument('--datasetDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender")
        g_ours.add_argument('--shapeQueryDir', type=str, default="/mnt/tanjiale/shape_query")
        g_ours.add_argument('--visual_check', action='store_true', help="Visual check of the sampled in and out points")
        g_ours.add_argument('--totalNumFrame', type=int, default="108720", help="total data number: N*M'*4 = 6795*4*4 = 108720")
        g_ours.add_argument('--online_sampling', action='store_true', help='online query point sampling, or offline')
        g_ours.add_argument('--resolution_x', type=int, default=171, help='# of grid in mesh reconstruction')
        g_ours.add_argument('--resolution_y', type=int, default=256, help='# of grid in mesh reconstruction')
        g_ours.add_argument('--resolution_z', type=int, default=171, help='# of grid in mesh reconstruction')
        g_ours.add_argument('--preModelDir', type=str, default="./results/results_final_19_09_30_10_29_33", help="if mode is 'finetune' then load pre-trained model from this dir")
        g_ours.add_argument('--resultsDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender/deepHumanResults/expName")
        g_ours.add_argument('--splitNum', type=int, default="8", help="for multi-process running")
        g_ours.add_argument('--splitIdx', type=int, default="0", help="{0, ..., splitNum-1}")
        g_ours.add_argument('--visual_demo_mesh', type=int, default="0", help="num of frames used in visual demo")
        g_ours.add_argument('--shuffle_train_test_ids', action='store_true', help='shuffle training, test data indices or not')
        g_ours.add_argument('--sampleType', type=str, default="sigma3.5_pts5k")
        g_ours.add_argument('--colorSampleType', type=str, default="color_samples")
        g_ours.add_argument('--epoch_range', nargs='+', default=[0, 15], type=int, help='epoch range names used for offline query-pts sampling')
        g_ours.add_argument('--resume_name', type=str, default='example', help='name of the experiment. It decides where to load weights to resume training')
        g_ours.add_argument('--upsample_mode', type=str, default='bicubic', help='bicubic | nearest')
        g_ours.add_argument('--recover_dim', action='store_true', help='recover stack-hour-glass output feature dimensions from BVx256x128x128 to BVx256x512x512')
        g_ours.add_argument('--epoch_offline_len', type=int, default="15", help="number of epochs that have been sampled offline")
        g_ours.add_argument('--load_single_view_meshVoxels', action='store_true', help='load meshVoxels for a single view in order to train the VRN network')
        g_ours.add_argument('--vrn_net_input_height', type=int, default="384", help="vrn network image input height 192*2")
        g_ours.add_argument('--vrn_net_input_width', type=int, default="256", help="vrn network image input width 128*2")
        g_ours.add_argument('--vrn_num_modules', type=int, default=4, help='num of stack-hour-glass')
        g_ours.add_argument('--vrn_num_hourglass', type=int, default=2, help='depth of each hour-glass')
        g_ours.add_argument('--partial_load', action='store_true', help='set strict=False for net.load_state_dict function, useful when you need to load weights for partial networks')
        g_ours.add_argument('--load_from_multi_GPU_shape', action='store_true', help='load weights to single-GPU model, from shape models trained with nn.DataParallel function')
        g_ours.add_argument('--load_from_multi_GPU_color', action='store_true', help='load weights to single-GPU model, from color models trained with nn.DataParallel function')
        g_ours.add_argument('--give_idx', nargs='+', default=[None], type=int, help='list of idx for visual demo')
        g_ours.add_argument('--weight_occu', type=float, default="1000.")
        g_ours.add_argument('--weight_rgb_recon', type=float, default="200.")
        g_ours.add_argument('--vrn_occupancy_loss_type', type=str, default='ce', help='mse | ce')
        g_ours.add_argument('--use_view_pred_loss', action='store_true', help='apply view prediction losses upon deep voxels')
        g_ours.add_argument('--use_3d_gan', action='store_true', help='apply 3d GAN losses upon deep voxels')
        g_ours.add_argument('--view_probs_front_right_back_left', nargs='+', default=[0.15, 0.30, 0.25, 0.30], type=float, help='4-view sampling probs when training with view rendering losses, must sum to 1.0')
        g_ours.add_argument('--use_view_discriminator', action='store_true', help='also apply patch-GAN losses when view prediction losses are in use')
        g_ours.add_argument('--dataType', type=str, default='test', help='train | test | both')
        g_ours.add_argument('--dataTypeZip', type=str, default='both', help='train | test | both')
        g_ours.add_argument('--deepVoxels_fusion', type=str, default=None, help='early | late')
        g_ours.add_argument('--deepVoxels_c_len', type=int, default=8, help='len of deepVoxel features when conducting fusion with 2D aligned features')
        g_ours.add_argument('--deepVoxels_c_len_intoLateFusion', type=int, default=8, help='len of deepVoxel features into the late fusion layers')
        g_ours.add_argument('--multiRanges_deepVoxels', action='store_true', help='use xyz-3-direction deepvoxels sampling')
        g_ours.add_argument('--displacment', type=float, default="0.0722", help="0.035 | 0.0722, displacment used when conducting multiRanges_deepVoxels")
        g_ours.add_argument('--deepVoxelsDir', type=str, default="/trainman-mount/trainman-storage-d5c0a121-bb5d-4afb-8020-c53f096d2a5c/data/humanRender/pifuResults/ourDataShape_vrn_ce_6gpu/train")
        g_ours.add_argument('--mlp_dim_3d', nargs='+', default=[56, 256, 128, 1], type=int, help='# of dimensions of mlp for DeepVoxels 3d branch')
        g_ours.add_argument('--mlp_dim_joint', nargs='+', default=[0, 256, 128, 1], type=int, help='# of dimensions of mlp for joint 2d-3d branch')
        g_ours.add_argument('--discriminator_accuracy_update_threshold', type=float, default="0.8", help="only update the discriminator if fake/real accuracies are both below this threshold, to avoid discriminator going too fast")
        g_ours.add_argument('--weight_3d_gan_gen', type=float, default="15.", help="weight for 3d-gan generator loss, to be comparable with the occupancy loss")
        g_ours.add_argument('--must_run_in_train_modes', type=str, default="ourDataShape_vrn_ce_6gpu_3dGAN,XXX", help='some models have to be run in train modes due to some hacky issues of BN layers')
        g_ours.add_argument('--num_skip_frames', type=int, default="1", help="num of frames to skip when generating visual demos")
        g_ours.add_argument('--use_embedder', action='store_true', help='Use embedder for positional encoding')
        g_ours.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
        g_ours.add_argument("--embedder_input_dim", type=int, default=3, help='1 or 2 or 3')

        # pix2pixHD related
        g_ours.add_argument('--use_pix2pix', action='store_true', help='Use pix2pixHD for back view inference')
        g_ours.add_argument('--isTrain', action='store_true', help="This is for joint training (not implemented)")
        g_ours.add_argument('--model', type=str, default='pix2pixHD', help='which model to use for pix2pix')
        g_ours.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        g_ours.add_argument('--norm_pix2pix', type=str, default='instance', help='instance normalization or batch normalization')        
        g_ours.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        g_ours.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        g_ours.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        g_ours.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        g_ours.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # for training
        g_ours.add_argument('--checkpoints_dir', type=str, default='', help='Checkpoint for saving')
        g_ours.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        g_ours.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        g_ours.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        g_ours.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        g_ours.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        g_ours.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        #input/output sizes
        # g_ours.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        g_ours.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        g_ours.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        g_ours.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        g_ours.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for generator
        g_ours.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        g_ours.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        g_ours.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        g_ours.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        g_ours.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        g_ours.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        g_ours.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # for instance-wise features
        g_ours.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        g_ours.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        g_ours.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        g_ours.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        g_ours.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        g_ours.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        g_ours.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        g_ours.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')        

        # for discriminators        
        g_ours.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        g_ours.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        g_ours.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        g_ours.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        g_ours.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        g_ours.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        g_ours.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        g_ours.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataroot', type=str, default='./data',
                            help='path to images (data folder)')

        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')

        g_exp.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
        g_exp.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        g_train.add_argument('--batch_size', type=int, default=2, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--learning_rate_3d_gan', type=float, default=1e-5, help='adam learning rate')
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=100, help='num epoch to train')

        g_train.add_argument('--freq_plot', type=int, default=10, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
       
        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')
        
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--resume_iter', type=int, default=-1, help='iter resuming the training, within the resume_epoch defined above')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=5.0, help='perturbation standard deviation for positions')

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')
        g_sample.add_argument('--num_sample_color', type=int, default=0, help='# of sampling points')

        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--norm_color', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')

        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_color', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')

        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')

        # for train
        parser.add_argument('--random_flip', action='store_true', help='if random flip')
        parser.add_argument('--random_trans', action='store_true', help='if random flip')
        parser.add_argument('--random_scale', action='store_true', help='if random flip')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[3, 6, 10],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1')
        parser.add_argument('--occupancy_loss_type', type=str, default='mse', help='mse | l1 | ce')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--startIndex', type=int, default=0)
        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_netV_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netC_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask')
        parser.add_argument('--img_path', type=str, help='path for input image')
        parser.add_argument('--deepVoxelPath', type=str, help='path for deep voxel')
        parser.add_argument('--mesh_name', type=str, help='name')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
