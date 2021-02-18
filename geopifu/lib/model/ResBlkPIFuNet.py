import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
import functools
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from ..net_util import *
from .pix2pixHD.models.models import create_model
from .pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from .pix2pixHD.util import util
import pdb
import cv2

def get_embedder(opt):
    """
        opt.multires
        opt.i
    """
    i = 0
    multires = opt.multires

    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
        'include_input' : True,
        'input_dims' : opt.embedder_input_dim,
        'max_freq_log2' : multires-1,
        'num_freqs' : multires,
        'log_sampling' : True,
        'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embedder_obj, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ResBlkPIFuNet(BasePIFuNet):
    def __init__(self, opt,
                 projection_mode='orthogonal'):
        if opt.color_loss_type == 'l1':
            error_term = nn.L1Loss()
        elif opt.color_loss_type == 'mse':
            error_term = nn.MSELoss()

        super(ResBlkPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'respifu'
        self.opt = opt

        norm_type = get_norm_layer(norm_type=opt.norm_color) # default: nn.InstanceNorm2d without {affine, tracked statistics}
        self.image_filter = ResnetFilter(opt, norm_layer=norm_type) # input image filter

        mlp_dim_color = self.opt.mlp_dim_color
        if opt.use_embedder:
            self.embedder, self.embedder_ch = get_embedder(opt)
            # pdb.set_trace()
            # 512 + 63 = 576 concat image_feature + xyz
            mlp_dim_color[0] += self.embedder_ch - 1
            # 513 + 21 = 534 concat image_feature + upscaled z + z_feat
            # mlp_dim_color[0] += self.embedder_ch
            # pdb.set_trace()

        if opt.use_pix2pix:
            mlp_dim_color[0] += 256
         
        self.surface_classifier = SurfaceClassifier(
            filter_channels=mlp_dim_color, # default: 513, 1024, 512, 256, 128, 3
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual, # default: False
            last_op=nn.Tanh(), # output float -1 ~ 1, RGB colors
            opt=self.opt)

        self.normalizer = DepthNormalizer(opt)

        # weights initialization for conv, fc, batchNorm layers
        init_net(self)

        # use pix2pix for back view inference (don't init this)
        if opt.use_pix2pix:
            print("Using pix2pix for back view inference...")
            # create the back view inference model
            self.pix2pixHD = create_model(opt) # output (3,512,512)
            # create an extra image filter? or share the same image filter?
            self.image_filter_back = ResnetFilter(opt, norm_layer=norm_type)
            # pdb.set_trace()

    def filter(self, images):
        '''
        Filter the input images store all intermediate features.

        Input:
            images: (BV, 3, 512, 512) input images

        output:
            self.im_feat: (BV, 256, 128, 128)
            and
            self.im_feat_back: (BV, 256, 128, 128)
        '''

        self.im_feat = self.image_filter(images)

        if self.image_filter_back:
            self.im_feat_back = self.image_filter_back(self.image_back)

    def attach(self, im_feat):

        if self.image_filter_back:
            # (BV*num_views, 768, 128, 128)
            self.im_feat = torch.cat([im_feat, self.im_feat, self.im_feat_back], 1)
        else:
            self.im_feat = torch.cat([im_feat, self.im_feat], 1)

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point. Image features should be pre-computed before this call. store all intermediate features.
        query() function may behave differently during training/testing.

        Input
            points: (B * num_views, 3, 5000), near surface points, loat XYZ coords are inside the 3d-volume of [self.B_MIN, self.B_MAX]
            calibs: (B * num_views, 4, 4) calibration matrix
            transforms: default is None
            labels: (B, 3, 5000), gt-RGB color, -1 ~ 1 float

        Output
            self.pred: (B, 3, 5000) RGB predictions for each point, float -1 ~ 1
        '''

        if labels is not None:
            self.labels = labels # (B, 3, 5000), gt-RGB color, -1 ~ 1 float

        # (B * num_view, 3, N), points are projected onto XY (-1,1)x(-1,1) plane of the cam coord.
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :] # (B * num_view, 2, N)
        z = xyz[:, 2:3, :] # (B * num_view, 1, N)

        # (B * num_view, 1, N)
        z_feat = self.normalizer(z)

        if self.opt.use_embedder:
            # use positional encoding
            xyz = self.embedder.embed(torch.cat([xy, z_feat], 1).transpose(1,2)).transpose(1,2)
            point_local_feat_list = [self.index(self.im_feat, xy), xyz]                   # [(B * num_views, 512, 5000), (B * num_view, 63, 5000)]
            # embed_z = self.embedder.embed(z_feat.transpose(1,2)).transpose(1,2)
            # point_local_feat_list = [self.index(self.im_feat, xy), embed_z, z_feat]     # [(B*num_views, 512, 8000), (B*num_view, 32, 8000)]
        else:    
            # [(B * num_views, 512, 5000), (B * num_view, 1, 5000)]
            point_local_feat_list = [self.index(self.im_feat, xy), z_feat]

        # (B * num_views, 512+1, 5000) or (B * num_views, 512+63, 5000) or (B*num_views, 768+1 or +63, 5000)
        point_local_feat = torch.cat(point_local_feat_list, 1)

        # (B, 3, 5000), num_views are canceled out by mean pooling, float -1 ~ 1. RGB rolor
        self.preds = self.surface_classifier(point_local_feat)

    def infer_back(self, images):
        """
        input
            images: (BV, 3, 512, 512)
        output
            image_back: (BV, 3, 512, 512)
        """
    
        self.image_back = self.pix2pixHD.inference(images, torch.Tensor(0), torch.Tensor(0))

        # cv2.imwrite('input_img.jpg', cv2.cvtColor(util.tensor2im(images.data[0]), cv2.COLOR_RGB2BGR))
        # cv2.imwrite('synthesized_img.jpg', cv2.cvtColor(util.tensor2im(self.image_back[0]), cv2.COLOR_RGB2BGR))

        # pdb.set_trace()

    def forward(self, images, im_feat, points, calibs, transforms=None, labels=None):
        """
        input
            images: (BV, 3, 512, 512)
            im_feat: (BV, 256, 128, 128) from the stacked-hour-glass filter
        """

        if self.image_filter_back:
            self.infer_back(images) # infer the back image using input image (BV, 3, 512, 512)

        # extract self.im_feat, (BV, 256, 128, 128), not the input im_feat
        self.filter(images)

        # concat the input im_feat with self.im_feat and get the new self.im_feat: (BV, 512, 128, 128)
        self.attach(im_feat) # (BV, 512, 128, 128) or (BV, 768, 128, 128) if using back view inference

        # extract self.preds: (B, 3, 5000), num_views are canceled out by mean pooling, float -1 ~ 1. RGB rolor
        self.query(points, calibs, transforms, labels)

        # return self.preds, (B, 3, 5000)
        res = self.get_preds()

        # get the error, default is nn.L1Loss()
        error = self.get_error() # R

        return res, error


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer, default: 64
            norm_layer          -- normalization layer, default is nn.InstanceNorm2d without {affine, tracked statistics}
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert (n_blocks >= 0)
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers, i in {0, 1}
            mult = 2 ** i # in {1, 2}
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling # default: 4
        for i in range(n_blocks):  # add ResNet blocks, default: 6

            # the last resnet block
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, last=True)] # input/output dimensions same

            # the previous resnet blocks
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)] # input/output dimensions same

        if opt.use_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)












