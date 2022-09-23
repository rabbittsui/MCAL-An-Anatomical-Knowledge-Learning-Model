from unet_blocks1res import *
import torch.nn.functional as F
from functools import partial
from utils1 import init_weights, init_weights_orthogonal_normal, l2_regularisation

nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class Film(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, latent_dim, num_dim):
        super(Film, self).__init__()
        self.num_dim = num_dim  # output channels
        self.latent_dim = latent_dim
        self.layers1 = nn.Sequential(nn.Linear(self.latent_dim, self.num_dim), nn.ReLU(inplace=True))
        self.layers2 = nn.Sequential(nn.Linear(self.num_dim, self.num_dim * 2))
        self.conv_layer1 = nn.Sequential(nn.Conv2d(self.num_dim, self.num_dim, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(self.num_dim), nn.ReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(nn.Conv2d(self.num_dim, self.num_dim, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(self.num_dim))
        self.activate = nn.ReLU(inplace=True)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        z = z.rsample()
        # z = torch.squeeze(z, dim=2)
        # z = torch.squeeze(z, dim=2)
        output = self.layers1(z)
        output = self.layers2(output)
        beita = output[:, :self.num_dim]
        gamma = output[:, self.num_dim:]
        beita = torch.unsqueeze(beita, 2)
        beita = self.tile(beita, 2, feature_map.shape[2]).cuda()
        beita = torch.unsqueeze(beita, 3)
        beita = self.tile(beita, 3, feature_map.shape[3]).cuda()
        gamma = torch.unsqueeze(gamma, 2)
        gamma = self.tile(gamma, 2, feature_map.shape[2]).cuda()
        gamma = torch.unsqueeze(gamma, 3)
        gamma = self.tile(gamma, 3, feature_map.shape[3]).cuda()
        feature_map = self.conv_layer1(feature_map)
        feature_map1 = feature_map
        feature_map = self.conv_layer1(feature_map)
        # feature_map = torch.cat((output,feature_map ), dim=self.channel_axis)
        output = torch.mul(feature_map, gamma) + beita
        output = self.activate(output)

        return output + feature_map1  # self.cat_layer(feature_map)


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, latent_dim, no_convs_fcomb, initializers,
                 apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()
        self.latent_dim = latent_dim
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = initializers
        self.prior = Film(self.latent_dim, self.num_filters[-1] + 4)

        for i in range(len(self.num_filters)):
            input = 1 if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2

        for i in range(n, -1, -1):

            if i == n:
                output = self.num_filters[i + 1] + 4
            else:
                output = self.num_filters[i + 1]

            # input = output + self.num_filters[i] + 5
            input = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(input, num_classes, kernel_size=1)
            # nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            # nn.init.normal_(self.last_layer.bias)

        self.dblock = DACblock(self.num_filters[-1])
        self.spp = SPPblock(self.num_filters[-1])
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1).cuda()
        self.convx0 = nn.Sequential(nn.Conv2d(self.num_filters[2], 1, kernel_size=3, stride=1, padding=1))
        self.convx1 = nn.Sequential(nn.Conv2d(self.num_filters[1], 1, kernel_size=3, stride=1, padding=1))
    # self.conv_layer=nn.Sequential(nn.Flatten(), nn.ReLU(inplace=True), nn.Linear(28*28*256, 2 * self.latent_dim))
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)
        encoding = x
        # encoding = torch.flatten(encoding, start_dim=2, end_dim=-1)
        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        x_f = dist
        x = self.dblock(x)
        x = self.spp(x)
        prior = self.prior.forward(x, dist)
        feature_list = []
        for i, up in enumerate(self.upsampling_path):
            if i == 0:
                x = prior
            else:
                x = x
            x = up(x, blocks[-i - 1])
            feature_list.append(x)
            # prior =  nn.functional.interpolate(prior, mode='bilinear', scale_factor=2, align_corners=True)
            # x = torch.cat((x, prior), dim=1)
        del blocks

        # Used for saving the activations and plotting
        # if val:
        # self.activation_maps.append(x)
        feature_x0 = feature_list[0]
        feature_x1 = feature_list[1]
        x0 = nn.functional.interpolate(feature_x0, mode='bilinear', scale_factor=4, align_corners=True)
        x1 = nn.functional.interpolate(feature_x1, mode='bilinear', scale_factor=2, align_corners=True)
        x0 = self.convx0(x0)
        x1 = self.convx1(x1)

        if self.apply_last_layer:
            x = self.last_layer(x)
        return x, x_f, x0, x1


