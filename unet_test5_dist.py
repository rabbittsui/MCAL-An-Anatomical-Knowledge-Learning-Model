# This code is based on: https://github.com/SimonKohl/probabilistic_unet
import os
import torch
import torch.nn as nn
from unet_t5_dist import Unet#unet_t5_dist,unet_t5_dist_film
from utils1 import init_weights, init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from lovasz_loss import LovaszSoftmax

os.environ["CUDA_VISIBLE_DEVICES"]='1'
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, input, target):
        input = F.softmax(input)
        dist = target.cuda()
        score = torch.mean(input * dist)
        return score

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        '''if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1'''

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(norm1)
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,
                               posterior=self.posterior)
        #self.conv_layer=nn.Sequential(nn.Flatten(), nn.ReLU(inplace=True), nn.Linear(28*28*256, 2 * self.latent_dim))
        self.conv_layer1 = nn.Conv2d(num_filters[-1], self.latent_dim, (1, 1), stride=1)
        self.conv_layer2 = nn.Conv2d(num_filters[-1], self.latent_dim, (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0


    def forward(self, input):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        encoding = self.encoder(input)
        self.show_enc = encoding


        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        mu = self.conv_layer1(encoding)
        log_sigma = self.conv_layer2(encoding)

        '''
        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        '''

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = torch.distributions.Normal(mu, log_sigma.mul(0.5).exp())
        return dist


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32, 64, 128, 256], latent_dim=6, no_convs_fcomb=4,
                 beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.apply_last_layer = True
        self.padding = True,
        self.prior = Unet(self.input_channels, self.num_classes, self.num_filters, self.latent_dim,self.no_convs_fcomb,self.initializers)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,self.latent_dim, self.initializers, posterior=True)
    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(segm)
        self.input = patch
        self.seg_out,self.prior_latent_space,self.x0,self.x1 = self.prior.forward(patch)
    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            #kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
            kl_div=kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm,bdm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)
        criterion1 = LovaszSoftmax()
        criterion2 = BoundaryLoss()

        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        self.reconstruction = self.seg_out
        reconstruction_loss = criterion(input=self.reconstruction, target=segm) + criterion1(self.reconstruction, segm)+ criterion1(self.x0, segm)+\
                              criterion1(self.x1, segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)
        return (self.mean_reconstruction_loss + self.beta * self.kl)+10*criterion2(self.reconstruction,bdm)
    def output(self):
        self.reconstruction = self.seg_out
        return self.reconstruction