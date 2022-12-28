import torch

class ConvNet(torch.nn.Module):
    def __init__(self, latent_channels=64, latent_dims=784,
                 sigma=None, mu=None):
        super(ConvNet, self).__init__()
        # size c x 14 x 14
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=latent_channels,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1)
        # size c x 7 x 7 
        self.conv2 = torch.nn.Conv2d(in_channels=latent_channels,
                                     out_channels=2 * latent_channels,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1)
        self.fc_mu = torch.nn.Linear(
            in_features=latent_channels * 2 * 7 * 7,
            out_features=latent_dims)
        self.fc_logvar = torch.nn.Linear(
            in_features=latent_channels * 2 * 7 * 7,
            out_features=latent_dims)
        self.sigma_learnable = 1
        if sigma == 'identity':
          print('std. deviation is set to Identity')
          self.sigma_learnable = 0

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        # flatten batch of multi-channel feature maps
        # to a batch of feature vectors
        x = x.view(x.size(0), -1) 
        x_mu = self.fc_mu(x)
        x_logvar = (1 - self.sigma_learnable) * self.fc_logvar(x)
        return x_mu, x_logvar