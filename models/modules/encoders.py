import torch
from torch import nn

class ConvGaussianEncoder(nn.Module):
    def __init__(self, img_shape, a_shape, chan_hidden=32, hidden_layers=1):
        super().__init__()
        if len(img_shape) != 3 or len(a_shape) != 3:
            raise ValueError("need shapes in format: [chan, dim_x, dim_y]")
        chan_in = img_shape[0]
        self.a_shape = a_shape
        self.input_layer = nn.Sequential(
            nn.Conv2d(chan_in, chan_hidden, 4, 2, 1),
            nn.SiLU(),
            # nn.BatchNorm2d(chan_hidden)
            nn.GroupNorm(1, chan_hidden)
        )
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    chan_hidden * (2**i), 
                    chan_hidden * (2 ** (i+1)), 
                    4, 2, 1
                ),
                nn.SiLU(),
                # nn.BatchNorm2d(chan_hidden)
                nn.GroupNorm(1, chan_hidden * (2 ** (i+1)))
            ) for i in range(hidden_layers)
        ])
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((a_shape[1],a_shape[2]))
        pooling_dim = chan_hidden * (2 ** hidden_layers)

        self.mu_layer = nn.Conv2d(pooling_dim, a_shape[0], 1, 1)
        self.logsigma_layer = nn.Sequential(
            nn.Conv2d(pooling_dim, a_shape[0], 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.global_avg_pooling(x)
        mu = self.mu_layer(x)
        logsigma = self.logsigma_layer(x)
        return mu, logsigma

class DenseGaussianEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers=1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.SiLU()
        )
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.SiLU()
            ) for _ in range(hidden_layers)
        ])
        self.mu_layer = nn.Linear(dim_hidden, dim_out)
        self.logsigma_layer = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        mu = self.mu_layer(x)
        logsigma = self.logsigma_layer(x)
        return mu, logsigma