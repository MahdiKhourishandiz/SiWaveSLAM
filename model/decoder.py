import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    def __init__(self):
        self.mlp_bias_on = True
        self.pos_encoding_band = 10
        self.pos_input_dim = 3
        self.feature_dim = 64
        self.main_loss_type = "mse"
        self.device = "cuda"
        self.logistic_gaussian_ratio = 1.0
        self.sigma_sigmoid_m = 1.0
        self.use_gaussian_pe = False

class Decoder(nn.Module):
    def __init__(
            self,
            config: Config,
            hidden_dim,
            hidden_level,
            out_dim,
            is_time_conditioned=False,
    ):
        super().__init__()

        self.out_dim = out_dim
        bias_on = config.mlp_bias_on
        self.use_leaky_relu = True
        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim

        if config.use_gaussian_pe:
            position_dim = config.pos_input_dim + 2 * config.pos_encoding_band
        else:
            position_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)

        feature_dim = config.feature_dim
        input_layer_count = feature_dim + position_dim

        if is_time_conditioned:
            input_layer_count += 1
        layers = []
        for i in range(hidden_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, hidden_dim, bias_on))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(hidden_dim, out_dim, bias_on)

        if config.main_loss_type == "bce":
            self.sdf_scale = config.logistic_gaussian_ratio * config.sigma_sigmoid_m
        else:
            self.sdf_scale = 1.0

        self.to(config.device)

    def sdf(self, features):
        input_features = features
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.leaky_relu(l(features)) if self.use_leaky_relu else F.relu(l(features))
            else:
                h = F.leaky_relu(l(h)) if self.use_leaky_relu else F.relu(l(h))
            if k % 2 == 1:
                h = h + input_features
                input_features = h

        out = self.lout(h).squeeze(1)
        out *= self.sdf_scale
        return out

    def sem_label_prob(self, features):
        input_features = features
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.leaky_relu(l(features)) if self.use_leaky_relu else F.relu(l(features))
            else:
                h = F.leaky_relu(l(h)) if self.use_leaky_relu else F.relu(l(h))
            if k % 2 == 1:
                h = h + input_features
                input_features = h

        out = F.log_softmax(self.lout(h), dim=-1)
        return out

    def regress_color(self, features):
        input_features = features
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.leaky_relu(l(features)) if self.use_leaky_relu else F.relu(l(features))
            else:
                h = F.leaky_relu(l(h)) if self.use_leaky_relu else F.relu(l(h))
            if k % 2 == 1:
                h = h + input_features
                input_features = h

        out = torch.clamp(self.lout(h), 0.0, 1.0)
        return out