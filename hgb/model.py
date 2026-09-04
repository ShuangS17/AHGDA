import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flip_gradient import GRL

def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


class Transformer(nn.Module):
    '''
        The transformer-based semantic fusion
    '''
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class LinearPerMetapath(nn.Module):
    '''
        Linear projection per metapath for feature projection
    '''
    def __init__(self, cin, cout, num_metapaths):
        super(LinearPerMetapath, self).__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


class DomainDiscriminator(nn.Module):
    def __init__(self, n_emb):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(n_emb, 256)
        self.h_dann_2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 2)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        h_grl = F.relu(self.h_dann_1(h_grl))
        h_grl = F.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit

unfold_nested_list = lambda x: sum(x, [])

class SeHGNN(nn.Module):
    def __init__(self, hidden1, hidden2, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_drop, n_fp_layers, n_task_layers, act,
                 residual=False, data_size=None, num_heads=1):
        super(SeHGNN, self).__init__()
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(self.feat_keys) + len(self.label_feat_keys)
        self.tgt_type = tgt_type
        self.residual = residual
        self.input_drop = nn.Dropout(input_drop)
        self.data_size = data_size
        self.embeding = nn.ParameterDict({})

        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, hidden1))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(torch.Tensor(nclass, hidden1))
        else:
            self.labels_embeding = {}

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(hidden1, hidden2, num_channels),
               nn.LayerNorm([num_channels, hidden2]),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               LinearPerMetapath(hidden2, hidden2, num_channels),
               nn.LayerNorm([num_channels, hidden2]),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(n_fp_layers - 1)])
            )
        )

        self.semantic_fusion = Transformer(hidden2, num_heads=num_heads, att_drop=att_drop, act=act)
        self.fc_after_concat = nn.Linear(num_channels * hidden2, hidden2)

        if self.residual:
            self.res_fc = nn.Linear(hidden1, hidden2)


        self.task_mlp = nn.Sequential(
            *([nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               nn.Linear(hidden2, hidden2),
               nn.BatchNorm1d(hidden2, affine=False),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(n_task_layers - 1)])
            + [nn.Linear(hidden2, nclass),
               nn.BatchNorm1d(nclass, affine=False, track_running_stats=False)]
            )
        )

        self.nodeP_discriminator = DomainDiscriminator(n_emb=hidden2)
        self.nodeA_discriminator = DomainDiscriminator(n_emb=hidden1)
        self.nodeV_discriminator = DomainDiscriminator(n_emb=hidden1)
        self.metapath_discriminator = DomainDiscriminator(n_emb=hidden2)
        self.grl = GRL()
        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()

    def forward(self, feat_S_batch, label_feats_S_batch, feat_T_batch, label_feats_T_batch=None):
        output_S, output_T, d_logits_meta = None, None, None
        d_logits_P, d_logits_A, d_logits_V = None, None, None
        x_S, x_T = None, None

        if feat_S_batch is not None:
            features_S = {k: self.input_drop(x @ self.embeding[k]) for k, x in feat_S_batch.items()}
            labels_S = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_feats_S_batch.items()}
            x_list_S = [features_S[k] for k in self.feat_keys if k in features_S]

            if label_feats_S_batch:
                x_list_S.extend([labels_S[k] for k in self.label_feat_keys if k in labels_S])

            if len(x_list_S) < self.num_channels:
                x_list_S.extend([torch.zeros_like(x_list_S[0])] * (self.num_channels - len(x_list_S)))

            x_S_stacked = torch.stack(x_list_S, dim=1) # [B_S, num_channels, hidden1]

            B_S = num_Snode = features_S[self.tgt_type].shape[0]
            x_S_proj = self.feature_projection(x_S_stacked) # [B_S, num_channels, hidden2]
            x_S_fused = self.semantic_fusion(x_S_proj, mask=None).transpose(1, 2) # [B_S, hidden2, num_channels]
            x_S = self.fc_after_concat(x_S_fused.reshape(B_S, -1)) # [B_S, hidden2*num_channels]->[B_S, hidden2]
            if self.residual:
                x_S = x_S + self.res_fc(features_S[self.tgt_type])


        if feat_T_batch is not None:
            features_T = {k: self.input_drop(x @ self.embeding[k]) for k, x in feat_T_batch.items()}
            x_list_T_label = []
            if label_feats_T_batch:
                labels_T = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_feats_T_batch.items()}
                x_list_T_label = [labels_T[k] for k in self.label_feat_keys if k in labels_T]
            x_list_T = [features_T[k] for k in self.feat_keys if k in features_T] + x_list_T_label

            if len(x_list_T) < self.num_channels:
                x_list_T.extend([torch.zeros_like(x_list_T[0])] * (self.num_channels - len(x_list_T)))

            x_T_stacked = torch.stack(x_list_T, dim=1)

            B_T = num_Tnode = features_T[self.tgt_type].shape[0]
            x_T_proj = self.feature_projection(x_T_stacked)
            x_T_fused = self.semantic_fusion(x_T_proj, mask=None).transpose(1, 2)
            x_T = self.fc_after_concat(x_T_fused.reshape(B_T, -1))
            if self.residual:
                x_T = x_T + self.res_fc(features_T[self.tgt_type])

        if x_S is not None and x_T is not None:
            combined_P = torch.cat((x_S, x_T), dim=0)
            d_logits_P = self.nodeP_discriminator(self.grl(combined_P))

            feat_meta_S = x_S_proj.reshape(-1, x_S_proj.size(-1))  # [B_S * M, hidden2]
            feat_meta_T = x_T_proj.reshape(-1, x_T_proj.size(-1))  # [B_T * M, hidden2]

            combined_meta = torch.cat((feat_meta_S, feat_meta_T), dim=0)
            d_logits_meta = self.metapath_discriminator(self.grl(combined_meta))

        if x_S is not None:
            output_S = self.task_mlp(x_S)

        if x_T is not None:
            output_T = self.task_mlp(x_T)

        return output_S, output_T, d_logits_P, d_logits_meta, x_S, x_T

    def forward_neighbor_alignment(self, ntype, feat_S, feat_T):
        emb_S = self.input_drop(feat_S @ self.embeding[ntype])
        emb_T = self.input_drop(feat_T @ self.embeding[ntype])

        combined = torch.cat((emb_S, emb_T), dim=0)
        h_grl = self.grl(combined)

        if ntype == 'A':
            d_logits = self.nodeA_discriminator(h_grl)
        else:  # V (DBLP) or D (IMDB)
            d_logits = self.nodeV_discriminator(h_grl)

        return d_logits
