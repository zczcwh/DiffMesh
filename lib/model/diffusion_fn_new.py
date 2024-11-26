import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .transformer_cross import Transformer_cross, Transformer

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x_in, mesh_template, num_seq, add_step, diffusion_model, condition_generator, b):
    with torch.no_grad():
        take_mean = False

        if num_seq != x_in.shape[1]:
            x_in = x_in.repeat(1,num_seq,1)
            take_mean = True


        x_0 = mesh_template
        B, seq_len, emb_dim = x_in.shape
        feat_g = torch.zeros((B,add_step,emb_dim)).to(device=x_in.device)
        feat_g = torch.cat((feat_g, x_in), dim=1)
        feat_g = condition_generator(feat_g)
        seq = range(0, num_seq+add_step, 1)
        n = x_0.size(0)
        seq_next = [-1] + list(seq[:-1])

        xs = [x_0]
        xs_start = xs[-1]
        x_out = []
        condition_pred = []
        for i, j, k in zip(reversed(seq), reversed(seq_next), seq):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            et, cond_pred_next = diffusion_model(xt, feat_g[:,k,:], t.float())
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = (
                    0 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * xs_start  + c2 * et
            xs.append(xt_next)
            if k > (add_step-1):
                x_out.append(xt_next)
                condition_pred.append(cond_pred_next)

        x_out = torch.cat(x_out, axis=1)
        condition_pred = torch.cat(condition_pred, axis=1)[:, :-1, :]

        if take_mean:
            x_out = x_out.mean(dim=1).unsqueeze(dim=1)
            condition_pred = condition_pred.mean(dim=1).unsqueeze(dim=1)

    return x_out, condition_pred


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class diffusion_model(nn.Module):
    def __init__(self, num_joints=17, diff_dim=256):
        super(diffusion_model, self).__init__()

        self.num_joints = num_joints
        self.diff_dim = diff_dim
        self.temb_fc = nn.Sequential(
            nn.Linear(self.diff_dim, self.diff_dim),
            nn.GELU(),
            nn.Linear(self.diff_dim, self.diff_dim),
        )

        self.diffmodel_encoder = Transformer_cross(embed_dim=self.diff_dim, depth=4, num_heads=4, mlp_ratio=2., drop_path_rate=0.0, )

    def forward(self, x_cur, feat_g_cur, t):

        B = x_cur.shape[0]
        x_cur = x_cur.view(B, self.num_joints, self.diff_dim)
        temb = get_timestep_embedding(t, self.diff_dim)
        temb = self.temb_fc(temb).view(B, 1, -1)
        feat_g_cur = feat_g_cur.view(B, self.num_joints, self.diff_dim)

        x_cur = x_cur + temb
        feat_g_cur =  feat_g_cur + temb
        x = torch.cat((x_cur, feat_g_cur), dim=1)

        x = self.diffmodel_encoder(x)

        x_out = x[:,:self.num_joints,:].view(B, 1, self.num_joints*self.diff_dim)
        cond_next = x[:,self.num_joints:,:].view(B, 1, self.num_joints*self.diff_dim)


        return x_out, cond_next




class diffusion_process(nn.Module):
    def __init__(self, seq_len, add_step, num_joints, diff_dim):
        super(diffusion_process, self).__init__()
        self.device = device = torch.device("cuda")
        self.seq_len = seq_len
        self.step = add_step


        self.diffusion_model = diffusion_model(num_joints, diff_dim)
        self.condition_generator = Transformer(num_patch=seq_len+add_step, in_embed_dim=diff_dim*17, depth=2, mlp_ratio=0.125)

    def forward(self, x_in, mesh_template):
        # generate nosiy sample based on seleted time t and beta
        betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.001,
            num_diffusion_timesteps=self.seq_len + self.step,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]


        pred_x,  condition_pred = generalized_steps(x_in, mesh_template, self.seq_len, self.step, self.diffusion_model, self.condition_generator, self.betas, )


        if x_in.shape[1] != self.seq_len:
            cond_gt = x_in
        else:
            cond_gt = x_in[:, 1:, :]

        loss_diff = torch.cat((cond_gt, condition_pred), dim=1)

        return pred_x, loss_diff


######################################

class DiffMesh_Process(nn.Module):
    def __init__(self, seq_len, add_step, num_joints=17, diff_dim=256):
        super(DiffMesh_Process, self).__init__()
        self.seq_len = seq_len
        self.entire_process = diffusion_process(seq_len, add_step, num_joints, diff_dim)


        self.fc1 = nn.Sequential(nn.Linear(diff_dim, diff_dim),
                                 nn.GELU(),
                                 )


    def forward(self, x_in, mesh_template):
        B, T, J, C = x_in.shape
        x_in = x_in.view(B,T,J*C)
        mesh_template = mesh_template.view(B,1,J*C)

        pred_x, loss_diff = self.entire_process(x_in, mesh_template)
        x = self.fc1(pred_x.view(B,T,J,C)) + x_in.view(B,T,J,C)

        return x, loss_diff

