import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from .transformer_cross import Transformer_cross
from .diffusion_fn_new import DiffMesh_Process




######################################
class SMPLRegressor(nn.Module):
    def __init__(self, args, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.dropout = nn.Dropout(p=dropout_ratio)
        diff_dim = 128
        self.pool = nn.AdaptiveAvgPool2d((None, 1))

        self.fc_ds = nn.Sequential(
            nn.Linear(dim_rep,diff_dim),
            nn.ReLU(inplace=True)
        )


        self.head_pose = nn.Linear(num_joints*diff_dim, param_pose_dim)
        self.head_shape = nn.Linear(num_joints*diff_dim, 10)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)

        # hidden_dim = 1024
        ##########
        seq_len = args.clip_len
        SMPL_MEAN_vertices = os.path.join(args.data_root, 'smpl_mean_vertices.npy')
        mesh_template = torch.from_numpy(np.load(SMPL_MEAN_vertices)).unsqueeze(0)
        self.register_buffer('mesh_template', mesh_template)
        self.fc_mesh_temp = nn.Sequential(
            nn.Linear(1378, diff_dim),
            nn.GELU(),
            nn.Conv1d(15,num_joints,1),
            nn.GELU(),

        )


        self.diffmesh_process = DiffMesh_Process(seq_len, add_step=14, num_joints=num_joints, diff_dim=diff_dim, )
        ##########
        self.smpl = SMPL(
            args.data_root,
            batch_size=64,
            create_transl=False,
        )
        mean_params = np.load(self.smpl.smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.J_regressor = self.smpl.J_regressor_h36m

    def forward(self, feat, init_pose=None, init_shape=None):
        B, T, J, C = feat.shape
        B_T = B * T

        #### diffusion model
        feat_pose = self.fc_ds(feat)

        mesh_template = self.mesh_template.expand(B, -1, -1).to(device=feat_pose.device)
        mesh_template = self.fc_mesh_temp(mesh_template.transpose(1,2).reshape(B,15,-1))
        mesh_template = mesh_template.view(B,1,-1)
        feat_pose, loss_diff = self.diffmesh_process(feat_pose, mesh_template)
        feat_pose = feat_pose.view(B, T, -1)

        feat_shape = feat_pose.permute(0, 2, 1)  # (B, T or 1, JC) -> (B, JC, T or 1)
        feat_shape = self.pool(feat_shape).reshape(B, -1)  # (B, JC)


        pred_pose = self.init_pose.expand(B_T, -1)  # (NT, 24*6)
        pred_shape = self.init_shape.expand(B, -1)  # (N, 10)

        pred_pose = self.head_pose(feat_pose.reshape(B_T, -1)) + pred_pose
        pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_shape = pred_shape.expand(T, B, -1).permute(1, 0, 2).reshape(B_T, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = pred_output.vertices * 1000.0
        assert self.J_regressor is not None
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{
            'theta': torch.cat([pose, pred_shape], dim=1),  # B*T, 72+10)
            'verts': pred_vertices,  # (B*T, 6890, 3)
            'kp_3d': pred_joints,  # (B*T, 17, 3)
            'loss_diff': loss_diff,  # (B, 2*T, 128)
        }]
        return output


class MeshRegressor(nn.Module):
    def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5):
        super(MeshRegressor, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)

    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
        '''
            Input: (N x T x 17 x 3)
        '''
        N, T, J, C = x.shape
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, T, self.feat_J, -1])  # (N, T, J, C)
        smpl_output = self.head(feat)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['verts'] = s['verts'].reshape(N, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
        return smpl_output