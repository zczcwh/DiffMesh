import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import rotation_matrix_to_angle_axis, rot6d_to_rotmat, smpl_aa_to_ortho6d
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

        self.fc1 = nn.Sequential(
            nn.Linear(num_joints * diff_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Conv1d(param_pose_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )


        self.head_pose = nn.Conv1d(2*hidden_dim, param_pose_dim, kernel_size=5, stride=1, padding=2)
        # self.head_shape = nn.Linear(num_joints*diff_dim, 10)
        # nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)

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
        # mean_params = np.load(self.smpl.smpl_mean_params)
        # init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        # init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        # self.register_buffer('init_pose', init_pose)
        # self.register_buffer('init_shape', init_shape)
        self.J_regressor = self.smpl.J_regressor_h36m

    def forward(self, feat, init_pose=None, init_shape=None):
        B, T, J, C = feat.shape
        B_T = B * T

        #### diffusion model
        feat = self.fc_ds(feat)

        mesh_template = self.mesh_template.expand(B, -1, -1).to(device=feat.device)
        mesh_template = self.fc_mesh_temp(mesh_template.transpose(1,2).reshape(B,15,-1))
        mesh_template = mesh_template.view(B,1,-1)
        feat, loss_diff = self.diffmesh_process(feat, mesh_template)
        feat = feat.view(B, T, -1)  ## (B, T, J*C0 )

        ###############


        feat = self.dropout(feat)
        feat_pose = feat.reshape(B_T, -1)  # (B*T, J*C0)
        feat_pose = self.fc1(feat_pose)  # (B*T, C)
        init_pose = init_pose.reshape(B, T, -1).permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        # feat_pose = self.fc1(feat_pose)
        # feat_pose = self.bn1(feat_pose)
        # feat_pose = self.relu1(feat_pose)  # (BT, C)
        #
        feat_pred = self.fc2(init_pose)  # (B, C, T)

        feat_pose = feat_pose.reshape(B, T, -1).permute(0, 2, 1)  # (BT, C) -> (B, C, T)
        feat_pose = torch.cat((feat_pose, feat_pred), 1)  # (B, 2C, T)
        pred_pose = self.head_pose(feat_pose) + init_pose  # (B, param_pose_dim, T)

        pred_pose = pred_pose.permute(0, 2, 1).reshape(B_T, -1)
        pred_shape = init_shape.reshape(B_T, -1)
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
            'theta': torch.cat([pose, pred_shape], dim=1),  # (N*T, 72+10)
            'verts': pred_vertices,  # (N*T, 6890, 3)
            'kp_3d': pred_joints,  # (N*T, 17, 3)
            'loss_diff': loss_diff,  # (B, 2*T, 128)
        }]
        return output


# class MeshRegressor(nn.Module):
#     def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5):
#         super(MeshRegressor, self).__init__()
#         self.backbone = backbone
#         self.feat_J = num_joints
#         self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)
#
#     def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
#         '''
#             Input: (N x T x 17 x 3)
#         '''
#         N, T, J, C = x.shape
#         feat = self.backbone.get_representation(x)
#         feat = feat.reshape([N, T, self.feat_J, -1])  # (N, T, J, C)
#         smpl_output = self.head(feat)
#         for s in smpl_output:
#             s['theta'] = s['theta'].reshape(N, T, -1)
#             s['verts'] = s['verts'].reshape(N, T, -1, 3)
#             s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
#         return smpl_output

class MeshRegressor(nn.Module):
    def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5):
        super(MeshRegressor, self).__init__()
        self.backbone = backbone
        self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)

    def forward(self, x, init_pose=None, init_shape=None):
        '''
            Input: (N x T x 17 x 3)
        '''
        N, T, J, C = x.shape
        feat = self.backbone.get_representation(x)
        init_pose = smpl_aa_to_ortho6d(init_pose)
        smpl_output = self.head(feat, init_pose, init_shape)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['verts'] = s['verts'].reshape(N, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
        return smpl_output