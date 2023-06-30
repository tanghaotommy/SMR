import torch
import torch.nn.functional as F


'''Quaternion representation'''
def quat2mat(x):
    B = x.shape[0]
    device = x.device

    rotMat = quat2mat_transform(x[:,:4])  # [B,3,3]
    
    res = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
    res[:,:3,:3] = rotMat
    res[:,3, :3] = x[:,4:]
    return res


def quat2mat_transform(quat):
    """Convert quaternion coefficients to rotation matrix.
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def mat2quat(x):
    # x: SE3 matrix in shape [B,4,4]
    trans = x[:,3,:3]
    rot = x[:,:3,:3]
    quat = mat2quat_transform(rot)
    return torch.cat([quat, trans], dim=1)


def mat2quat_transform(rotation_matrix, eps=1e-6):
    """Convert 3x3 rotation matrix to 4d quaternion vector"""
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def get_relative_pose(cam_1, cam_2):
    '''
    cam_1: pose of camera 1 in world frame, in shape [4,4]
    cam_2: pose of camera 2 in world frame, in shape [t,4,4] for t cameras
    In math, we have:
        P^W = T^w_wToc1 @ P^c1,     T^w_wToc1 is the pose of camera in world frame
        P^W = T^w_wToc2 @ P^c2,     ...
    We want to get T^c1_c1Toc2, which is the relative pose of camera 2 to camera 1, we have
        P^c1 = T^c1_c1Toc2 @ P^c2
    => T^c1_c1Toc2 = T^w_wToc1.inv() @ T^w_wToc2
    If we denote camera pose as |R, t|, we have
                                |0, 1|
        T^c1_c1Toc2 = |R1, t1|-1 @ |R2, t2| = |R1.T @ R2, R1.T @ (t2 - t1)|
                      |0,  1 |     |0,  1 |   |0,         1               |
    '''
    assert len(cam_2.shape) == 3
    b = cam_2.shape[0]

    if len(cam_1.shape) == 2:
        cam_1 = cam_1.unsqueeze(0).repeat(b,1,1)
    
    R1 = cam_1[:,:3,:3]   # [t,3,3]
    t1 = cam_1[:,3,:3]    # [t,3]
    R2 = cam_2[:,:3,:3]   # [t,3,3]
    t2 = cam_2[:,3,:3]    # [t,3]

    R1_T = R1.permute(0,2,1)    # [t,3,3]
    R = torch.matmul(R1_T, R2)  # [t,3,3]
    t = torch.matmul(R1_T, (t2 - t1).view(b,3,1)).squeeze(-1)  # [t,3]

    pose = torch.zeros(b,4,4)      # T_c1_to_c2
    pose[:,:3,:3] = R
    pose[:,3,:3] = t
    pose[:,3,3] = 1

    return pose


def get_relative_pose_RT(RT1, RT2):
    """
    Find relative pose from RT1 -> RT2
    RT1: [1 or B, 3, 3], [1 or B, 3], canonical pose
    RT2: [B, 3, 3], [B, 3]
    """
    R1, T1 = RT1
    R2, T2 = RT2
    
    # [B, 3, 3]
    assert len(R2.shape) == len(R2.shape) == 3
    
    B = R2.shape[0]
    if R1.shape[0] != B:
        assert R1.shape[0] == 1
        R1 = R1.repeat(B, 1, 1)
        T1 = T1.repeat(B, 1)
        
    R = R1.transpose(1, 2) @ R2
    T = R1.transpose(1, 2) @ (T2 - T1).unsqueeze(2)
    T = T.squeeze(-1)
    
    return [R, T]


def transform_relative_pose_RT(RT1, RT2):
    """
    Apply relative pose from RT1 -> RT2
    RT1: [1 or B, 3, 3], [1 or B, 3], canonical pose
    RT2: [B, 3, 3], [B, 3]
    """
    R1, T1 = RT1
    R2, T2 = RT2
    
    R = R1 @ R2
    T = (R1 @ T2.unsqueeze(2)).squeeze(-1) + T1
    
    return [R, T]


def get_relative_cam_transform(cam1, cam2):
    """
    Find relative pose from cam1 -> cam2
    camera transform in the shape of
    [R 0
     T 1]
    cam1: [1, 4, 4]
    cam2: [B, 4, 4]
    """
    # [B, 3, 3]
    assert len(cam1.shape) == len(cam2.shape) == 3
    
    B = cam2.shape[0]
    if cam1.shape[0] != B:
        assert cam1.shape[0] == 1
        cam1 = cam1.repeat(B, 1, 1)
        
    return torch.inverse(cam1) @ cam2
    

def transform_relative_pose(cam1, relative_cam_transform):
    """
    Apply relative pose from cam1 by using relative_cam_transform
    cam1: [1, 4, 4]
    cam2: [B, 4, 4]
    """
    return cam1 @ relative_cam_transform


def get_cam_transform(R, T):
    B = R.shape[0]
    assert R.shape[0] == T.shape[0]
    
    res = torch.cat([R, T.unsqueeze(1)], dim=1)
    c = torch.FloatTensor([0, 0, 0, 1]).to(R.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
    
    return torch.cat([res, c], dim=2)