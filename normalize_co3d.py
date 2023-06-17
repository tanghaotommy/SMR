import gzip
import json
import os
from transforms3d.euler import euler2mat
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
import pytorch3d
import cv2
from tqdm import tqdm
import pickle


# These functions are copied from Gen6D
def project_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    mask0 = (np.abs(dpt)<1e-4) & (np.abs(dpt)>0)
    if np.sum(mask0)>0: dpt[mask0]=1e-4
    mask1=(np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1)>0: dpt[mask1]=-1e-4
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

def compute_normalized_ratio(pc):
    min_pt = np.min(pc,0)
    max_pt = np.max(pc,0)
    dist = np.linalg.norm(max_pt - min_pt)
    scale_ratio = 2.0 / dist
    return scale_ratio

def pose_inverse(pose):
    R = pose[:,:3].T
    t = - R @ pose[:,3:]
    return np.concatenate([R,t],-1)

def let_me_look_at(pose, K, obj_center):
    image_center, _ = project_points(obj_center[None, :], pose, K)
    return let_me_look_at_2d(image_center[0], K)

def let_me_look_at_2d(image_center, K):
    f_raw = (K[0, 0] + K[1, 1]) / 2
    image_center = image_center - K[:2, 2]
    f_new = np.sqrt(np.linalg.norm(image_center, 2, 0) ** 2 + f_raw ** 2)
    image_center_ = image_center / f_raw
    R_new = look_at_rotation(image_center_)
    return R_new, f_new

def look_at_rotation(point):
    """
    @param point: point in normalized image coordinate not in pixels
    @return: R
    R @ x_raw -> x_lookat
    """
    x, y = point
    R1 = euler2mat(-np.arctan2(x, 1),0,0,'syxz')
    R2 = euler2mat(np.arctan2(y, 1),0,0,'sxyz')
    return R2 @ R1

def look_at_crop(img, K, pose, position, angle, scale, h, w):
    """rotate the image with "angle" and resize it with "scale", then crop the image on "position" with (h,w)"""
    # this function will return
    # 1) the resulted pose (pose_new) and intrinsic (K_new);
    # 2) pose_new = pose_compose(pose, pose_rect): "pose_rect" is the difference between the "pose_new" and the "pose"
    # 3) H is the homography that transform the "img" to "img_new"
    R_new, f_new = let_me_look_at_2d(position, K)
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
    R_new = R_z @ R_new
    f_new = f_new * scale
    K_new = np.asarray([[f_new, 0, w / 2], [0, f_new, h / 2], [0, 0, 1]], np.float32)

    H = K_new @ R_new @ np.linalg.inv(K)
    img_new = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    pose_rect = np.concatenate([R_new, np.zeros([3, 1])], 1).astype(np.float32)
    pose_new = pose_compose(pose, pose_rect)
    return img_new, K_new, pose_new, pose_rect, H

def pose_compose(pose0, pose1):
    """
    apply pose0 first, then pose1
    :param pose0:
    :param pose1:
    :return:
    """
    t = pose1[:,:3] @ pose0[:,3:] + pose1[:,3:]
    R = pose1[:,:3] @ pose0[:,:3]
    return np.concatenate([R,t], 1)


def pytorch3d_to_opencv(R, T):
    pass

def read_co3d_pose_from_dict(pose_dict, key):
    entry = pose_dict[key]
    vp = entry["viewpoint"]
    R = torch.tensor(vp["R"], dtype=torch.float)
    T = torch.tensor(vp["T"], dtype=torch.float)

    scale = 1
    margin = 0.05
    img_size = torch.tensor(entry["image"]["size"], dtype=torch.float)
    focal_length = torch.tensor(vp["focal_length"], dtype=torch.float)
    principal_point = torch.tensor(vp["principal_point"], dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(img_size.tolist())), dtype=torch.float)
        / 2.0
    )
    principal_point_px = (
        -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
    )
    focal_length_px = focal_length * half_image_size_wh_orig
    K = torch.FloatTensor([
        [focal_length_px[0], 0, principal_point_px[0]],
        [0, focal_length_px[1], principal_point_px[1]],
        [0, 0, 1]
    ])

    return R, T, K


def normalize_views(ref_poses, ref_Ks, verts, images, masks, ref_size=128, margin=0.05,
                    rectify_rot=True, input_pose=None, input_K=None,
                    add_rots=False, rots_list=None
                    ):
    size = ref_size
    scale_ratio = compute_normalized_ratio(verts)
    scaled_verts = verts.copy() * scale_ratio

    min_pt = np.min(scaled_verts, 0)
    max_pt = np.max(scaled_verts, 0)
    object_center = (max_pt + min_pt) / 2
    object_diameter = 2

    ref_poses[:, :, 3] = ref_poses[:, :, 3] * scale_ratio
    ref_cens = np.asarray([project_points(object_center[None], pose, K)[0][0] for pose,K in zip(ref_poses, ref_Ks)]) # rfn,2
    ref_cams = np.stack([pose_inverse(pose)[:,3] for pose in ref_poses], 0) # rfn, 3

    # ensure that the output reference images have the same scale
    ref_dist = np.linalg.norm(ref_cams - object_center[None,], 2, 1) # rfn
    ref_focal_look = np.asarray([let_me_look_at(pose, K, object_center)[1] for pose, K in zip(ref_poses, ref_Ks)]) # rfn
    ref_focal_new = size * (1 - margin) / object_diameter * ref_dist
    ref_scales = ref_focal_new / ref_focal_look


    # object_vert = np.asarray([0,0,1], np.float32)
    # ref_vert_2d = np.asarray([(pose[:,:3] @ object_vert)[:2] for pose in ref_poses])
    # mask = np.linalg.norm(ref_vert_2d,2,1)<1e-5
    # ref_vert_2d[mask] += 1e-5 * np.sign(ref_vert_2d[mask]) # avoid 0 vector
    # ref_vert_angle = -np.arctan2(ref_vert_2d[:,1],ref_vert_2d[:,0])-np.pi/2

    ref_vert_angle = np.zeros(len(images),np.float32)
    ref_imgs_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_masks_new, ref_imgs_rots = [], [], [], [], [], []
    for k in range(len(images)):
    #     ref_img = database.get_image(ref_ids[k])
        ref_img = images[k]
        if add_rots:
            ref_img_rot = np.stack([look_at_crop(ref_img, ref_Ks[k], ref_poses[k], ref_cens[k], ref_vert_angle[k]+rot, ref_scales[k], size, size)[0] for rot in rots_list],0)
            ref_imgs_rots.append(ref_img_rot)

        ref_img_new, ref_K_new, ref_pose_new, ref_pose_rect, ref_H = look_at_crop(
            ref_img, ref_Ks[k], ref_poses[k], ref_cens[k], ref_vert_angle[k], ref_scales[k], size, size)
        ref_imgs_new.append(ref_img_new)
        ref_Ks_new.append(ref_K_new)
        ref_poses_new.append(ref_pose_new)
        ref_Hs.append(ref_H)
    #     ref_mask = database.get_mask(ref_ids[k]).astype(np.float32)
        ref_mask = masks[k]
        ref_masks_new.append(cv2.warpPerspective(ref_mask, ref_H, (size, size), flags=cv2.INTER_LINEAR))

    ref_imgs_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_masks_new = \
        np.stack(ref_imgs_new, 0), np.stack(ref_Ks_new,0), np.stack(ref_poses_new,0), np.stack(ref_Hs,0), np.stack(ref_masks_new,0)

    return ref_imgs_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_masks_new, scaled_verts

def main():
    ref_size = 128
    margin = 0.05
    co3d_raw_dir = "/datasets01/co3d/081922"
    co3d_normalized_dir = "/checkpoint/haotang/data/co3d_normalized"
    categories = ["toyplane"]
    pose_dict = {}
    scene_paths = []

    # Load meta info about the dataset
    for cat in categories:
        scene_ids = os.listdir(os.path.join(co3d_raw_dir, cat))
        valid_scene_ids = []
        frame_annotation_path = os.path.join(co3d_raw_dir, cat, "frame_annotations.jgz")
        with gzip.open(frame_annotation_path, "rt", encoding="utf8") as zipfile:
            frame_annots_list = json.load(zipfile)
            for entry in frame_annots_list:
                frame_path = entry["image"]["path"]
                pose_dict[frame_path] = entry

        for scene_id in scene_ids:
            if os.path.isdir(os.path.join(co3d_raw_dir, cat, scene_id)):
                valid_scene_ids.append(scene_id)

        scene_paths.extend([os.path.join(cat, scene_id) for scene_id in valid_scene_ids])

    # Process per scan
    for scene_path in tqdm(scene_paths, total=len(scene_paths)):
        mask_dir = os.path.join(scene_path, "masks")
        image_dir = os.path.join(scene_path, "images")
        pointcloud_path = os.path.join(co3d_raw_dir, scene_path, "pointcloud.ply")
        frame_ids = [f.split(".jpg")[0] for f in os.listdir(os.path.join(co3d_raw_dir, image_dir))]
        if not os.path.exists(pointcloud_path):
            continue
        verts, _ = pytorch3d.io.load_ply(os.path.join(co3d_raw_dir, scene_path, "pointcloud.ply"))

        RTs = []
        Ks = []
        images, masks = [], []
        for frame_id in frame_ids:
            images.append(plt.imread(os.path.join(co3d_raw_dir, image_dir, f"{frame_id}.jpg")))
            masks.append(plt.imread(os.path.join(co3d_raw_dir, mask_dir, f"{frame_id}.png")))
            R, T, K = read_co3d_pose_from_dict(pose_dict, os.path.join(scene_path, "images", f"{frame_id}.jpg"))

            # pytorch3d uses column vector, need to transpose R
            RT = torch.cat([R.T, T.unsqueeze(1)], dim=1)
            # convert from pytorch3d camera view coordinate to opencv
            RT = torch.from_numpy(np.diag([-1,-1,1])).float() @ RT

            RTs.append(RT)
            Ks.append(K)

        RTs = torch.stack(RTs).numpy()
        Ks = torch.stack(Ks).numpy()
        ref_imgs_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_masks_new, verts_new = normalize_views(
            RTs, Ks, verts.numpy(), images, masks, 
            ref_size=ref_size, margin=margin
        )

        output_dir = os.path.join(co3d_normalized_dir, scene_path)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        for i, frame_id in enumerate(frame_ids):
            plt.imsave(os.path.join(output_dir, "masks", f"{frame_id}.png"), ref_masks_new[i])
            plt.imsave(os.path.join(output_dir, "images", f"{frame_id}.jpg"), ref_imgs_new[i])
        
        pytorch3d.io.save_ply(os.path.join(co3d_normalized_dir, scene_path, "pointcloud.ply"), torch.from_numpy(verts_new))
        with open(os.path.join(output_dir, "meta_info.pkl"), 'wb') as f:
            pickle.dump([ref_Ks_new, ref_poses_new], f)
                

if __name__ == "__main__":
    main()