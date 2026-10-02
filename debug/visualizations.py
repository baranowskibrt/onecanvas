import os
import numpy as np
import open3d as o3d
import torch
import geometry
from geometry import lift_to_3d_with_intrinsics
from PIL import Image


def run_debug_visualization(images, depths, poses, intrinsics, center_point, generate_pcd, scene_id=None):
    # Ensure inputs are torch tensors
    # Using as_tensor avoids copying if they are already tensors
    images = torch.from_numpy(np.array(images)).to(depths.device)
    device = depths.device
    print(images.shape)
    
    # 1. Save the PLY (Geometry only)
    if generate_pcd:
        # Create identity matrix in torch
        align_mat = torch.eye(4, device=device)
        align_mat[:3, 3] = -torch.as_tensor(center_point, device=device)
        # save_ply_data handles the conversion to numpy for Open3D internally
        save_ply_data(images, depths, poses, intrinsics, align_mat, output_path="debug_reconstruction.ply")

    combined_pts = []
    combined_vals = []
    
    # 2. Lift pixels to 3D
    for i in range(len(depths)):
        # Torch-native finite check
        if not torch.isfinite(poses[i]).all(): 
            continue
            
        curr_k = intrinsics[i] if isinstance(intrinsics[0], (list, tuple)) else intrinsics
        
        # Use our updated lift function
        pts, vals = lift_to_3d_with_intrinsics(images[i], depths[i], poses[i], curr_k, stride=2)
        combined_pts.append(pts)
        combined_vals.append(vals)

    if not combined_pts:
        print("No valid points to visualize")
        return

    all_pts = torch.cat(combined_pts, dim=0)
    all_vals = torch.cat(combined_vals, dim=0)

    # 3. Project to panorama
    # Center point converted to tensor
    cp_tensor = torch.as_tensor(center_point, device=all_pts.device)

    # 2:1 aspect ratio for 360x180 degree panorama
    w_proj, h_proj = 1600, 800
    indices, mask, _ = geometry.compute_equirectangular_mapping(
        all_pts,
        width=w_proj,
        height=h_proj,
        center_point=cp_tensor
    )
    total_w = w_proj
    canvas = torch.zeros((h_proj * total_w, 3), dtype=torch.float32, device=device)
    counts = torch.zeros((h_proj * total_w, 1), dtype=torch.float32, device=device)
    
    # Filter only valid indices based on the projection mask
    valid_indices = indices[mask].long()
    valid_vals = all_vals[mask].float()
    
    # Perform atomic addition (equivalent to np.add.at)
    canvas.index_add_(0, valid_indices, valid_vals)
    counts.index_add_(0, valid_indices, torch.ones((valid_vals.shape[0], 1), device=device))
    
    # Normalize by counts to get the average (avoid division by zero)
    final_img_flat = canvas / torch.clamp(counts, min=1.0)
    final_img = final_img_flat.reshape(h_proj, total_w, 3)
    
    # 4. Save debug image (Convert to CPU numpy only for PIL saving)
    final_img_np = final_img.cpu().numpy()
    debug_image = Image.fromarray((np.clip(final_img_np, 0, 1) * 255).astype(np.uint8))
    if scene_id is not None:
        os.makedirs("images", exist_ok=True)
        out_path = f"images/debug_visualization_{scene_id}.png"
    else:
        out_path = "room_360_debug.png"
    debug_image.save(out_path)

    print(f"Debug visualization saved to {out_path}")
    return final_img_np

def save_ply_data(images, depths, poses, intrinsics, alignment_matrix, output_path="scene.ply", stride=1, subsample_pct=0.1):
    combined_pcd = o3d.geometry.PointCloud()
    
    # Use depths for the loop count
    num_entries = len(depths)
    
    # Ensure alignment_matrix is a numpy array for Open3D
    alignment_np = np.asarray(alignment_matrix)
    
    for i in range(num_entries):
        # Torch-based finite check
        curr_pose = torch.as_tensor(poses[i])
        if i >= len(poses) or not torch.isfinite(curr_pose).all(): 
            continue
            
        # Intrinsic handling logic
        curr_k = intrinsics[i] if isinstance(intrinsics[0], (list, tuple)) else intrinsics
        
        # This calls your Torch-based lift function
        # Returning torch tensors for pts_world and rgb_vals
        pts_world, rgb_vals = lift_to_3d_with_intrinsics(
            images[i], 
            depths[i], 
            curr_pose, 
            curr_k, 
            stride=stride
        )

        # Convert to numpy only for Open3D ingestion
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb_vals.cpu().numpy())
        
        # Apply the alignment transformation
        pcd.transform(alignment_np)
        combined_pcd += pcd
        
    if 0 < subsample_pct < 1.0:
        combined_pcd = combined_pcd.random_down_sample(subsample_pct)
        
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved point cloud to {output_path}")
    


def features_to_pca_rgb(features):
    if torch.is_tensor(features):
        features = features.detach().cpu().to(torch.float32).numpy()
    
    feat_centered = features - features.mean(axis=0)
    
    # Using SVD for PCA to handle high dimensional features
    u, s, vh = np.linalg.svd(feat_centered, full_matrices=False)
    pca_feats = feat_centered @ vh[:3, :].T
    
    min_val = pca_feats.min(axis=0)
    max_val = pca_feats.max(axis=0)
    
    pca_rgb = (pca_feats - min_val) / (max_val - min_val + 1e-8)
    return pca_rgb


def save_feature_ply_data(
    feature_stack,
    depths,
    poses,
    intrinsics,
    image_dims,
    alignment_matrix,
    output_path="features.ply",
    layer_idx=0,
    subsample_pct=0.1,
):
    """Save 3D point cloud coloured by PCA of backprojected feature vectors.

    Parameters
    ----------
    feature_stack : Tensor [num_imgs, num_layers, H_f, W_f, D]  or
                    list of such tensors (one per scene/batch item).
                    Only ``layer_idx`` is used.
    depths        : Tensor [num_imgs, H_d, W_d]  (uint16 mm or float32 m).
    poses         : Tensor [num_imgs, 4, 4]  camera-to-world.
    intrinsics    : Tensor [num_imgs, 4]  (fx, fy, cx, cy) at original resolution,
                    or a list of per-image tensors/lists.
    image_dims    : list of (W, H) tuples – original image size for each view.
    alignment_matrix : [4, 4] tensor/array  (e.g. center-translate).
    layer_idx     : which feature layer to visualise (0 = first/only).
    subsample_pct : fraction of points to keep (0–1); 1.0 keeps all.

    Returns
    -------
    open3d.geometry.PointCloud
    """
    import torch.nn.functional as F

    # Accept single scene tensor or list of scene tensors
    if not isinstance(feature_stack, (list, tuple)):
        feature_stack = [feature_stack]

    # Flatten to per-image lists (mirrors project_features_to_grid_adaptive)
    img_feats, img_depths, img_poses, img_ks, img_dims = [], [], [], [], []
    flat_idx = 0
    for scene_feats in feature_stack:
        n_imgs = scene_feats.shape[0]
        for j in range(n_imgs):
            img_feats.append(scene_feats[j, layer_idx].cpu())   # [H_f, W_f, D] — always CPU
            img_depths.append(depths[flat_idx])
            img_poses.append(poses[flat_idx])
            k = intrinsics[flat_idx] if hasattr(intrinsics[0], '__len__') else intrinsics
            img_ks.append(k)
            img_dims.append(image_dims[flat_idx])
            flat_idx += 1

    alignment_np = np.asarray(
        alignment_matrix.cpu() if torch.is_tensor(alignment_matrix) else alignment_matrix,
        dtype=np.float64,
    )

    all_pts_np: list[np.ndarray] = []
    all_feat_np: list[np.ndarray] = []

    for i in range(len(img_feats)):
        curr_pose = torch.as_tensor(img_poses[i])
        if not torch.isfinite(curr_pose).all():
            continue

        f_map = img_feats[i].float()        # [H_f, W_f, D]
        feat_h, feat_w, D = f_map.shape
        device = f_map.device

        # Resize depth to feature-map resolution
        curr_depth = img_depths[i].to(torch.float32)
        d_tmp = curr_depth.unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(d_tmp, size=(feat_h, feat_w), mode='nearest').squeeze()

        # Depth is MILLIMETRES (API-wide contract; the debug path is fed the same
        # mm depths as reproject_scene) -> metres.
        depth_m = depth_resized / 1000.0

        # Scale intrinsics from original resolution to feature-map resolution
        orig_w, orig_h = float(img_dims[i][0]), float(img_dims[i][1])
        k = img_ks[i]
        sw, sh = feat_w / orig_w, feat_h / orig_h
        fx = float(k[0]) * sw
        fy = float(k[1]) * sh
        cx_ = float(k[2]) * sw
        cy_ = float(k[3]) * sh

        # Back-project each feature pixel to 3D
        v_grid, u_grid = torch.meshgrid(
            torch.arange(feat_h, device=device),
            torch.arange(feat_w, device=device),
            indexing='ij',
        )
        z = depth_m.flatten()
        u = u_grid.flatten().float()
        v = v_grid.flatten().float()

        x = (u - cx_) * z / fx
        y = (v - cy_) * z / fy

        pts_cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1).to(curr_pose.dtype).to(curr_pose.device)
        pts_world = (pts_cam @ curr_pose.T)[:, :3]

        valid = (z > 0.1) & (z < 50.0) & torch.isfinite(pts_world).all(dim=-1)
        if valid.sum() == 0:
            continue

        all_pts_np.append(pts_world[valid].cpu().numpy())
        all_feat_np.append(f_map.reshape(-1, D)[valid.to(device)].cpu().numpy())

    if not all_pts_np:
        print("[feature PLY] No valid points — file not written.")
        return None

    pts_cat = np.concatenate(all_pts_np, axis=0)
    feat_cat = np.concatenate(all_feat_np, axis=0)

    # Global PCA: D-dim feature vector → RGB colour
    pca_rgb = features_to_pca_rgb(feat_cat)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cat.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(pca_rgb, 0.0, 1.0).astype(np.float64))
    pcd.transform(alignment_np)

    if 0 < subsample_pct < 1.0:
        pcd = pcd.random_down_sample(subsample_pct)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[feature PLY] saved {len(pcd.points):,} points → {output_path}")
    return pcd


