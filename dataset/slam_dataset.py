import csv
import math
import os
import sys
from pathlib import Path
from typing import List
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import wandb
from numpy.linalg import inv
from rich import print
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.dataloaders import dataset_factory
from eval.eval_traj_utils import absolute_error, plot_trajectories, relative_error
from utils.config import Config

from utils.tools import (
    deskewing,
    get_time,
    plot_timing_detail,
    tranmat_close_to_identity,
    transform_torch,
    voxel_down_sample_torch,
)

class SLAMDataset(Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.silence = config.silence
        self.dtype = config.dtype
        self.device = config.device
        self.run_path = config.run_path
        max_frame_number: int = 100000

        self.poses_ts = None
        self.gt_poses = None
        self.calib = {"Tr": np.eye(4)}
        
        self.loader = None
        if config.use_dataloader: 

            self.loader = dataset_factory(
                dataloader=config.data_loader_name,
                data_dir=Path(config.pc_path),
                sequence=config.data_loader_seq,
                topic=config.data_loader_seq,
            )
            config.end_frame = min(len(self.loader), config.end_frame)
            used_frame_count = int((config.end_frame - config.begin_frame) / config.step_frame)
            self.total_pc_count = used_frame_count
            max_frame_number = self.total_pc_count
            if hasattr(self.loader, 'gt_poses'):
                self.gt_poses = self.loader.gt_poses[config.begin_frame:config.end_frame:config.step_frame]
                self.gt_pose_provided = True
            else:
                self.gt_pose_provided = False
            if hasattr(self.loader, 'calibration'):
                self.calib["Tr"][:3, :4] = self.loader.calibration["Tr"].reshape(3, 4)
        else:
            if config.pc_path != "":
                from natsort import natsorted
                # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
                self.pc_filenames = natsorted(os.listdir(config.pc_path))    
                self.total_pc_count_in_folder = len(self.pc_filenames)
                config.end_frame = min(config.end_frame, self.total_pc_count_in_folder)
                self.pc_filenames = self.pc_filenames[config.begin_frame:config.end_frame:config.step_frame]
                self.total_pc_count = len(self.pc_filenames)
                max_frame_number = self.total_pc_count
            else:
                if not config.run_with_ros:
                    sys.exit("Input point cloud directory is not specified")

            self.gt_pose_provided = True
            if config.pose_path == "":
                self.gt_pose_provided = False
            else:
                if config.calib_path != "":
                    self.calib = read_kitti_format_calib(config.calib_path)
                poses_uncalib = None
                if config.pose_path.endswith("txt"):
                    poses_uncalib = read_kitti_format_poses(config.pose_path)
                    if poses_uncalib is None:
                        poses_uncalib, poses_ts = read_tum_format_poses(config.pose_path)
                        self.poses_ts = np.array(poses_ts[config.begin_frame:config.end_frame:config.step_frame])
                    poses_uncalib = np.array(poses_uncalib[config.begin_frame:config.end_frame:config.step_frame])
                if poses_uncalib is None:
                    sys.exit("Wrong pose file format")
                self.gt_poses = apply_kitti_format_calib(poses_uncalib, inv(self.calib["Tr"]))
                if config.first_frame_ref:
                    gt_poses_first_inv = inv(self.gt_poses[0])
                    for i in range(self.total_pc_count):
                        self.gt_poses[i] = gt_poses_first_inv @ self.gt_poses[i]

                if self.total_pc_count > 2000:
                    config.local_map_context = True

        self.odom_poses = None
        if config.track_on:
            self.odom_poses = np.broadcast_to(np.eye(4), (max_frame_number, 4, 4)).copy()

        self.pgo_poses = None
        if config.pgo_on:
            self.pgo_poses = np.broadcast_to(np.eye(4), (max_frame_number, 4, 4)).copy()

        self.travel_dist = np.zeros(max_frame_number) 
        self.time_table = []

        self.processed_frame: int = 0
        self.shift_ts: float = 0.0
        self.lose_track: bool = False
        self.consecutive_lose_track_frame: int = 0
        self.color_available: bool = False
        self.intensity_available: bool = False
        self.color_scale: float = 255.0
        self.last_pose_ref = np.eye(4)
        self.last_odom_tran = np.eye(4)
        self.cur_pose_ref = np.eye(4)
        self.stop_count: int = 0
        self.stop_status = False

        if self.config.kitti_correction_on:
            self.last_odom_tran[0, 3] = (
                self.config.max_range * 1e-2
            )
            self.color_scale = 1.0

        self.cur_frame_o3d = o3d.geometry.PointCloud()
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.map_down_o3d = o3d.geometry.PointCloud()
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()

        self.static_mask = None

        self.cur_point_cloud_torch = None
        self.cur_point_ts_torch = None
        self.cur_sem_labels_torch = None
        self.cur_sem_labels_full = None

        self.cur_source_points = None
        self.cur_source_normals = None
        self.cur_source_colors = None

    def read_frame_ros(self, msg):

        from utils import point_cloud
        self.cur_pose_ref = np.eye(4)
        self.cur_pose_torch = torch.tensor(
            self.cur_pose_ref, device=self.device, dtype=self.dtype
        )

        points, point_ts = point_cloud2.read_point_cloud(msg)

        if point_ts is None:
            print(
                "The point cloud message does not contain the time stamp field"
            )

        self.cur_point_cloud_torch = torch.tensor(
            points, device=self.device, dtype=self.dtype
        )

        if self.config.deskew:
            self.get_point_ts(point_ts)
    def read_frame_with_loader(self, frame_id):

        self.set_ref_pose(frame_id)

        frame_id_in_folder = self.config.begin_frame + frame_id * self.config.step_frame
        data = self.loader[frame_id_in_folder]
    
        point_ts = None
        if isinstance(data, tuple):
            points, point_ts = data
        else:
            points = data
        self.cur_point_cloud_torch = torch.tensor(points, device=self.device, dtype=self.dtype)

        if self.config.deskew: 
            self.get_point_ts(point_ts)

    def read_frame(self, frame_id):

        self.set_ref_pose(frame_id)
        
        point_ts = None
        frame_filename = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])
        if not self.silence:
            print(frame_filename)
        if not self.config.semantic_on:
            point_cloud, point_ts = read_point_cloud(
                frame_filename, self.config.color_channel
            )
            if self.config.color_channel > 0:
                point_cloud[:, -self.config.color_channel :] /= self.color_scale
            self.cur_sem_labels_torch = None
        else:
            label_filename = os.path.join(
                self.config.label_path,
                self.pc_filenames[frame_id].replace("bin", "label"),
            )
            point_cloud, sem_labels, sem_labels_reduced = read_semantic_point_label(
                frame_filename, label_filename
            )
            self.cur_sem_labels_torch = torch.tensor(
                sem_labels_reduced, device=self.device, dtype=torch.int
            )
            self.cur_sem_labels_full = torch.tensor(
                sem_labels, device=self.device, dtype=torch.int
            )

        self.cur_point_cloud_torch = torch.tensor(
            point_cloud, device=self.device, dtype=self.dtype
        )

        if self.config.deskew:
            self.get_point_ts(point_ts)

    def get_point_ts(self, point_ts=None):
        if self.config.deskew:
            if point_ts is not None and min(point_ts) < 1.0:
                if not self.silence:
                    print("Pointwise timestamp available")
                self.cur_point_ts_torch = torch.tensor(
                    point_ts, device=self.device, dtype=self.dtype
                )
            else: # point_ts not available, guess the ts
                if (
                    self.cur_point_cloud_torch.shape[0] == 64 * 1024
                ):  # for Ouster 64-beam LiDAR
                    if not self.silence:
                        print("Ouster-64 point cloud deskewed")
                    self.cur_point_ts_torch = (
                        (torch.floor(torch.arange(64 * 1024) / 64) / 1024)
                        .reshape(-1, 1)
                        .to(self.cur_point_cloud_torch)
                    )
                elif (
                    self.cur_point_cloud_torch.shape[0] == 128 * 1024
                ):  # for Ouster 128-beam LiDAR
                    if not self.silence:
                        print("Ouster-128 point cloud deskewed")
                    self.cur_point_ts_torch = (
                        (torch.floor(torch.arange(128 * 1024) / 128) / 1024)
                        .reshape(-1, 1)
                        .to(self.cur_point_cloud_torch)
                    )
                else:
                    yaw = -torch.atan2(
                        self.cur_point_cloud_torch[:, 1],
                        self.cur_point_cloud_torch[:, 0],
                    )
                    if self.config.lidar_type_guess == "velodyne":
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 1.0)  # [0,1]
                        if not self.silence:
                            print("Velodyne point cloud deskewed")
                    else:
                        self.cur_point_ts_torch = 0.5 * (
                            yaw / math.pi + 0.5
                        )
                        self.cur_point_ts_torch[
                            self.cur_point_ts_torch < 0
                        ] += 1.0  # [0,1]
                        if not self.silence:
                            print("HESAI point cloud deskewed")
    
    def set_ref_pose(self, frame_id):
        if self.gt_pose_provided:
            self.cur_pose_ref = self.gt_poses[frame_id]
        else:
            self.cur_pose_ref = np.eye(4)
        self.cur_pose_torch = torch.tensor(
            self.cur_pose_ref, device=self.device, dtype=self.dtype
        )

    def preprocess_frame(self):
        frame_id = self.processed_frame
        cur_pose_init_guess = self.cur_pose_ref
        if frame_id == 0:
            if self.config.track_on:
                self.odom_poses[frame_id] = self.cur_pose_ref
            if self.config.pgo_on:
                self.pgo_poses[frame_id] = self.cur_pose_ref
            self.travel_dist[frame_id] = 0.0
            self.last_pose_ref = self.cur_pose_ref
        elif frame_id > 0:
            if self.config.uniform_motion_on and not self.lose_track:
                cur_pose_init_guess = (
                    self.last_pose_ref @ self.last_odom_tran
                )
            else:
                cur_pose_init_guess = self.last_pose_ref

            if not self.config.track_on and self.gt_pose_provided:
                cur_pose_init_guess = self.gt_poses[frame_id]
            self.cur_pose_guess_torch = torch.tensor(
                cur_pose_init_guess, dtype=torch.float64, device=self.device
            )

        if self.config.adaptive_range_on:
            pc_max_bound, _ = torch.max(self.cur_point_cloud_torch[:, :3], dim=0)
            pc_min_bound, _ = torch.min(self.cur_point_cloud_torch[:, :3], dim=0)

            min_x_range = min(torch.abs(pc_max_bound[0]), torch.abs(pc_min_bound[0]))
            min_y_range = min(torch.abs(pc_max_bound[1]), torch.abs(pc_min_bound[1]))
            max_x_y_min_range = max(min_x_range, min_y_range)

            crop_max_range = min(self.config.max_range, 2.0 * max_x_y_min_range)
        else:
            crop_max_range = self.config.max_range

        train_voxel_m = (
            crop_max_range / self.config.max_range
        ) * self.config.vox_down_m
        source_voxel_m = (
            crop_max_range / self.config.max_range
        ) * self.config.source_vox_down_m

        original_count = self.cur_point_cloud_torch.shape[0]
        if original_count < 10:
            print("[bold red]Not enough input point cloud, skip this frame[/bold red]")
            if self.config.track_on:
                self.odom_poses[frame_id] = cur_pose_init_guess
            if self.config.pgo_on:
                self.pgo_poses[frame_id] = cur_pose_init_guess
            return False

        if self.config.rand_downsample:
            kept_count = int(original_count * self.config.rand_down_r)
            idx = torch.randint(0, original_count, (kept_count,), device=self.device)
        else:
            idx = voxel_down_sample_torch(
                self.cur_point_cloud_torch[:, :3], train_voxel_m
            )
        self.cur_point_cloud_torch = self.cur_point_cloud_torch[idx]
        if self.cur_point_ts_torch is not None:
            self.cur_point_ts_torch = self.cur_point_ts_torch[idx]
        if self.cur_sem_labels_torch is not None:
            self.cur_sem_labels_torch = self.cur_sem_labels_torch[idx]
            self.cur_sem_labels_full = self.cur_sem_labels_full[idx]

        if self.cur_sem_labels_torch is not None:
            self.cur_point_cloud_torch, self.cur_sem_labels_torch = filter_sem_kitti(
                self.cur_point_cloud_torch,
                self.cur_sem_labels_torch,
                self.cur_sem_labels_full,
                True,
                self.config.filter_moving_object,
            )
        else:
            self.cur_point_cloud_torch, self.cur_point_ts_torch = crop_frame(
                self.cur_point_cloud_torch,
                self.cur_point_ts_torch,
                self.config.min_z,
                self.config.max_z,
                self.config.min_range,
                crop_max_range,
            )

        if self.config.kitti_correction_on:
            self.cur_point_cloud_torch = intrinsic_correct(
                self.cur_point_cloud_torch, self.config.correction_deg
            )

        if frame_id > 0:

            cur_source_torch = (
                self.cur_point_cloud_torch.clone()
            )
            idx = voxel_down_sample_torch(cur_source_torch[:, :3], source_voxel_m)
            cur_source_torch = cur_source_torch[idx]
            self.cur_source_points = cur_source_torch[:, :3]
            if self.config.color_on:
                self.cur_source_colors = cur_source_torch[:, 3:]

            if self.cur_point_ts_torch is not None:
                cur_ts = self.cur_point_ts_torch.clone()
                cur_source_ts = cur_ts[idx]
            else:
                cur_source_ts = None
            if self.config.deskew and not self.lose_track:
                self.cur_source_points = deskewing(
                    self.cur_source_points,
                    cur_source_ts,
                    torch.tensor(
                        self.last_odom_tran, device=self.device, dtype=self.dtype
                    )
                )
        return True

    def update_odom_pose(self, cur_pose_torch: torch.tensor): 
        
        cur_frame_id = self.processed_frame
        assert (cur_frame_id > 0), "This function needs to be used from at least the second frame"
        self.cur_pose_torch = cur_pose_torch.detach()
        self.cur_pose_ref = self.cur_pose_torch.cpu().numpy()
        self.last_odom_tran = inv(self.last_pose_ref) @ self.cur_pose_ref
        if tranmat_close_to_identity(
            self.last_odom_tran, 1e-3, self.config.voxel_size_m * 0.1
        ):
            self.stop_count += 1
        else:
            self.stop_count = 0

        if self.stop_count > self.config.stop_frame_thre:
            self.stop_status = True
            if not self.silence:
                print("Robot stopped")
        else:
            self.stop_status = False

        if self.config.pgo_on:
            self.pgo_poses[cur_frame_id] = self.cur_pose_ref
        if self.odom_poses is not None:
            cur_odom_pose = self.odom_poses[cur_frame_id-1] @ self.last_odom_tran
            self.odom_poses[cur_frame_id] = cur_odom_pose

        cur_frame_travel_dist = np.linalg.norm(self.last_odom_tran[:3, 3])
        if (
            cur_frame_travel_dist > self.config.surface_sample_range_m * 40.0
        ):
            self.lose_track = True
            self.write_results()
            sys.exit("Too large translation in one frame, system failed")

        accu_travel_dist = self.travel_dist[cur_frame_id-1] + cur_frame_travel_dist
        self.travel_dist[cur_frame_id] = accu_travel_dist
        if not self.silence:
            print("Accumulated travel distance (m): %f" % accu_travel_dist)
        
        self.last_pose_ref = self.cur_pose_ref
        if self.config.deskew and not self.lose_track:
            self.cur_point_cloud_torch = deskewing(
                self.cur_point_cloud_torch,
                self.cur_point_ts_torch,
                torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype),
            )

        if self.lose_track:
            self.consecutive_lose_track_frame += 1
        else:
            self.consecutive_lose_track_frame = 0

        if self.consecutive_lose_track_frame > 10:
            self.write_results()
            sys.exit("Lose track for a long time, system failed") 

    def update_poses_after_pgo(self, pgo_cur_pose, pgo_poses):
        self.cur_pose_ref = pgo_cur_pose
        self.last_pose_ref = pgo_cur_pose
        self.pgo_poses[:self.processed_frame+1] = pgo_poses

    def update_o3d_map(self):

        frame_down_torch = self.cur_point_cloud_torch
        frame_o3d = o3d.geometry.PointCloud()
        frame_points_np = (
            frame_down_torch[:, :3].detach().cpu().numpy().astype(np.float64)
        )

        frame_o3d.points = o3d.utility.Vector3dVector(frame_points_np)
        if self.config.dynamic_filter_on and self.static_mask is not None:
            static_mask = self.static_mask.detach().cpu().numpy()
            frame_colors_np = np.ones_like(frame_points_np) * 0.7
            frame_colors_np[~static_mask, 1:] = 0.0
            frame_colors_np[~static_mask, 0] = 1.0
            frame_o3d.colors = o3d.utility.Vector3dVector(
                frame_colors_np.astype(np.float64)
            )

        frame_o3d = frame_o3d.transform(self.cur_pose_ref)

        if self.config.color_channel > 0:
            frame_colors_np = (
                frame_down_torch[:, 3:].detach().cpu().numpy().astype(np.float64)
            )
            if self.config.color_channel == 1:
                frame_colors_np = np.repeat(frame_colors_np.reshape(-1, 1), 3, axis=1)
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_colors_np)
        elif self.cur_sem_labels_torch is not None:
            frame_label_torch = self.cur_sem_labels_torch
            frame_label_np = frame_label_torch.detach().cpu().numpy()
            frame_label_color = [
                sem_kitti_color_map[sem_label] for sem_label in frame_label_np
            ]
            frame_label_color_np = (
                np.asarray(frame_label_color, dtype=np.float64) / 255.0
            )
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_label_color_np)

        self.cur_frame_o3d = frame_o3d
        if self.cur_frame_o3d.has_points():
            self.cur_bbx = self.cur_frame_o3d.get_axis_aligned_bounding_box()

        cur_max_z = self.cur_bbx.get_max_bound()[-1]
        cur_min_z = self.cur_bbx.get_min_bound()[-1]

        bbx_center = self.cur_pose_ref[:3, 3]
        bbx_min = np.array(
            [
                bbx_center[0] - self.config.max_range,
                bbx_center[1] - self.config.max_range,
                cur_min_z,
            ]
        )
        bbx_max = np.array(
            [
                bbx_center[0] + self.config.max_range,
                bbx_center[1] + self.config.max_range,
                cur_max_z,
            ]
        )

        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)

    def write_results_log(self):
        log_folder = "log"
        frame_str = str(self.processed_frame)
        
        if self.config.track_on:
            write_traj_as_o3d(
                self.odom_poses[:self.processed_frame+1],
                os.path.join(self.run_path, log_folder, frame_str + "_odom_poses.ply"),
            )
        if self.config.pgo_on:
            write_traj_as_o3d(
                self.pgo_poses[:self.processed_frame+1],
                os.path.join(self.run_path, log_folder, frame_str + "_slam_poses.ply"),
            )
        if self.gt_pose_provided:
            write_traj_as_o3d(
                self.gt_poses[:self.processed_frame+1],
                os.path.join(self.run_path, log_folder, frame_str + "_gt_poses.ply"),
            )

    def get_poses_np_for_vis(self):
        odom_poses = None
        if self.odom_poses is not None:
            odom_poses = self.odom_poses[:self.processed_frame+1]
        gt_poses = None
        if self.gt_poses is not None:
            gt_poses = self.gt_poses[:self.processed_frame+1]
        pgo_poses = None
        if self.pgo_poses is not None:
            pgo_poses = self.pgo_poses[:self.processed_frame+1]
        
        return odom_poses, gt_poses, pgo_poses

    def write_results(self):
        odom_poses = self.odom_poses[:self.processed_frame+1]
        odom_poses_out = apply_kitti_format_calib(odom_poses, self.calib["Tr"])
        write_kitti_format_poses(os.path.join(self.run_path, "odom_poses"), odom_poses_out)
        write_tum_format_poses(os.path.join(self.run_path, "odom_poses"), odom_poses_out, self.poses_ts, 0.1*self.config.step_frame)
        write_traj_as_o3d(odom_poses, os.path.join(self.run_path, "odom_poses.ply"))

        if self.config.pgo_on:
            pgo_poses = self.pgo_poses[:self.processed_frame+1]
            slam_poses_out = apply_kitti_format_calib(pgo_poses, self.calib["Tr"])
            write_kitti_format_poses(
                os.path.join(self.run_path, "slam_poses"), slam_poses_out
            )
            write_tum_format_poses(
                os.path.join(self.run_path, "slam_poses"), slam_poses_out, self.poses_ts, 0.1*self.config.step_frame
            )
            write_traj_as_o3d(pgo_poses, os.path.join(self.run_path, "slam_poses.ply"))
        
        # timing report
        time_table = np.array(self.time_table)
        mean_time_s = np.sum(time_table) / self.processed_frame * 1.0
        mean_time_without_init_s = np.sum(time_table[1:]) / (self.processed_frame-1) * 1.0
        if not self.silence:
            print("Consuming time per frame        (s):", f"{mean_time_without_init_s:.3f}")
            print("Calculated over %d frames" % self.processed_frame)
        np.save(
            os.path.join(self.run_path, "time_table.npy"), time_table
        )  # save detailed time table

        pose_eval = None

        # pose estimation evaluation report
        if self.gt_pose_provided:
            gt_poses = self.gt_poses[:self.processed_frame+1]
            write_traj_as_o3d(gt_poses, os.path.join(self.run_path, "gt_poses.ply"))

            print("Odometry evaluation:")
            avg_tra, avg_rot = relative_error(gt_poses, odom_poses)
            ate_rot, ate_trans, align_mat = absolute_error(
                gt_poses, odom_poses, self.config.eval_traj_align
            )
            if avg_tra == 0:  # for rgbd dataset (shorter sequence)
                print("Absoulte Trajectory Error      (cm):", f"{ate_trans*100.0:.3f}")
            else:
                print("Average Translation Error       (%):", f"{avg_tra:.3f}")
                print("Average Rotational Error (deg/100m):", f"{avg_rot*100.0:.3f}")
                print("Absoulte Trajectory Error       (m):", f"{ate_trans:.3f}")

            if self.config.wandb_vis_on:
                wandb_log_content = {
                    "Average Translation Error [%]": avg_tra,
                    "Average Rotational Error [deg/m]": avg_rot,
                    "Absoulte Trajectory Error [m]": ate_trans,
                    "Absoulte Rotational Error [deg]": ate_rot,
                    "Consuming time per frame [s]": mean_time_without_init_s,
                }
                wandb.log(wandb_log_content)

            if self.config.pgo_on:
                print("SLAM evaluation:")
                avg_tra_slam, avg_rot_slam = relative_error(
                    gt_poses, pgo_poses
                )
                ate_rot_slam, ate_trans_slam, align_mat_slam = absolute_error(
                    gt_poses, pgo_poses, self.config.eval_traj_align
                )
                if avg_tra_slam == 0:  # for rgbd dataset (shorter sequence)
                    print(
                        "Absoulte Trajectory Error      (cm):",
                        f"{ate_trans_slam*100.0:.3f}",
                    )
                else:
                    print("Average Translation Error       (%):", f"{avg_tra_slam:.3f}")
                    print(
                        "Average Rotational Error (deg/100m):",
                        f"{avg_rot_slam*100.0:.3f}",
                    )
                    print(
                        "Absoulte Trajectory Error       (m):", f"{ate_trans_slam:.3f}"
                    )

                if self.config.wandb_vis_on:
                    wandb_log_content = {
                        "SLAM Average Translation Error [%]": avg_tra_slam,
                        "SLAM Average Rotational Error [deg/m]": avg_rot_slam,
                        "SLAM Absoulte Trajectory Error [m]": ate_trans_slam,
                        "SLAM Absoulte Rotational Error [deg]": ate_rot_slam,
                    }
                    wandb.log(wandb_log_content)

            csv_columns = [
                "Average Translation Error [%]",
                "Average Rotational Error [deg/m]",
                "Absoulte Trajectory Error [m]",
                "Absoulte Rotational Error [deg]",
                "Consuming time per frame [s]",
                "Frame count",
            ]
            pose_eval = [
                {
                    csv_columns[0]: avg_tra,
                    csv_columns[1]: avg_rot,
                    csv_columns[2]: ate_trans,
                    csv_columns[3]: ate_rot,
                    csv_columns[4]: mean_time_without_init_s,
                    csv_columns[5]: int(self.processed_frame),
                }
            ]
            if self.config.pgo_on:
                slam_eval_dict = {
                    csv_columns[0]: avg_tra_slam,
                    csv_columns[1]: avg_rot_slam,
                    csv_columns[2]: ate_trans_slam,
                    csv_columns[3]: ate_rot_slam,
                    csv_columns[4]: mean_time_without_init_s,
                    csv_columns[5]: int(self.processed_frame),
                }
                pose_eval.append(slam_eval_dict)
            output_csv_path = os.path.join(self.run_path, "pose_eval.csv")
            try:
                with open(output_csv_path, "w") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in pose_eval:
                        writer.writerow(data)
            except IOError:
                print("I/O error")

            plot_timing_detail(
                time_table,
                os.path.join(self.run_path, "time_details.png"),
                self.config.pgo_on,
            )

            # if self.config.o3d_vis_on:  # x service issue for remote server
            output_traj_plot_path_2d = os.path.join(self.run_path, "traj_plot_2d.png")
            output_traj_plot_path_3d = os.path.join(self.run_path, "traj_plot_3d.png")
            # trajectory not aligned yet in the plot
            # require list of numpy arraies as the input

            gt_position_list = [self.gt_poses[i] for i in range(self.processed_frame)]
            odom_position_list = [self.odom_poses[i] for i in range(self.processed_frame)]

            if self.config.pgo_on:
                pgo_position_list = [self.pgo_poses[i] for i in range(self.processed_frame)]
                plot_trajectories(
                    output_traj_plot_path_2d,
                    pgo_position_list,
                    gt_position_list,
                    odom_position_list,
                    plot_3d=False,
                )
                plot_trajectories(
                    output_traj_plot_path_3d,
                    pgo_position_list,
                    gt_position_list,
                    odom_position_list,
                    plot_3d=True,
                )
            else:
                plot_trajectories(
                    output_traj_plot_path_2d,
                    odom_position_list,
                    gt_position_list,
                    plot_3d=False,
                )
                plot_trajectories(
                    output_traj_plot_path_3d,
                    odom_position_list,
                    gt_position_list,
                    plot_3d=True,
                )

        return pose_eval

    def write_merged_point_cloud(self, down_vox_m=None, 
                                use_gt_pose=False, 
                                out_file_name="merged_point_cloud",
                                frame_step = 1,
                                merged_downsample = False):

        print("Begin to replay the dataset ...")

        o3d_device = o3d.core.Device("CPU:0")
        o3d_dtype = o3d.core.float32
        map_out_o3d = o3d.t.geometry.PointCloud(o3d_device)
        map_points_np = np.empty((0, 3))
        map_intensity_np = np.empty(0)
        map_color_np = np.empty((0, 3))

        for frame_id in tqdm(
            range(0, self.total_pc_count, frame_step)
        ):  # frame id as the idx of the frame in the data folder without skipping
            if self.config.use_dataloader:
                self.read_frame_with_loader(frame_id)
            else:
                self.read_frame(frame_id)

            if self.config.kitti_correction_on:
                self.cur_point_cloud_torch = intrinsic_correct(
                    self.cur_point_cloud_torch, self.config.correction_deg
                )

            if self.config.deskew and frame_id < self.total_pc_count-1:
                if use_gt_pose and self.gt_pose_provided:
                    tran_in_frame = (
                        np.linalg.inv(self.gt_poses[frame_id + 1])
                        @ self.gt_poses[frame_id]
                    )
                else:
                    if self.config.track_on:
                        tran_in_frame = (
                            np.linalg.inv(self.odom_poses[frame_id + 1])
                            @ self.odom_poses[frame_id]
                        )
                    elif self.gt_pose_provided:
                        tran_in_frame = (
                            np.linalg.inv(self.gt_poses[frame_id + 1])
                            @ self.gt_poses[frame_id]
                        )
                self.cur_point_cloud_torch = deskewing(
                    self.cur_point_cloud_torch,
                    self.cur_point_ts_torch,
                    torch.tensor(
                        tran_in_frame, device=self.device, dtype=torch.float64
                    )
                )

            if down_vox_m is None:
                down_vox_m = self.config.vox_down_m
            idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:, :3], down_vox_m)

            frame_down_torch = self.cur_point_cloud_torch[idx]

            frame_down_torch, _ = crop_frame(
                frame_down_torch,
                None,
                self.config.min_z,
                self.config.max_z,
                self.config.min_range,
                self.config.max_range,
            )
            # get pose
            if use_gt_pose and self.gt_pose_provided:
                cur_pose_torch = torch.tensor(
                    self.gt_poses[frame_id], device=self.device, dtype=torch.float64
                )
            else:
                if self.config.pgo_on:
                    cur_pose_torch = torch.tensor(
                        self.pgo_poses[frame_id],
                        device=self.device,
                        dtype=torch.float64,
                    )
                elif self.config.track_on:
                    cur_pose_torch = torch.tensor(
                        self.odom_poses[frame_id],
                        device=self.device,
                        dtype=torch.float64,
                    )
                elif self.gt_pose_provided:
                    cur_pose_torch = torch.tensor(
                        self.gt_poses[frame_id], device=self.device, dtype=torch.float64
                    )
            frame_down_torch[:, :3] = transform_torch(
                frame_down_torch[:, :3], cur_pose_torch
            )

            frame_points_np = frame_down_torch[:, :3].detach().cpu().numpy()
            map_points_np = np.concatenate((map_points_np, frame_points_np), axis=0)
            if self.config.color_channel == 1:
                frame_intensity_np = frame_down_torch[:, 3].detach().cpu().numpy()
                map_intensity_np = np.concatenate(
                    (map_intensity_np, frame_intensity_np), axis=0
                )
            elif self.config.color_channel == 3:
                frame_color_np = frame_down_torch[:, 3:].detach().cpu().numpy()
                map_color_np = np.concatenate((map_color_np, frame_color_np), axis=0)

        print("Replay done")

        map_out_o3d.point["positions"] = o3d.core.Tensor(
            map_points_np, o3d_dtype, o3d_device
        )
        if self.config.color_channel == 1:
            map_out_o3d.point["intensity"] = o3d.core.Tensor(
                np.expand_dims(map_intensity_np, axis=1), o3d_dtype, o3d_device
            )
        elif self.config.color_channel == 3:
            map_out_o3d.point["colors"] = o3d.core.Tensor(
                map_color_np, o3d_dtype, o3d_device
            )

        if merged_downsample:
            map_out_o3d = map_out_o3d.voxel_down_sample(voxel_size=down_vox_m)

        if self.run_path is not None:
            print("Output merged point cloud map")
            o3d.t.io.write_point_cloud(
                os.path.join(self.run_path, "map", out_file_name+".ply"), map_out_o3d
            )


def read_point_cloud(
    filename: str, color_channel: int = 0, bin_channel_count: int = 4
) -> np.ndarray:

    if ".bin" in filename:
        data_loaded = np.fromfile(filename, dtype=np.float32)
        points = data_loaded.reshape((-1, bin_channel_count))
        ts = None

    elif ".ply" in filename:
        pc_load = o3d.t.io.read_point_cloud(filename)
        pc_load = {k: v.numpy() for k, v in pc_load.point.items()}

        keys = list(pc_load.keys())
        points = pc_load["positions"]

        if "t" in keys:
            ts = pc_load["t"] * 1e-8
        elif "timestamp" in keys:
            ts = pc_load["timestamp"]
        else:
            ts = None

        if "colors" in keys and color_channel == 3:
            colors = pc_load["colors"]
            points = np.hstack((points, colors))
        elif "intensity" in keys and color_channel == 1:
            intensity = pc_load["intensity"]
            points = np.hstack((points, intensity))
    elif ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points, dtype=np.float64)
        ts = None
    elif ".las" in filename:
        import laspy
        with laspy.open(filename) as fh:
            las = fh.read()
            x = las.points.X * las.header.scale[0] + las.header.offset[0]
            y = las.points.Y * las.header.scale[1] + las.header.offset[1]
            z = las.points.Z * las.header.scale[2] + las.header.offset[2]
            points = np.array([x, y, z], dtype=np.float64).T
            if color_channel == 1:
                intensity = np.array(las.points.intensity).reshape(-1, 1)
                # print(intensity)
                points = np.hstack((points, intensity))
            ts = None
    else:
        sys.exit(
            "The format of the imported point cloud is wrong"
        )
    return points, ts
def read_semantic_point_label(
    bin_filename: str, label_filename: str, color_on: bool = False
):

    if ".bin" in bin_filename:
        points = np.fromfile(bin_filename, dtype=np.float32).reshape(-1, 4)
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *bin)")

    if ".label" in label_filename:
        labels = np.fromfile(label_filename, dtype=np.uint32).reshape(-1)
    else:
        sys.exit(
            "The format of the imported point labels is wrong (support only *label)"
        )

    labels = labels & 0xFFFF
    labels_reduced = np.vectorize(sem_map_function)(labels).astype(
        np.int32
    )
    labels = np.array(labels, dtype=np.int32)
    return points, labels, labels_reduced


def read_kitti_format_calib(filename: str):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib


def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    poses = []
    with open(filename, 'r') as file:            
        for line in file:
            values = line.strip().split()
            if len(values) < 12:
                print('Not a kitti format pose file')
                return None
            values = [float(value) for value in values]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(pose)
    return poses

def read_tum_format_poses(filename: str):
    from pyquaternion import Quaternion
    poses = []
    timestamps = []
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        if any(char.isdigit() for char in first_line):
            file.seek(0)
        
        for line in file:
            values = line.strip().split()
            if len(values) != 8 and len(values) != 9: 
                print('Not a tum format pose file')
                return None, None
            idx_col =  len(values) - 8
            values = [float(value) for value in values]
            timestamps.append(values[idx_col])
            trans = np.array(values[1+idx_col:4+idx_col])
            quat = Quaternion(np.array([values[7+idx_col], values[4+idx_col], values[5+idx_col], values[6+idx_col]])) # w, i, j, k
            rot = quat.rotation_matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            poses.append(odom_tf)
    
    return poses, timestamps


def write_kitti_format_poses(filename: str, poses_np: np.ndarray, direct_use_filename = False):
    poses_out = poses_np[:, :3, :]
    poses_out_kitti = poses_out.reshape(poses_out.shape[0], -1)

    if direct_use_filename:
        fname = filename
    else:
        fname = f"{filename}_kitti.txt"
    
    np.savetxt(fname=fname, X=poses_out_kitti)

def write_tum_format_poses(filename: str, poses_np: np.ndarray, timestamps=None, frame_s = 0.1, 
                           with_header = False, direct_use_filename = False):
    from pyquaternion import Quaternion

    frame_count = poses_np.shape[0]
    tum_out = np.empty((frame_count,8))
    for i in range(frame_count):
        tx, ty, tz = poses_np[i, :3, -1].flatten()
        qw, qx, qy, qz = Quaternion(matrix=poses_np[i], atol=0.01).elements
        if timestamps is None:
            ts = i * frame_s
        else:
            ts = float(timestamps[i])
        tum_out[i] = np.array([ts, tx, ty, tz, qx, qy, qz, qw])

    if with_header:
        header = "timestamp tx ty tz qx qy qz qw"
    else:
        header = ''
        
    if direct_use_filename:
        fname = filename
    else:
        fname = f"{filename}_tum.txt"

    np.savetxt(fname=fname, X=tum_out, fmt="%.4f", header=header)

def apply_kitti_format_calib(poses_np: np.ndarray, calib_T_cl: np.ndarray):
    poses_calib_np = poses_np.copy()
    for i in range(poses_np.shape[0]):
        poses_calib_np[i, :, :] = calib_T_cl @ poses_np[i, :, :] @ inv(calib_T_cl)
    return poses_calib_np

# torch version
def crop_frame(
    points: torch.tensor,
    ts: torch.tensor,
    min_z_th=-3.0,
    max_z_th=100.0,
    min_range=2.75,
    max_range=100.0,
):
    dist = torch.norm(points[:, :3], dim=1)
    filtered_idx = (
        (dist > min_range)
        & (dist < max_range)
        & (points[:, 2] > min_z_th)
        & (points[:, 2] < max_z_th)
    )
    points = points[filtered_idx]
    if ts is not None:
        ts = ts[filtered_idx]
    return points, ts


def intrinsic_correct(points: torch.tensor, correct_deg=0.0):
    if correct_deg == 0.0:
        return points

    dist = torch.norm(points[:, :3], dim=1)
    kitti_var_vertical_ang = correct_deg / 180.0 * math.pi
    v_ang = torch.asin(points[:, 2] / dist)
    v_ang_c = v_ang + kitti_var_vertical_ang
    hor_scale = torch.cos(v_ang_c) / torch.cos(v_ang)
    points[:, 0] *= hor_scale
    points[:, 1] *= hor_scale
    points[:, 2] = dist * torch.sin(v_ang_c)

    return points
def filter_sem_kitti(
    points: torch.tensor,
    sem_labels_reduced: torch.tensor,
    sem_labels: torch.tensor,
    filter_outlier=True,
    filter_moving=False,
):

    if filter_outlier:
        inlier_mask = sem_labels > 1
    else:
        inlier_mask = sem_labels >= 0

    if filter_moving:
        static_mask = sem_labels < 100
        inlier_mask = inlier_mask & static_mask

    points = points[inlier_mask]
    sem_labels_reduced = sem_labels_reduced[inlier_mask]

    return points, sem_labels_reduced


def write_traj_as_o3d(poses_np, path):

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(poses_np[:, :3, 3])

    ts_np = np.linspace(0, 1, poses_np.shape[0])
    color_map = cm.get_cmap("jet")
    ts_color = color_map(ts_np)[:, :3].astype(np.float64)
    o3d_pcd.colors = o3d.utility.Vector3dVector(ts_color)

    if path is not None:
        o3d.io.write_point_cloud(path, o3d_pcd)

    return o3d_pcd