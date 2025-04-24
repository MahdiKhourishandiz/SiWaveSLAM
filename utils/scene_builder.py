import math
import sys
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import wandb
from rich import print
from tqdm import tqdm

from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.smart_cloud import NeuralPoints
from utils.config import Config
from utils.data_sampler import DataSampler
from utils.loss import color_diff_loss, sdf_bce_loss, sdf_diff_loss, sdf_zhong_loss
from utils.tools import (
    get_gradient,
    get_time,
    setup_optimizer,
    transform_batch_torch,
    transform_torch,
)


class Mapper:
    def __init__(
        self,
        config: Config,
        dataset: SLAMDataset,
        neural_points: NeuralPoints,
        geo_mlp: Decoder,
        sem_mlp: Decoder,
        color_mlp: Decoder,
    ):

        self.config = config
        self.silence = config.silence
        self.dataset = dataset
        self.neural_points = neural_points
        self.geo_mlp = geo_mlp
        self.sem_mlp = sem_mlp
        self.color_mlp = color_mlp
        self.device = config.device
        self.dtype = config.dtype
        self.used_poses = None
        self.require_gradient = False
        if (
            config.ekional_loss_on
            or config.proj_correction_on
            or config.consistency_loss_on
        ):
            self.require_gradient = True
        if (
            config.numerical_grad
            and not config.proj_correction_on
            and not config.consistency_loss_on
        ):
            self.require_gradient = False
        self.total_iter: int = 0
        self.sdf_scale = config.logistic_gaussian_ratio * config.sigma_sigmoid_m

        # initialize the data sampler
        self.sampler = DataSampler(config)
        self.ray_sample_count = (
            1 + config.surface_sample_n + config.free_behind_n + config.free_front_n
        )

        self.new_idx = None
        self.ba_done_flag = False
        self.adaptive_iter_offset = 0

        # data pool
        self.coord_pool = torch.empty(
            (0, 3), device=self.device, dtype=self.dtype
        )  # coordinate in each frame's coordinate frame
        self.global_coord_pool = torch.empty(
            (0, 3), device=self.device, dtype=self.dtype
        )  # coordinate in global frame
        self.sdf_label_pool = torch.empty((0), device=self.device, dtype=self.dtype)
        self.color_pool = torch.empty(
            (0, self.config.color_channel), device=self.device, dtype=self.dtype
        )
        self.sem_label_pool = torch.empty((0), device=self.device, dtype=torch.int)
        self.normal_label_pool = torch.empty(
            (0, 3), device=self.device, dtype=self.dtype
        )
        self.weight_pool = torch.empty((0), device=self.device, dtype=self.dtype)
        self.time_pool = torch.empty((0), device=self.device, dtype=torch.int)

    def dynamic_filter(points_torch, config, neural_points, geo_mlp, mask_mlp, type_2_on: bool = False):
        if type_2_on:
            points_torch.requires_grad_(True)
        geo_feature, _, weight_knn, _, certainty = neural_points.query_feature(
            points_torch, training_mode=False
        )

        sdf_pred = geo_mlp.sdf(geo_feature)
        if not config.weighted_first:
            sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)

        if type_2_on:
            sdf_grad = get_gradient(points_torch, sdf_pred).detach()
            grad_norm = sdf_grad.norm(dim=-1, keepdim=True).squeeze()

        alpha = config.dynamic_sdf_ratio_thre
        beta = config.dynamic_certainty_thre
        gamma = 0.3
        combined_features = torch.cat([geo_feature, sdf_pred.unsqueeze(-1), weight_knn, certainty.unsqueeze(-1)],
                                      dim=-1)

        static_mask_pred = mask_mlp(combined_features).squeeze(-1)
        loss_sdf = F.mse_loss(sdf_pred, torch.full_like(sdf_pred, alpha * config.voxel_size_m))
        loss_certainty = F.mse_loss(certainty, torch.full_like(certainty, beta))
        loss_grad_norm = F.mse_loss(grad_norm, torch.full_like(grad_norm, gamma)) if type_2_on else torch.tensor(0.0)

        total_loss = loss_sdf + loss_certainty + loss_grad_norm

        static_mask = (certainty < beta) | (sdf_pred < alpha * config.voxel_size_m)

        if type_2_on:
            static_mask_2 = (grad_norm > gamma) | (certainty < beta)
            static_mask = static_mask & static_mask_2

        return static_mask

    def determine_used_pose(self):

        cur_frame = self.dataset.processed_frame
        if self.config.pgo_on:
            self.used_poses = torch.tensor(
                self.dataset.pgo_poses[:cur_frame+1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.config.track_on:
            self.used_poses = torch.tensor(
                self.dataset.odom_poses[:cur_frame+1],
                device=self.device,
                dtype=torch.float64,
            )
        elif self.dataset.gt_pose_provided:  # for pure reconstruction with known pose
            self.used_poses = torch.tensor(
                self.dataset.gt_poses[:cur_frame+1],
                device=self.device,
                dtype=torch.float64
            )

    def process_frame(
        self,
        point_cloud_torch: torch.tensor,
        frame_label_torch: torch.tensor,
        cur_pose_torch: torch.tensor,
        frame_id: int,
        filter_dynamic: bool = False,
    ):

        frame_origin_torch = cur_pose_torch[:3, 3]
        frame_orientation_torch = cur_pose_torch[:3, :3]
        frame_point_torch = point_cloud_torch[:, :3]
        self.static_mask = torch.ones(
            frame_point_torch.shape[0], dtype=torch.bool, device=self.config.device
        )
        if filter_dynamic:
            self.neural_points.reset_local_map(frame_origin_torch, frame_orientation_torch, frame_id)
            frame_point_torch_global = transform_torch(
                frame_point_torch, cur_pose_torch
            )
            self.static_mask = self.dynamic_filter(frame_point_torch_global)
            dynamic_count = (self.static_mask == 0).sum().item()
            if not self.silence:
                print("# Dynamic points filtered: ", dynamic_count)
            frame_point_torch = frame_point_torch[self.static_mask]

        frame_color_torch = None
        if self.config.color_on:
            frame_color_torch = point_cloud_torch[:, 3:]
            if filter_dynamic:
                frame_color_torch = frame_color_torch[self.static_mask]

        if frame_label_torch is not None:
            if filter_dynamic:
                frame_label_torch = frame_label_torch[self.static_mask]
        frame_normal_torch = None

        self.dataset.static_mask = self.static_mask

        (
            coord,
            sdf_label,
            normal_label,
            sem_label,
            color_label,
            weight,
        ) = self.sampler.sample(
            frame_point_torch, frame_normal_torch, frame_label_torch, frame_color_torch
        )
        # coord is in sensor local frame

        time_repeat = torch.tensor(
            frame_id, dtype=torch.int, device=self.device
        ).repeat(coord.shape[0])

        self.cur_sample_count = sdf_label.shape[0]  # before filtering
        self.pool_sample_count = self.sdf_label_pool.shape[0]

        T2 = get_time()

        # update the neural point map
        if self.config.from_sample_points:
            if self.config.from_all_samples:
                update_points = coord
            else:
                update_points = coord[
                    torch.abs(sdf_label)
                    < self.config.surface_sample_range_m
                    * self.config.map_surface_ratio,
                    :,
                ]
                update_points = transform_torch(update_points, cur_pose_torch)
        else:
            update_points = transform_torch(frame_point_torch, cur_pose_torch)

        # prune map and recreate hash
        if self.config.prune_map_on and ((frame_id + 1) % self.config.prune_freq_frame == 0):
            if self.neural_points.prune_map(self.config.max_prune_certainty):
                self.neural_points.recreate_hash(None, None, True, True, frame_id)
        self.neural_points.update(
            update_points, frame_origin_torch, frame_orientation_torch, frame_id
        )
        # local map is also updated here

        if not self.silence:
            self.neural_points.print_memory()

        T3 = get_time()

        # concat with current observations
        self.coord_pool = torch.cat((self.coord_pool, coord), 0)
        self.weight_pool = torch.cat((self.weight_pool, weight), 0)
        self.sdf_label_pool = torch.cat((self.sdf_label_pool, sdf_label), 0)
        self.time_pool = torch.cat((self.time_pool, time_repeat), 0)

        if sem_label is not None:
            self.sem_label_pool = torch.cat((self.sem_label_pool, sem_label), 0)
        else:
            self.sem_label_pool = None
        if color_label is not None:
            self.color_pool = torch.cat((self.color_pool, color_label), 0)
        else:
            self.color_pool = None
        if normal_label is not None:
            self.normal_label_pool = torch.cat(
                (self.normal_label_pool, normal_label), 0
            )
        else:
            self.normal_label_pool = None

        self.determine_used_pose()

        if self.ba_done_flag:  # bundle adjustment is not done
            self.global_coord_pool = transform_batch_torch(
                self.coord_pool, self.used_poses[self.time_pool]
            )  # very slow here [if ba is not done, then you don't need to transform the whole data pool]
            self.ba_done_flag = False

        else:  # used when ba is not enabled
            global_coord = transform_torch(coord, cur_pose_torch)
            self.global_coord_pool = torch.cat(
                (self.global_coord_pool, global_coord), 0
            )

        T3_1 = get_time()

        if (frame_id + 1) % self.config.pool_filter_freq == 0:
            pool_relatve = self.global_coord_pool - frame_origin_torch
            # print(pool_relatve.shape)
            pool_relative_dist = torch.sum(pool_relatve**2, dim=-1)
            dist_mask = pool_relative_dist < self.config.window_radius**2

            filter_mask = dist_mask

            true_indices = torch.nonzero(filter_mask).squeeze()

            pool_sample_count = true_indices.shape[0]

            if pool_sample_count > self.config.pool_capacity:
                discard_count = pool_sample_count - self.config.pool_capacity
                discarded_index = torch.randint(
                    0, pool_sample_count, (discard_count,), device=self.device
                )
                filter_mask[
                    true_indices[discarded_index]
                ] = False

            self.coord_pool = self.coord_pool[filter_mask]
            self.global_coord_pool = self.global_coord_pool[
                filter_mask
            ]  # make global here
            self.sdf_label_pool = self.sdf_label_pool[filter_mask]
            self.weight_pool = self.weight_pool[filter_mask]
            self.time_pool = self.time_pool[filter_mask]

            if normal_label is not None:
                self.normal_label_pool = self.normal_label_pool[filter_mask]
            if sem_label is not None:
                self.sem_label_pool = self.sem_label_pool[filter_mask]
            if color_label is not None:
                self.color_pool = self.color_pool[filter_mask]

            cur_sample_filter_mask = filter_mask[
                -self.cur_sample_count :
            ]  # typically all true
            self.cur_sample_count = (
                cur_sample_filter_mask.sum().item()
            )  # number of current samples
            self.pool_sample_count = filter_mask.sum().item()
        else:
            self.cur_sample_count = coord.shape[0]
            self.pool_sample_count = self.coord_pool.shape[0]


        if (
            self.config.bs_new_sample > 0
        ):

            cur_sample_filtered = self.global_coord_pool[
                -self.cur_sample_count :
            ]  # newly added samples
            cur_sample_filtered_count = cur_sample_filtered.shape[0]
            bs = self.config.infer_bs
            iter_n = math.ceil(cur_sample_filtered_count / bs)
            cur_sample_certainty = torch.zeros(
                cur_sample_filtered_count, device=self.device
            )
            cur_label_filtered = self.sdf_label_pool[-self.cur_sample_count :]

            self.neural_points.set_search_neighborhood(
                num_nei_cells=1, search_alpha=0.0
            )
            for n in range(iter_n):
                head = n * bs
                tail = min((n + 1) * bs, cur_sample_filtered_count)
                batch_coord = cur_sample_filtered[head:tail, :]
                batch_certainty = self.neural_points.query_certainty(batch_coord)
                cur_sample_certainty[head:tail] = batch_certainty

            # dirty fix
            self.neural_points.set_search_neighborhood(
                num_nei_cells=self.config.num_nei_cells,
                search_alpha=self.config.search_alpha,
            )

            self.new_idx = torch.where(
                (cur_sample_certainty < self.config.new_certainty_thre)
                & (
                    torch.abs(cur_label_filtered)
                    < self.config.surface_sample_range_m * 3.0
                )
            )[0]

            self.new_idx += (
                self.pool_sample_count - self.cur_sample_count
            )
            new_sample_count = self.new_idx.shape[0]


            self.adaptive_iter_offset = 0
            new_obs_ratio = new_sample_count / self.cur_sample_count
            if self.config.adaptive_iters:
                if new_obs_ratio < self.config.new_sample_ratio_less:
                    # print('Train less:', new_obs_ratio)
                    self.adaptive_iter_offset = -5
                elif new_obs_ratio > self.config.new_sample_ratio_more:
                    # print('Train more:', new_obs_ratio)
                    self.adaptive_iter_offset = 5
                    if (
                        frame_id > self.config.freeze_after_frame
                        and new_obs_ratio > self.config.new_sample_ratio_restart
                    ):
                        self.adaptive_iter_offset = 10

    def get_batch(self, global_coord=False):

        if (
            self.config.bs_new_sample > 0
            and self.new_idx is not None
            and not self.dataset.lose_track
            and not self.dataset.stop_status
        ):
            # half, half for the history and current samples
            new_idx_count = self.new_idx.shape[0]
            if new_idx_count > 0:
                bs_new = min(new_idx_count, self.config.bs_new_sample)
                bs_history = self.config.bs - bs_new
                index_history = torch.randint(
                    0, self.pool_sample_count, (bs_history,), device=self.device
                )
                index_new_batch = torch.randint(
                    0, new_idx_count, (bs_new,), device=self.device
                )
                index_new = self.new_idx[index_new_batch]
                index = torch.cat((index_history, index_new), dim=0)
            else:  # uniformly sample the pool
                index = torch.randint(
                    0, self.pool_sample_count, (self.config.bs,), device=self.device
                )
        else:  # uniformly sample the pool
            index = torch.randint(
                0, self.pool_sample_count, (self.config.bs,), device=self.device
            )

        if global_coord:
            coord = self.global_coord_pool[index, :]
        else:
            coord = self.coord_pool[index, :]
        sdf_label = self.sdf_label_pool[index]
        ts = self.time_pool[index]  # frame number as the timestamp
        weight = self.weight_pool[index]

        if self.sem_label_pool is not None:
            sem_label = self.sem_label_pool[index]
        else:
            sem_label = None
        if self.color_pool is not None:
            color_label = self.color_pool[index]
        else:
            color_label = None
        if self.normal_label_pool is not None:
            normal_label = self.normal_label_pool[index, :]
        else:
            normal_label = None

        return coord, sdf_label, ts, normal_label, sem_label, color_label, weight

    # get a batch of training samples (only those measured end points) and labels for local bundle adjustment
    def get_ba_samples(self, subsample_count):

        surface_sample_idx = torch.where(self.sdf_label_pool == 0)[0]
        surface_sample_count = surface_sample_idx.shape[0]

        coord_pool_surface = self.coord_pool[surface_sample_idx]
        time_pool_surface = self.time_pool[surface_sample_idx]
        weight_pool_surface = self.weight_pool[surface_sample_idx]

        # uniformly sample the pool
        index = torch.randint(
            0, surface_sample_count, (subsample_count,), device=self.device
        )

        local_coord = coord_pool_surface[index, :]
        weight = weight_pool_surface[index]
        ts = time_pool_surface[index]  # frame number as the timestamp

        return local_coord, weight, ts

    # transform the data pool after pgo pose correction
    def transform_data_pool(self, pose_diff_torch: torch.tensor):
        # pose_diff_torch [N,4,4]
        self.global_coord_pool = transform_batch_torch(
            self.global_coord_pool, pose_diff_torch[self.time_pool]
        )

    # for visualization
    def get_data_pool_o3d(self, down_rate=1, only_cur_data=False):

        if only_cur_data:
            pool_coord_np = (
                self.global_coord_pool[-self.cur_sample_count :: 3]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )
        else:
            pool_coord_np = (
                self.global_coord_pool[::down_rate]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )

        data_pool_pc_o3d = o3d.geometry.PointCloud()
        data_pool_pc_o3d.points = o3d.utility.Vector3dVector(pool_coord_np)

        if self.sdf_label_pool is None:
            return data_pool_pc_o3d

        if only_cur_data:
            pool_label_np = (
                self.sdf_label_pool[-self.cur_sample_count :: 3]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )
        else:
            pool_label_np = (
                self.sdf_label_pool[::down_rate]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )

        min_sdf = self.config.free_sample_end_dist_m * -2.0
        max_sdf = -min_sdf
        pool_label_np = np.clip(
            (pool_label_np - min_sdf) / (max_sdf - min_sdf), 0.0, 1.0
        )

        color_map = cm.get_cmap("seismic")
        colors = color_map(pool_label_np)[:, :3].astype(np.float64)

        data_pool_pc_o3d.colors = o3d.utility.Vector3dVector(colors)

        return data_pool_pc_o3d

    def free_pool(self):
        self.coord_pool = None
        self.weight_pool = None
        self.sdf_label_pool = None
        self.time_pool = None
        self.sem_label_pool = None
        self.color_pool = None
        self.normal_label_pool = None

    # PIN map online training (mapping) given the fixed pose
    # the main training function
    def mapping(self, iter_count):

        iter_count = max(1, iter_count + self.adaptive_iter_offset)

        neural_point_feat = list(self.neural_points.parameters())
        geo_mlp_param = list(self.geo_mlp.parameters())
        if self.config.semantic_on:
            sem_mlp_param = list(self.sem_mlp.parameters())
        else:
            sem_mlp_param = None
        if self.config.color_on:
            color_mlp_param = list(self.color_mlp.parameters())
        else:
            color_mlp_param = None

        opt = setup_optimizer(
            self.config,
            neural_point_feat,
            geo_mlp_param,
            sem_mlp_param,
            color_mlp_param,
        )

        for iter in tqdm(range(iter_count), disable=self.silence):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)

            T00 = get_time()
            # we do not use the ray rendering loss here for the incremental mapping
            coord, sdf_label, ts, _, sem_label, color_label, weight = self.get_batch(
                global_coord=not self.ba_done_flag
            )  # coord here is in global frame if no ba pose update

            T01 = get_time()

            poses = self.used_poses[ts]
            origins = poses[:, :3, 3]

            if self.ba_done_flag:
                coord = transform_batch_torch(
                    coord, poses
                )  # transformed to global frame

            if self.require_gradient:
                coord.requires_grad_(True)

            (
                geo_feature,
                color_feature,
                weight_knn,
                _,
                certainty,
            ) = self.neural_points.query_feature(
                coord, ts, query_color_feature=self.config.color_on
            )

            T02 = get_time()
            # predict the scaled sdf with the feature

            sdf_pred = self.geo_mlp.sdf(
                geo_feature
            )  # predict the scaled sdf with the feature # [N, K, 1]
            if not self.config.weighted_first:
                sdf_pred = torch.sum(sdf_pred * weight_knn, dim=1).squeeze(1)  # N

            if self.config.semantic_on:
                sem_pred = self.sem_mlp.sem_label_prob(geo_feature)
                if not self.config.weighted_first:
                    sem_pred = torch.sum(sem_pred * weight_knn, dim=1)  # N, S
            if self.config.color_on:
                color_pred = self.color_mlp.regress_color(color_feature)  # [N, K, C]
                if not self.config.weighted_first:
                    color_pred = torch.sum(color_pred * weight_knn, dim=1)  # N, C

            surface_mask = (
                torch.abs(sdf_label) < self.config.surface_sample_range_m
            )  # weight > 0

            if self.require_gradient:
                g = get_gradient(coord, sdf_pred)  # to unit m
            elif (
                self.config.numerical_grad
            ):  # do not use this for the tracking, still analytical grad for tracking
                g = self.get_numerical_gradient(
                    coord[:: self.config.gradient_decimation],
                    sdf_pred[:: self.config.gradient_decimation],
                    self.config.voxel_size_m * self.config.num_grad_step_ratio,
                )

            T03 = get_time()

            if self.config.proj_correction_on:  # [not used]
                cos = torch.abs(F.cosine_similarity(g, coord - origins))
                sdf_label = sdf_label * cos

            if self.config.consistency_loss_on:  # [not used]
                near_index = torch.randint(
                    0,
                    coord.shape[0],
                    (min(self.config.consistency_count, coord.shape[0]),),
                    device=self.device,
                )
                random_shift = (
                    torch.rand_like(coord) * 2 * self.config.consistency_range
                    - self.config.consistency_range
                )  # 10 cm
                coord_near = coord + random_shift
                coord_near = coord_near[
                    near_index, :
                ]  # only use a part of these coord to speed up
                coord_near.requires_grad_(True)
                (
                    geo_feature_near,
                    _,
                    weight_knn,
                    _,
                    _,
                ) = self.neural_points.query_feature(coord_near)
                pred_near = self.geo_mlp.sdf(geo_feature_near)
                if not self.config.weighted_first:
                    pred_near = torch.sum(pred_near * weight_knn, dim=1).squeeze(1)  # N
                g_near = get_gradient(coord_near, pred_near)

            cur_loss = 0.0
            weight = torch.abs(
                weight
            ).detach()  # weight's sign indicate the sample is around the surface or in the free space
            if self.config.main_loss_type == "bce":  # [used]
                sdf_loss = sdf_bce_loss(
                    sdf_pred,
                    sdf_label,
                    self.sdf_scale,
                    weight,
                    self.config.loss_weight_on,
                )
            elif self.config.main_loss_type == "zhong":  # [not used]
                sdf_loss = sdf_zhong_loss(
                    sdf_pred, sdf_label, None, weight, self.config.loss_weight_on
                )
            elif self.config.main_loss_type == "sdf_l1":  # [not used]
                sdf_loss = sdf_diff_loss(sdf_pred, sdf_label, weight, l2_loss=False)
            elif self.config.main_loss_type == "sdf_l2":  # [not used]
                sdf_loss = sdf_diff_loss(sdf_pred, sdf_label, weight, l2_loss=True)
            else:
                sys.exit("Please choose a valid loss type")
            cur_loss += sdf_loss

            # optional consistency regularization loss
            consistency_loss = 0.0
            if self.config.consistency_loss_on:  # [not used]
                consistency_loss = (
                    1.0 - F.cosine_similarity(g[near_index, :], g_near)
                ).mean()
                cur_loss += self.config.weight_c * consistency_loss

            # ekional loss
            eikonal_loss = 0.0
            if (
                self.config.ekional_loss_on and self.config.weight_e > 0
            ):  # MSE with regards to 1
                surface_mask_decimated = surface_mask[
                    :: self.config.gradient_decimation
                ]
                # weight_used = (weight.clone())[::self.config.gradient_decimation] # point-wise weight not used
                if self.config.ekional_add_to == "freespace":
                    g_used = g[~surface_mask_decimated]
                    # weight_used = weight_used[~surface_mask_decimated]
                elif self.config.ekional_add_to == "surface":
                    g_used = g[surface_mask_decimated]
                    # weight_used = weight_used[surface_mask_decimated]
                else:  # "all"  # both the surface and the freespace, used here # [used]
                    g_used = g
                eikonal_loss = (
                    (g_used.norm(2, dim=-1) - 1.0) ** 2
                ).mean()  # both the surface and the freespace
                cur_loss += self.config.weight_e * eikonal_loss

            # optional semantic loss
            sem_loss = 0.0
            if self.config.semantic_on and self.config.weight_s > 0:
                loss_nll = torch.nn.NLLLoss(reduction="mean")
                if self.config.freespace_label_on:
                    label_mask = (
                        sem_label >= 0
                    )  # only use the points with labels (-1, unlabled would not be used)
                else:
                    label_mask = (
                        sem_label > 0
                    )  # only use the points with labels (even those with free space labels would not be used)
                sem_pred = sem_pred[label_mask]
                sem_label = sem_label[label_mask].long()
                sem_loss = loss_nll(
                    sem_pred[:: self.config.sem_label_decimation, :],
                    sem_label[:: self.config.sem_label_decimation],
                )
                cur_loss += self.config.weight_s * sem_loss

            # optional color (intensity) loss
            color_loss = 0.0
            if self.config.color_on and self.config.weight_i > 0:
                color_loss = color_diff_loss(
                    color_pred[surface_mask],
                    color_label[surface_mask],
                    weight[surface_mask],
                    self.config.loss_weight_on,
                    l2_loss=False,
                )
                cur_loss += self.config.weight_i * color_loss

            T04 = get_time()

            opt.zero_grad(set_to_none=True)
            cur_loss.backward(retain_graph=False)
            opt.step()

            self.total_iter += 1
            if self.config.wandb_vis_on:
                wandb_log_content = {
                    "iter": self.total_iter,
                    "loss/total_loss": cur_loss,
                    "loss/sdf_loss": sdf_loss,
                    "loss/eikonal_loss": eikonal_loss,
                    "loss/consistency_loss": consistency_loss,
                    "loss/sem_loss": sem_loss,
                    "loss/color_loss": color_loss,
                }
                wandb.log(wandb_log_content)

        # update the global map
        self.neural_points.assign_local_to_global()

    def sdf(self, x, get_std=False):
        geo_feature, _, weight_knn, _, _ = self.neural_points.query_feature(x)
        sdf_pred = self.geo_mlp.sdf(
            geo_feature
        )  # predict the scaled sdf with the feature # [N, K, 1]
        sdf_std = None
        if not self.config.weighted_first:
            sdf_pred_mean = torch.sum(sdf_pred * weight_knn, dim=1)  # N
            if get_std:
                sdf_var = torch.sum(
                    (weight_knn * (sdf_pred - sdf_pred_mean.unsqueeze(-1)) ** 2), dim=1
                )
                sdf_std = torch.sqrt(sdf_var).squeeze(1)
            sdf_pred = sdf_pred_mean.squeeze(1)
        return sdf_pred, sdf_std

    def get_numerical_gradient(self, x, sdf_x=None, eps=0.02, two_side=True):

        N = x.shape[0]

        eps_x = torch.tensor([eps, 0.0, 0.0], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0.0, eps, 0.0], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0.0, 0.0, eps], dtype=x.dtype, device=x.device)  # [3]

        if two_side:
            x_pos = x + eps_x
            x_neg = x - eps_x
            y_pos = x + eps_y
            y_neg = x - eps_y
            z_pos = x + eps_z
            z_neg = x - eps_z

            x_posneg = torch.concat((x_pos, x_neg, y_pos, y_neg, z_pos, z_neg), dim=0)
            sdf_x_posneg = self.sdf(x_posneg)[0].unsqueeze(-1)

            sdf_x_pos = sdf_x_posneg[:N]
            sdf_x_neg = sdf_x_posneg[N : 2 * N]
            sdf_y_pos = sdf_x_posneg[2 * N : 3 * N]
            sdf_y_neg = sdf_x_posneg[3 * N : 4 * N]
            sdf_z_pos = sdf_x_posneg[4 * N : 5 * N]
            sdf_z_neg = sdf_x_posneg[5 * N :]

            gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
            gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
            gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)

        else:
            x_pos = x + eps_x
            y_pos = x + eps_y
            z_pos = x + eps_z

            x_all = torch.concat((x_pos, y_pos, z_pos), dim=0)
            sdf_x_all = self.sdf(x_all)[0].unsqueeze(-1)

            sdf_x = sdf_x.unsqueeze(-1)

            sdf_x_pos = sdf_x_all[:N]
            sdf_y_pos = sdf_x_all[N : 2 * N]
            sdf_z_pos = sdf_x_all[2 * N :]

            gradient_x = (sdf_x_pos - sdf_x) / eps
            gradient_y = (sdf_y_pos - sdf_x) / eps
            gradient_z = (sdf_z_pos - sdf_x) / eps

        gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=1)  # [...,3]

        return gradient
