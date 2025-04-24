from rich import print
from utils.tools import transform_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model

    def forward(self, x1, x2):
        output1 = self.base_model(x1)
        output2 = self.base_model(x2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        return output1, output2, euclidean_distance


# Define the base model for the Siamese Network
base_model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)


class NeuralPointMapContextManager:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype
        self.silence = config.silence

        self.des_shape = config.context_shape
        self.num_candidates = config.context_num_candidates
        self.ringkey_dist_thre = (config.max_z - config.min_z) * 0.25

        self.sc_cosdist_threshold = config.context_cosdist_threshold
        if config.local_map_context:
            self.sc_cosdist_threshold += 0.08
            if config.loop_with_feature:
                self.sc_cosdist_threshold += 0.08
                self.ringkey_dist_thre = 0.25

        self.max_length = config.npmc_max_dist
        self.ENOUGH_LARGE = config.end_frame

        self.contexts = [None] * self.ENOUGH_LARGE
        self.ringkeys = [None] * self.ENOUGH_LARGE
        self.contexts_feature = [None] * self.ENOUGH_LARGE
        self.ringkeys_feature = [None] * self.ENOUGH_LARGE

        self.query_contexts = []
        self.tran_from_frame = []
        self.curr_node_idx = 0

        self.virtual_step_m = config.context_virtual_step_m
        self.virtual_side_count = config.context_virtual_side_count
        self.virtual_sdf_thre = 0.0

        # Initialize the Siamese Network
        self.siamese_network = SiameseNetwork(base_model).to(self.device)

    def add_node(self, frame_id, ptcloud, ptfeatures=None):
        sc, sc_feature = ptcloud2sc_torch(ptcloud, ptfeatures, self.des_shape, self.max_length)
        rk = sc2rk(sc)

        self.curr_node_idx = frame_id
        self.contexts[frame_id] = sc
        self.ringkeys[frame_id] = rk

        if sc_feature is not None:
            rk_feature = sc2rk(sc_feature)
            self.contexts_feature[frame_id] = sc_feature
            self.ringkeys_feature[frame_id] = rk_feature

        self.query_contexts = []
        self.tran_from_frame = []

    def set_virtual_node(self, ptcloud_global, frame_pose, last_frame_pose, ptfeatures=None):
        if last_frame_pose is not None:
            tran_direction = frame_pose[:3, 3] - last_frame_pose[:3, 3]
            tran_direction_norm = torch.norm(tran_direction)
            tran_direction_unit = tran_direction / tran_direction_norm
            lat_rot = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=self.device, dtype=self.dtype)
            lat_direction_unit = lat_rot @ tran_direction_unit.float()
        else:
            lat_direction_unit = torch.tensor([0, 1, 0], device=self.device, dtype=self.dtype)

        dx = torch.arange(-self.virtual_side_count, self.virtual_side_count + 1,
                          device=self.device) * self.virtual_step_m
        lat_tran = dx.view(-1, 1) @ lat_direction_unit.view(1, 3)

        virtual_positions = frame_pose[:3, 3].float() + lat_tran
        virtual_pose_count = virtual_positions.shape[0]

        if not self.silence:
            print("# Augmented virtual context: ", virtual_pose_count)

        for idx in range(virtual_pose_count):
            cur_lat_tran = lat_tran[idx]
            cur_tran_from_frame = torch.eye(4, device=self.device, dtype=torch.float64)
            cur_tran_from_frame[:3, 3] = cur_lat_tran

            cur_virtual_pose = frame_pose @ torch.linalg.inv(cur_tran_from_frame)

            if torch.norm(cur_lat_tran) == 0:
                if ptfeatures is None:
                    cur_sc = self.contexts[self.curr_node_idx]
                else:
                    cur_sc_feature = self.contexts_feature[self.curr_node_idx]
            else:
                ptcloud = transform_torch(ptcloud_global, torch.linalg.inv(cur_virtual_pose))
                cur_sc, cur_sc_feature = ptcloud2sc_torch(ptcloud, ptfeatures, self.des_shape, self.max_length)

            if ptfeatures is None:
                self.query_contexts.append(cur_sc)
            else:
                self.query_contexts.append(cur_sc_feature)
            self.tran_from_frame.append(cur_tran_from_frame)

    def detect_global_loop(self, cur_pgo_poses, dist_thre, loop_candidate_mask, neural_points):
        dist_to_past = np.linalg.norm(cur_pgo_poses[:, :3, 3] - cur_pgo_poses[self.curr_node_idx, :3, 3], axis=1)
        dist_search_mask = dist_to_past < dist_thre
        global_loop_candidate_idx = np.where(loop_candidate_mask & dist_search_mask)[0]
        if global_loop_candidate_idx.shape[0] > 0:
            context_pc = neural_points.local_neural_points.detach()
            cur_pose = torch.tensor(cur_pgo_poses[self.curr_node_idx], device=self.device, dtype=torch.float64)
            last_pose = torch.tensor(cur_pgo_poses[self.curr_node_idx - 1], device=self.device,
                                     dtype=torch.float64) if self.curr_node_idx > 0 else None
            neural_points_feature = neural_points.local_geo_features[
                                    :-1].detach() if self.config.loop_with_feature else None
            self.set_virtual_node(context_pc, cur_pose, last_pose, neural_points_feature)

        loop_id, loop_cos_dist, loop_transform = self.detect_loop(global_loop_candidate_idx,
                                                                  use_feature=self.config.loop_with_feature)
        local_map_context_loop = False
        if loop_id is not None:
            if self.config.local_map_context:
                loop_transform = loop_transform @ np.linalg.inv(cur_pgo_poses[self.curr_node_idx]) @ cur_pgo_poses[-1]
                local_map_context_loop = True
            if not self.silence:
                print("[bold red]Candidate global loop event detected: [/bold red]", self.curr_node_idx, "---", loop_id,
                      "(", loop_cos_dist, ")")
        return loop_id, loop_cos_dist, loop_transform, local_map_context_loop

    def detect_loop(self, candidate_idx, use_feature: bool = False):
        if candidate_idx.shape[0] == 0:
            return None, None, None

        if use_feature:
            ringkey_feature_history = torch.stack([self.ringkeys_feature[i] for i in candidate_idx]).to(self.device)
            history_count = ringkey_feature_history.shape[0]
        else:
            ringkey_history = torch.stack([self.ringkeys[i] for i in candidate_idx]).to(self.device)

        min_dist_ringkey = 1e5
        min_loop_idx = None
        min_query_idx = 0

        if len(self.query_contexts) == 0:
            self.tran_from_frame.append(torch.eye(4, device=self.device, dtype=torch.float64))
            if use_feature:
                self.query_contexts.append(self.contexts_feature[self.curr_node_idx])
            else:
                self.query_contexts.append(self.contexts[self.curr_node_idx])

        for query_idx in range(len(self.query_contexts)):
            if use_feature:
                query_context_feature = self.query_contexts[query_idx].view(1, -1)
                dist_to_history = []
                for history_feature in ringkey_feature_history:
                    _, _, dist = self.siamese_network(query_context_feature, history_feature.view(1, -1))
                    dist_to_history.append(dist.item())
                dist_to_history = torch.tensor(dist_to_history, device=self.device)
            else:
                query_context = self.query_contexts[query_idx]
                query_ringkey = sc2rk(query_context)
                diff_to_history = query_ringkey - ringkey_history
                dist_to_history = torch.norm(diff_to_history, p=1, dim=1)

            min_idx = torch.argmin(dist_to_history)
            cur_min_idx_in_candidates = candidate_idx[min_idx].item()
            cur_dist_ringkey = dist_to_history[min_idx].item()

            if cur_dist_ringkey < min_dist_ringkey:
                min_dist_ringkey = cur_dist_ringkey
                min_loop_idx = cur_min_idx_in_candidates
                min_query_idx = query_idx

        if not self.silence:
            print("min ringkey dist:", min_dist_ringkey)

        if min_dist_ringkey > self.ringkey_dist_thre:
            return None, None, None

        if use_feature:
            query_sc_feature = self.query_contexts[min_query_idx]
            candidate_sc_feature = self.contexts_feature[min_loop_idx]
            _, _, cosdist = self.siamese_network(query_sc_feature.view(1, -1), candidate_sc_feature.view(1, -1))
            cosdist = cosdist.item()
            yaw_diff = 0
        else:
            query_sc = self.query_contexts[min_query_idx]
            candidate_sc = self.contexts[min_loop_idx]
            cosdist, yaw_diff = distance_sc_torch(candidate_sc, query_sc)

        if not self.silence:
            print("min context cos dist:", cosdist)

        if cosdist < self.sc_cosdist_threshold:
            yawdiff_deg = yaw_diff * (360.0 / self.des_shape[1])

            if not self.silence:
                print("yaw diff deg:", yawdiff_deg)

            yawdiff_rad = math.radians(yawdiff_deg)
            cos_yaw = math.cos(yawdiff_rad)
            sin_yaw = math.sin(yawdiff_rad)
            transformation = np.eye(4)
            transformation[0, 0] = cos_yaw
            transformation[0, 1] = sin_yaw
            transformation[1, 0] = -sin_yaw
            transformation[1, 1] = cos_yaw

            transformation = transformation @ self.tran_from_frame[min_query_idx].detach().cpu().numpy()

            return min_loop_idx, cosdist, transformation
        else:
            return None, None, None

def detect_local_loop(
        pgo_poses,
        loop_candidate_mask,
        cur_drift,
        cur_frame_id,
        loop_reg_failed_count=0,
        dist_thre=1.0,
        drift_thre=3.0,
        silence=False,
):

    dist_to_past = np.linalg.norm(pgo_poses[:, :3, 3] - pgo_poses[-1, :3, 3], axis=1)
    min_dist = np.min(dist_to_past[loop_candidate_mask])
    min_index = np.where(dist_to_past == min_dist)[0]
    if (
            min_dist < dist_thre and cur_drift < drift_thre and loop_reg_failed_count < 3
    ):  # local loop
        loop_id, loop_dist = min_index[0], min_dist  # a candidate found
        loop_transform = np.linalg.inv(pgo_poses[loop_id]) @ pgo_poses[-1]
        if not silence:
            print(
                "[bold red]Candidate local loop event detected: [/bold red]",
                cur_frame_id,
                "---",
                loop_id,
                "(",
                loop_dist,
                ")",
            )
        return loop_id, loop_dist, loop_transform
    else:
        return None, None, None


def ptcloud2sc_torch(ptcloud, pt_feature, sc_shape, max_length):
    r = torch.norm(ptcloud, dim=1)
    kept_mask = r < max_length
    points = ptcloud[kept_mask]
    r = r[kept_mask]

    num_ring = sc_shape[0]
    num_sector = sc_shape[1]
    gap_ring = max_length / num_ring
    gap_sector = 360.0 / num_sector

    sc = torch.zeros(num_ring * num_sector, dtype=points.dtype, device=points.device)
    sc_feature = None

    if pt_feature is not None:
        pt_feature_kept = (pt_feature.clone())[kept_mask]
        sc_feature = torch.zeros(
            num_ring * num_sector,
            pt_feature.shape[1],
            dtype=points.dtype,
            device=points.device,
        )

    theta = torch.atan2(points[:, 1], points[:, 0])
    theta_degrees = theta * 180.0 / math.pi + 180.0
    idx_ring = torch.clamp((r // gap_ring).long(), 0, num_ring - 1)
    idx_sector = torch.clamp((theta_degrees // gap_sector).long(), 0, num_sector - 1)

    grid_indices = idx_ring * num_sector + idx_sector

    sc = sc.scatter_reduce_(
        dim=0, index=grid_indices, src=points[:, 2], reduce="amax", include_self=False
    )
    sc = sc.view(num_ring, num_sector)  # R, S

    if pt_feature is not None:
        grid_indices = grid_indices.view(-1, 1).repeat(1, pt_feature.shape[1])
        sc_feature = sc_feature.scatter_reduce_(
            dim=0,
            index=grid_indices,
            src=pt_feature_kept,
            reduce="mean",
            include_self=False,
        )
        sc_feature = sc_feature.view(
            num_ring, num_sector, pt_feature.shape[1]
        )
    return sc, sc_feature


def sc2rk(sc):
    return torch.mean(sc, dim=1)


def distance_sc_torch(sc1, sc2):
    num_sectors = sc1.shape[1]

    _one_step = 1
    sim_for_each_cols = torch.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = torch.roll(
            sc1, _one_step, 1
        )
        cossim = F.cosine_similarity(sc1, sc2, dim=0)
        sim_for_each_cols[i] = torch.mean(cossim)
    yaw_diff = torch.argmax(sim_for_each_cols) + 1
    sim = torch.max(sim_for_each_cols)
    dist = 1 - sim
    return dist.item(), yaw_diff.item()
