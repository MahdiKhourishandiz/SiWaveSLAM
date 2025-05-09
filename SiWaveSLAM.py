import os
import sys
import numpy as np
import open3d as o3d
import torch
import wandb
from rich import print
from tqdm import tqdm

from dataset.dataset_indexing import set_dataset_path
from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.smart_cloud import NeuralPoints
from utils.config import Config
from utils.loop_detector import (
    NeuralPointMapContextManager,
    detect_local_loop,
)
from utils.scene_builder import Mapper
from utils.mesher import Mesher
from utils.pgo import PoseGraphManager
from utils.tools import (
    freeze_decoders,
    get_time,
    load_decoder,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    transform_torch,
)
from utils.pose_estimator import Tracker
from utils.visualizer import MapVisualizer


def setup_slam_system(
    config_path=r'config\lidar_slam\kitti_dataset.yaml',
    dataset_name=None,
    sequence_name=None,
    seed=42,
    input_path=None,
    output_path=None,
    frame_range=None,
    use_dataloader=False,
    visualize=True,
    cpu_only=False,
    log_on=False,
    wandb_on=False,
    save_map=False,
    save_mesh=False,
    save_merged_pc=False
):
    config = Config()
    config.load(config_path)

    if dataset_name and sequence_name:
        set_dataset_path(config, dataset_name, sequence_name)
    config.seed = seed
    config.use_dataloader = use_dataloader
    config.wandb_vis_on = wandb_on
    config.o3d_vis_on = True
    config.save_map = save_map
    config.save_mesh = save_mesh
    config.save_merged_pc = save_merged_pc
    if frame_range:
        config.begin_frame, config.end_frame, config.step_frame = frame_range
    if cpu_only:
        config.device = 'cuda'
    if input_path:
        config.pc_path = input_path
    if output_path:
        config.output_root = output_path

    argv = ['SiWaveSLAM.py', config_path]
    run_path = setup_experiment(config, argv)
    return config, run_path


def initialize_slam_components(config):
    # non-blocking visualizer
    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config)
    else:
        o3d_vis = None

    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level,
                      config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level,
                        config.color_channel) if config.color_on else None

    # initialize the feature octree
    neural_points = NeuralPoints(config)

    # Load the decoder model (not used in main flow)
    if config.load_model:
        load_decoder(config, geo_mlp, sem_mlp, color_mlp)

    # dataset
    dataset = SLAMDataset(config)

    # odometry tracker
    tracker = Tracker(config, neural_points, geo_mlp, sem_mlp, color_mlp)

    # mapper
    mapper = Mapper(config, dataset, neural_points, geo_mlp, sem_mlp, color_mlp)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, geo_mlp, sem_mlp, color_mlp)
    cur_mesh = None

    # pose graph manager (for back-end optimization) initialization
    pgm = PoseGraphManager(config)
    init_pose = dataset.gt_poses[0] if dataset.gt_pose_provided else np.eye(4)
    pgm.add_pose_prior(0, init_pose, fixed=True)

    # loop closure detector
    lcd_npmc = NeuralPointMapContextManager(config)

    return (o3d_vis, geo_mlp, sem_mlp, color_mlp, neural_points, dataset, tracker,
            mapper, mesher, cur_mesh, pgm, lcd_npmc)


def process_slam_frames(config, run_path, components):
    (o3d_vis, geo_mlp, sem_mlp, color_mlp, neural_points, dataset, tracker,
     mapper, mesher, cur_mesh, pgm, lcd_npmc) = components

    last_frame = dataset.total_pc_count - 1
    loop_reg_failed_count = 0

    if config.save_merged_pc and dataset.gt_pose_provided:
        dataset.write_merged_point_cloud(use_gt_pose=True, out_file_name='merged_gt_pc',
                                         frame_step=5, merged_downsample=True)

    for frame_id in tqdm(range(dataset.total_pc_count)):
        T0 = get_time()

        if config.use_dataloader:
            dataset.read_frame_with_loader(frame_id)
        else:
            dataset.read_frame(frame_id)

        T1 = get_time()

        valid_frame = dataset.preprocess_frame()
        if not valid_frame:
            dataset.processed_frame += 1
            continue

        T2 = get_time()

        # II. Odometry
        if frame_id > 0:
            if config.track_on:
                tracking_result = tracker.tracking(dataset.cur_source_points, dataset.cur_pose_guess_torch,
                                                  dataset.cur_source_colors, dataset.cur_source_normals,
                                                  vis_result=config.o3d_vis_on and not config.o3d_vis_raw)
                cur_pose_torch, cur_odom_cov, weight_pc_o3d, valid_flag = tracking_result
                dataset.lose_track = not valid_flag
                dataset.update_odom_pose(cur_pose_torch)

                if not valid_flag and config.o3d_vis_on and o3d_vis.debug_mode > 0:
                    o3d_vis.stop()
            else:
                if dataset.gt_pose_provided:
                    dataset.update_odom_pose(dataset.cur_pose_guess_torch)
                else:
                    sys.exit("You are using the mapping mode, but no pose is provided.")

        travel_dist = dataset.travel_dist[:frame_id + 1]
        neural_points.travel_dist = torch.tensor(travel_dist, device=config.device, dtype=config.dtype)

        T3 = get_time()

        # III. Loop detection and pgo
        if config.pgo_on:
            if config.global_loop_on:
                if config.local_map_context and frame_id >= config.local_map_context_latency:
                    local_map_frame_id = frame_id - config.local_map_context_latency
                    local_map_pose = torch.tensor(dataset.pgo_poses[local_map_frame_id], device=config.device, dtype=torch.float64)
                    if config.local_map_context_latency > 0:
                        neural_points.reset_local_map(local_map_pose[:3, 3], None, local_map_frame_id, False,
                                                      config.loop_local_map_time_window)
                    context_pc_local = transform_torch(neural_points.local_neural_points.detach(), torch.linalg.inv(local_map_pose))
                    neural_points_feature = neural_points.local_geo_features[:-1].detach() if config.loop_with_feature else None
                    lcd_npmc.add_node(local_map_frame_id, context_pc_local, neural_points_feature)
                else:
                    lcd_npmc.add_node(frame_id, dataset.cur_point_cloud_torch)
            pgm.add_frame_node(frame_id, dataset.pgo_poses[frame_id])
            pgm.init_poses = dataset.pgo_poses[:frame_id + 1]
            if frame_id > 0:
                cur_edge_cov = cur_odom_cov if config.use_reg_cov_mat else None
                pgm.add_odometry_factor(frame_id, frame_id - 1, dataset.last_odom_tran, cov=cur_edge_cov)
                pgm.estimate_drift(travel_dist, frame_id, correct_ratio=0.01)
                if config.pgo_with_pose_prior:
                    pgm.add_pose_prior(frame_id, dataset.pgo_poses[frame_id])
            local_map_context_loop = False
            if frame_id - pgm.last_loop_idx > config.pgo_freq and not dataset.stop_status:
                loop_candidate_mask = ((travel_dist[-1] - travel_dist) > (config.min_loop_travel_dist_ratio * config.local_map_radius))
                loop_id = None
                if np.any(loop_candidate_mask):
                    loop_id, loop_dist, loop_transform = detect_local_loop(dataset.pgo_poses[:frame_id + 1], loop_candidate_mask,
                                                                           pgm.drift_radius, frame_id, loop_reg_failed_count,
                                                                           config.local_loop_dist_thre, config.local_loop_dist_thre * 3.0)
                    if loop_id is None and config.global_loop_on:
                        loop_id, loop_cos_dist, loop_transform, local_map_context_loop = lcd_npmc.detect_global_loop(
                            dataset.pgo_poses[:frame_id + 1], pgm.drift_radius * config.loop_dist_drift_ratio_thre,
                            loop_candidate_mask, neural_points)
                if loop_id is not None:
                    if config.loop_z_check_on and abs(loop_transform[2, 3]) > config.voxel_size_m * 4.0:
                        loop_id = None
                if loop_id is not None:
                    pose_init_torch = torch.tensor((dataset.pgo_poses[loop_id] @ loop_transform), device=config.device, dtype=torch.float64)
                    neural_points.recreate_hash(pose_init_torch[:3, 3], None, True, True, loop_id)
                    loop_reg_source_point = dataset.cur_source_points.clone()
                    pose_refine_torch, loop_cov_mat, weight_pcd, reg_valid_flag = tracker.tracking(
                        loop_reg_source_point, pose_init_torch, loop_reg=True, vis_result=config.o3d_vis_on)
                    if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                        points_torch_init = transform_torch(loop_reg_source_point, pose_init_torch)
                        points_o3d_init = o3d.geometry.PointCloud()
                        points_o3d_init.points = o3d.utility.Vector3dVector(points_torch_init.detach().cpu().numpy().astype(np.float64))
                        loop_neural_pcd = neural_points.get_neural_points_o3d(query_global=False, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1)
                        o3d_vis.update(points_o3d_init, neural_points=loop_neural_pcd, pause_now=True)
                        o3d_vis.update(weight_pcd, neural_points=loop_neural_pcd, pause_now=True)
                    if reg_valid_flag:
                        pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                        loop_transform = np.linalg.inv(dataset.pgo_poses[loop_id]) @ pose_refine_np
                        cur_edge_cov = loop_cov_mat if config.use_reg_cov_mat else None
                        reg_valid_flag = pgm.add_loop_factor(frame_id, loop_id, loop_transform, cov=cur_edge_cov)
                    if reg_valid_flag:
                        pgm.optimize_pose_graph()
                        cur_loop_vis_id = frame_id - config.local_map_context_latency if local_map_context_loop else frame_id
                        pgm.loop_edges_vis.append(np.array([loop_id, cur_loop_vis_id], dtype=np.uint32))
                        pgm.loop_edges.append(np.array([loop_id, frame_id], dtype=np.uint32))
                        pgm.loop_trans.append(loop_transform)
                        pose_diff_torch = torch.tensor(pgm.get_pose_diff(), device=config.device, dtype=config.dtype)
                        dataset.cur_pose_torch = torch.tensor(pgm.cur_pose, device=config.device, dtype=config.dtype)
                        neural_points.adjust_map(pose_diff_torch)
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3, 3], None, (not config.pgo_merge_map),
                                                    config.rehash_with_time, frame_id)
                        mapper.transform_data_pool(pose_diff_torch)
                        dataset.update_poses_after_pgo(pgm.cur_pose, pgm.pgo_poses)
                        pgm.last_loop_idx = frame_id
                        pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
                        loop_reg_failed_count = 0
                        if config.o3d_vis_on:
                            o3d_vis.before_pgo = False
                    else:
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3, 3], None, True, True, frame_id)
                        loop_reg_failed_count += 1
                        if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                            o3d_vis.stop()

        T4 = get_time()

        # IV: Mapping and bundle adjustment
        if frame_id < 5 or (not dataset.lose_track and not dataset.stop_status):
            mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch,
                                 dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0))
        else:
            mapper.determine_used_pose()
            neural_points.reset_local_map(dataset.cur_pose_torch[:3, 3], None, frame_id)

        T5 = get_time()

        cur_iter_num = config.iters * config.init_iter_ratio if frame_id == 0 else config.iters
        if dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num - 10)
        if frame_id == config.freeze_after_frame:
            freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)

        if config.track_on and config.ba_freq_frame > 0 and (frame_id + 1) % config.ba_freq_frame == 0:
            mapper.lie_group_optimization(config.ba_iters, config.ba_frame)

        if frame_id % config.mapping_freq_frame == 0:
            mapper.mapping(cur_iter_num)

        T6 = get_time()

        # regular saving logs
        if config.log_freq_frame > 0 and (frame_id + 1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        # V: Mesh reconstruction and visualization
        cur_mesh = None
        if config.o3d_vis_on:
            o3d_vis.cur_frame_id = frame_id
            dataset.update_o3d_map()
            if config.track_on and frame_id > 0 and (not o3d_vis.vis_pc_color) and (weight_pc_o3d is not None):
                dataset.cur_frame_o3d = weight_pc_o3d

            T7 = get_time()

            if frame_id == last_frame:
                o3d_vis.vis_global = True
                o3d_vis.ego_view = False
                mapper.free_pool()

            neural_pcd = None
            if o3d_vis.render_neural_points or (frame_id == last_frame):
                neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global,
                                                                 color_mode=o3d_vis.neural_points_vis_mode,
                                                                 random_down_ratio=1)

            if config.mesh_freq_frame > 0:
                if o3d_vis.render_mesh and (frame_id == 0 or frame_id == last_frame or
                                            (frame_id + 1) % config.mesh_freq_frame == 0 or pgm.last_loop_idx == frame_id):

                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23)
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()

                    mesh_path = None
                    if frame_id == last_frame and config.save_mesh:
                        mc_cm_str = str(round(o3d_vis.mc_res_m * 1e2))
                        mesh_path = os.path.join(run_path, "mesh", 'mesh_frame_' + str(frame_id) + "_" + mc_cm_str + "cm.ply")

                    if not o3d_vis.vis_global:
                        chunks_aabb = split_chunks(global_neural_pcd_down, dataset.cur_bbx, o3d_vis.mc_res_m * 100)
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True,
                                                                      config.semantic_on, config.color_on,
                                                                      filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m * 300)
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False,
                                                                      config.semantic_on, config.color_on,
                                                                      filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)

            cur_sdf_slice = None
            if config.sdfslice_freq_frame > 0:
                if o3d_vis.render_sdf and (frame_id == 0 or frame_id == last_frame or
                                           (frame_id + 1) % config.sdfslice_freq_frame == 0):
                    slice_res_m = config.voxel_size_m * 0.2
                    sdf_bound = config.surface_sample_range_m * 4.0
                    query_sdf_locally = True
                    if o3d_vis.vis_global:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[2, 3] +
                                                                             o3d_vis.sdf_slice_height, slice_res_m, False,
                                                                             -sdf_bound, sdf_bound)
                    else:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[2, 3] +
                                                                             o3d_vis.sdf_slice_height, slice_res_m,
                                                                             query_sdf_locally, -sdf_bound, sdf_bound)
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0, 3],
                                                                             slice_res_m, query_sdf_locally, -sdf_bound,
                                                                             sdf_bound)
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h

            pool_pcd = mapper.get_data_pool_o3d(down_rate=17, only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
            loop_edges = pgm.loop_edges_vis if config.pgo_on else None
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)

            T8 = get_time()

        cur_frame_process_time = np.array([T2 - T1, T3 - T2, T5 - T4, T6 - T5, T4 - T3])
        dataset.time_table.append(cur_frame_process_time)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': frame_id, 'timing(s)/preprocess': T2 - T1, 'timing(s)/tracking': T3 - T2,
                                 'timing(s)/pgo': T4 - T3, 'timing(s)/mapping': T6 - T4}
            wandb.log(wandb_log_content)

        dataset.processed_frame += 1

    return (o3d_vis, geo_mlp, sem_mlp, color_mlp, neural_points, dataset, tracker,
            mapper, mesher, cur_mesh, pgm, lcd_npmc, neural_pcd, cur_sdf_slice, loop_edges, pool_pcd, weight_pc_o3d)


def finalize_and_save_results(config, run_path, components):
    (o3d_vis, geo_mlp, sem_mlp, color_mlp, neural_points, dataset, tracker,
     mapper, mesher, cur_mesh, pgm, lcd_npmc, neural_pcd, cur_sdf_slice, loop_edges, pool_pcd, weight_pc_o3d) = components

    # VI. Save results
    if config.track_on:
        pose_eval_results = dataset.write_results()
    if config.pgo_on and pgm.pgo_count > 0:
        print("# Loop corrected: ", pgm.pgo_count)
        pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
        pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
        if config.o3d_vis_on:
            pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)

    neural_points.recreate_hash(None, None, False, False)
    neural_points.prune_map(config.max_prune_certainty, 0)
    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=0)
    if config.save_map:
        o3d.io.write_point_cloud(os.path.join(run_path, "map", "neural_points.ply"), neural_pcd)
    if config.save_mesh and cur_mesh is None:
        output_mc_res_m = config.mc_res_m * 0.6
        chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), output_mc_res_m * 300)
        mc_cm_str = str(round(output_mc_res_m * 1e2))
        mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, output_mc_res_m, mesh_path, False,
                                                      config.semantic_on, config.color_on, filter_isolated_mesh=True,
                                                      mesh_min_nn=config.mesh_min_nn)
    neural_points.clear_temp()
    if config.save_map:
        save_implicit_map(run_path, neural_points, geo_mlp, color_mlp, sem_mlp)
    if config.save_merged_pc:
        dataset.write_merged_point_cloud()

    if config.o3d_vis_on:
        while True:
            o3d_vis.ego_view = False
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)

    return pose_eval_results


def execute_slam_pipeline(
    config_path=r'config\lidar_slam\kitti_dataset.yaml',
    dataset_name=None,
    sequence_name=None,
    seed=42,
    input_path=None,
    output_path=None,
    frame_range=None,
    use_dataloader=False,
    visualize=True,
    cpu_only=False,
    log_on=False,
    wandb_on=False,
    save_map=False,
    save_mesh=False,
    save_merged_pc=False
):
    config, run_path = setup_slam_system(config_path, dataset_name, sequence_name, seed, input_path, output_path,
                                         frame_range, use_dataloader, visualize, cpu_only, log_on, wandb_on,
                                         save_map, save_mesh, save_merged_pc)
    components = initialize_slam_components(config)
    updated_components = process_slam_frames(config, run_path, components)
    result = finalize_and_save_results(config, run_path, updated_components)
    return result


if __name__ == "__main__":
    execute_slam_pipeline()