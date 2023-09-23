from pytorch3d.ops import sample_farthest_points
import hydra
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from equivariant_pose_graph.models.transformer_flow import ResidualFlow, ResidualFlow_V1, \
    ResidualFlow_V2, ResidualFlow_V3, ResidualFlow_V4, ResidualFlow_Correspondence,\
    ResidualFlow_Identity, ResidualFlow_PE, ResidualFlow_DiffEmb
from equivariant_pose_graph.dataset.point_cloud_data_module import MultiviewDataModule
from equivariant_pose_graph.training.flow_equivariance_training_module import EquivarianceTrainingModule
import os
import os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import shutil
import pybullet as p

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from ndf_robot.utils.sim_utils import get_clouds, get_object_clouds
from airobot.utils.common import euler2quat

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import np2img
from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import path_util
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)
from equivariant_pose_graph.training.flow_equivariance_training_module_nocentering_eval_init import EquivarianceTestingModule
from equivariant_pose_graph.models.transformer_flow import ResidualFlow, ResidualFlow_V1, \
    ResidualFlow_V2, ResidualFlow_V3, ResidualFlow_V4, ResidualFlow_Correspondence,\
    ResidualFlow_Identity, ResidualFlow_PE, ResidualFlow_DiffEmb, ResidualFlow_DiffEmbTransformer
# posegraph imports
# from ndf_robot.utils.equivariant_pose_graph_utils import get_model
import sys
sys.path.insert(1, '/home/exx/Documents/equivariant_pose_graph/python')


def load_data(num_points, clouds, classes, action_class, anchor_class):

    points_raw_np = clouds
    classes_raw_np = classes

    points_action_np = points_raw_np[classes_raw_np == action_class].copy()
    points_action_mean_np = points_action_np.mean(axis=0)
    points_action_np = points_action_np - points_action_mean_np

    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
    points_anchor_np = points_anchor_np - points_action_mean_np
    points_anchor_mean_np = points_anchor_np.mean(axis=0)

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

    points_action, points_anchor = subsample(
        num_points, points_action, points_anchor)

    return points_action.cuda(), points_anchor.cuda()


def subsample(num_points, points_action, points_anchor):
    if(points_action.shape[1] > num_points):
        points_action, _ = sample_farthest_points(points_action,
                                                  K=num_points, random_start_point=True)
    elif(points_action.shape[1] < num_points):
        raise NotImplementedError(
            f'Action point cloud is smaller than cloud size ({points_action.shape[1]} < {num_points})')

    if(points_anchor.shape[1] > num_points):
        points_anchor, _ = sample_farthest_points(points_anchor,
                                                  K=num_points, random_start_point=True)
    elif(points_anchor.shape[1] < num_points):
        raise NotImplementedError(
            f'Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {num_points})')

    return points_action, points_anchor


@hydra.main(config_path="/home/exx/Documents/equivariant_pose_graph/configs/exx/", config_name="exx_dcpflow_residual0_test_ndf")
def main(hydra_cfg):
    obj_class = hydra_cfg.object_class
    shapenet_obj_dir = osp.join(
        path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')

    demo_load_dir = osp.join(path_util.get_ndf_data(),
                             'demos', obj_class, hydra_cfg.demo_exp)

    expstr = 'exp--' + str(hydra_cfg.exp)
    modelstr = 'model--' + str(hydra_cfg.model_path)
    seedstr = 'seed--' + str(hydra_cfg.seed)
    full_experiment_name = '_'.join([expstr, modelstr, seedstr])

    eval_save_dir = osp.join(path_util.get_ndf_eval_data(
    ), hydra_cfg.eval_data_dir, full_experiment_name)
    util.safe_makedirs(eval_save_dir)

    vnn_model_path = osp.join(
        path_util.get_ndf_model_weights(), hydra_cfg.model_path + '.pth')

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path
    )

    print("loading done!")
    if hydra_cfg.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    robot = Robot('franka', pb_cfg={'gui': hydra_cfg.pybullet_viz}, arm_cfg={
                  'self_collision': False, 'seed': hydra_cfg.seed})
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(hydra_cfg.seed)
    np.random.seed(hydra_cfg.seed)
    random.seed(hydra_cfg.seed)

    txt_file_name = "{}.txt".format(hydra_cfg.eval_data_dir)
    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_ndf_config(),
                            'eval_cfgs', hydra_cfg.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' %
                 config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(
        path_util.get_ndf_config(), hydra_cfg.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict['shapenet_obj_dir']
    obj_class = global_dict['object_class']
    eval_save_dir = global_dict['eval_save_dir']

    eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
    eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
    eval_place_imgs_dir = osp.join(eval_save_dir, 'place_imgs')
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)
    util.safe_makedirs(eval_place_imgs_dir)

    test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(
    ), '%s_test_object_split.txt' % obj_class), dtype=str).tolist()
    if obj_class == 'mug':
        avoid_shapenet_ids = bad_shapenet_mug_ids_list + cfg.MUG.AVOID_SHAPENET_IDS
    elif obj_class == 'bowl':
        avoid_shapenet_ids = bad_shapenet_bowls_ids_list + cfg.BOWL.AVOID_SHAPENET_IDS
    elif obj_class == 'bottle':
        avoid_shapenet_ids = bad_shapenet_bottles_ids_list + cfg.BOTTLE.AVOID_SHAPENET_IDS
    else:
        test_shapenet_ids = []

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10
    p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
    p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)

    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    preplace_horizontal_tf_list = cfg.PREPLACE_HORIZONTAL_OFFSET_TF
    preplace_horizontal_tf = util.list2pose_stamped(
        cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
    preplace_offset_tf = util.list2pose_stamped(cfg.PREPLACE_OFFSET_TF)

    if hydra_cfg.dgcnn:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='dgcnn',
            return_features=True,
            sigmoid=True,
            acts=hydra_cfg.acts).cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='pointnet',
            return_features=True,
            sigmoid=True).cuda()

    if not hydra_cfg.random:
        checkpoint_path = global_dict['vnn_checkpoint_path']
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        pass

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        load_shelf = True
    else:
        load_shelf = False

    # get filenames of all the demo files
    demo_filenames = os.listdir(global_dict['demo_load_dir'])
    assert len(
        demo_filenames), 'No demonstrations found in path: %s!' % global_dict['demo_load_dir']

    # strip the filenames to properly pair up each demo file
    grasp_demo_filenames_orig = [osp.join(global_dict['demo_load_dir'], fn)
                                 for fn in demo_filenames if 'grasp_demo' in fn]  # use the grasp names as a reference

    place_demo_filenames = []
    grasp_demo_filenames = []
    for i, fname in enumerate(grasp_demo_filenames_orig):
        shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
        place_fname = osp.join(
            '/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)

        if osp.exists(place_fname):
            grasp_demo_filenames.append(fname)
            place_demo_filenames.append(place_fname)
        else:
            log_warn(
                'Could not find corresponding placement demo: %s, skipping ' % place_fname)

    success_list = []
    place_success_list = []
    place_success_teleport_list = []
    grasp_success_list = []

    place_fail_list = []
    place_fail_teleport_list = []
    grasp_fail_list = []

    demo_shapenet_ids = []

    # get info from all demonstrations
    demo_target_info_list = []
    demo_rack_target_info_list = []

    if hydra_cfg.n_demos > 0:
        gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
        gp_fns = random.sample(gp_fns, hydra_cfg.n_demos)
        grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
        grasp_demo_filenames, place_demo_filenames = list(
            grasp_demo_filenames), list(place_demo_filenames)
        log_warn('USING ONLY %d DEMONSTRATIONS' % len(grasp_demo_filenames))
        print(grasp_demo_filenames, place_demo_filenames)
    else:
        log_warn('USING ALL %d DEMONSTRATIONS' % len(grasp_demo_filenames))

    grasp_demo_filenames = grasp_demo_filenames[:hydra_cfg.num_demo]
    place_demo_filenames = place_demo_filenames[:hydra_cfg.num_demo]

    max_bb_volume = 0
    place_xq_demo_idx = 0
    grasp_data_list = []
    place_data_list = []
    demo_rel_mat_list = []

    # load all the demo data and look at objects to help decide on query points
    for i, fname in enumerate(grasp_demo_filenames):
        print('Loading demo from fname: %s' % fname)
        grasp_demo_fn = grasp_demo_filenames[i]
        place_demo_fn = place_demo_filenames[i]
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
        place_data = np.load(place_demo_fn, allow_pickle=True)

        grasp_data_list.append(grasp_data)
        place_data_list.append(place_data)

        start_ee_pose = grasp_data['ee_pose_world'].tolist()
        end_ee_pose = place_data['ee_pose_world'].tolist()
        place_rel_mat = util.get_transform(
            pose_frame_target=util.list2pose_stamped(end_ee_pose),
            pose_frame_source=util.list2pose_stamped(start_ee_pose)
        )
        place_rel_mat = util.matrix_from_pose(place_rel_mat)
        demo_rel_mat_list.append(place_rel_mat)

        if i == 0:
            optimizer_gripper_pts, rack_optimizer_gripper_pts, shelf_optimizer_gripper_pts = process_xq_data(
                grasp_data, place_data, shelf=load_shelf)
            optimizer_gripper_pts_rs, rack_optimizer_gripper_pts_rs, shelf_optimizer_gripper_pts_rs = process_xq_rs_data(
                grasp_data, place_data, shelf=load_shelf)

            if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
                print('Using shelf points')
                place_optimizer_pts = shelf_optimizer_gripper_pts
                place_optimizer_pts_rs = shelf_optimizer_gripper_pts_rs
            else:
                print('Using rack points')
                place_optimizer_pts = rack_optimizer_gripper_pts
                place_optimizer_pts_rs = rack_optimizer_gripper_pts_rs

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            target_info, rack_target_info, shapenet_id = process_demo_data_shelf(
                grasp_data, place_data, cfg=None)
        else:
            target_info, rack_target_info, shapenet_id = process_demo_data_rack(
                grasp_data, place_data, cfg=None)

        if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
            rack_target_info['demo_query_pts'] = place_optimizer_pts
        demo_target_info_list.append(target_info)
        demo_rack_target_info_list.append(rack_target_info)
        demo_shapenet_ids.append(shapenet_id)

    place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_optimizer_pts,
        query_pts_real_shape=place_optimizer_pts_rs,
        opt_iterations=hydra_cfg.opt_iterations)

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=optimizer_gripper_pts,
        query_pts_real_shape=optimizer_gripper_pts_rs,
        opt_iterations=hydra_cfg.opt_iterations)
    grasp_optimizer.set_demo_info(demo_target_info_list)
    place_optimizer.set_demo_info(demo_rack_target_info_list)

    # get objects that we can use for testing
    test_object_ids = []
    train_object_ids = []
    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(
        shapenet_obj_dir)] if obj_class == 'mug' else os.listdir(shapenet_obj_dir)
    for s_id in shapenet_id_list:
        valid = s_id not in demo_shapenet_ids and s_id not in avoid_shapenet_ids
        train_valid = s_id in demo_shapenet_ids and s_id not in avoid_shapenet_ids
        if hydra_cfg.only_test_ids:
            valid = valid and (s_id in test_shapenet_ids)

        if valid:
            test_object_ids.append(s_id)
        if train_valid:
            train_object_ids.append(s_id)
    test_object_ids = sorted(test_object_ids)
    train_object_ids = test_object_ids[:len(test_object_ids)//2]
    test_object_ids = test_object_ids[len(test_object_ids)//2:]

    # f = open(txt_file_name, "a")
    # f.write("len(test_object_ids):{}\n".format(len(test_object_ids)))
    # f.close()
    # reset
    robot.arm.reset(force_reset=True)
    robot.cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z],
        dist=0.9,
        yaw=45,
        pitch=-25,
        roll=0)

    cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
    cam_info = {}
    cam_info['pose_world'] = []
    for cam in cams.cams:
        cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

    # put table at right spot
    table_ori = euler2quat([0, 0, np.pi / 2])

    # this is the URDF that was used in the demos -- make sure we load an identical one
    tmp_urdf_fname = osp.join(
        path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
    open(tmp_urdf_fname, 'w').write(grasp_data['table_urdf'].item())
    table_id = robot.pb_client.load_urdf(tmp_urdf_fname,
                                         cfg.TABLE_POS,
                                         table_ori,
                                         scaling=cfg.TABLE_SCALING)

    if obj_class == 'mug':
        rack_link_id = 0
        shelf_link_id = 1
    elif obj_class in ['bowl', 'bottle']:
        rack_link_id = None
        shelf_link_id = 0

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        placement_link_id = shelf_link_id
    else:
        placement_link_id = rack_link_id

    def hide_link(obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    test_or_train = hydra_cfg.test_or_train
    if test_or_train == 'train':
        object_ids = train_object_ids
    elif test_or_train == 'test':
        object_ids = test_object_ids

    viz_data_list = []
    pl.seed_everything(hydra_cfg.seed)

    if hydra_cfg.flow_compute_type == 0:
        if hydra_cfg.diff_emb:
            if hydra_cfg.diff_transformer:
                network = ResidualFlow_DiffEmbTransformer(
                    emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component, center_feature=hydra_cfg.center_feature,
                    inital_sampling_ratio=hydra_cfg.inital_sampling_ratio)
            else:
                network = ResidualFlow_DiffEmb(
                    emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component,
                    center_feature=hydra_cfg.center_feature, inital_sampling_ratio=hydra_cfg.inital_sampling_ratio)
        else:
            network = ResidualFlow(
                emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component, center_feature=hydra_cfg.center_feature)
    elif hydra_cfg.flow_compute_type == 1:
        network = ResidualFlow_V1(
            emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component)
    elif hydra_cfg.flow_compute_type == 2:
        network = ResidualFlow_V2(
            emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component)
    elif hydra_cfg.flow_compute_type == 3:
        network = ResidualFlow_V3(
            emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component)
    elif hydra_cfg.flow_compute_type == 4:
        network = ResidualFlow_V4(
            emb_nn=hydra_cfg.emb_nn, return_flow_component=hydra_cfg.return_flow_component)
    elif hydra_cfg.flow_compute_type == 5:
        network = ResidualFlow_Correspondence(emb_nn=hydra_cfg.emb_nn)
    elif hydra_cfg.flow_compute_type == 6:
        network = ResidualFlow_Identity(emb_nn=hydra_cfg.emb_nn)
    elif hydra_cfg.flow_compute_type == 'pe':
        network = ResidualFlow_PE(emb_nn=hydra_cfg.emb_nn)
    model = EquivarianceTestingModule(
        network,
        lr=hydra_cfg.lr,
        image_log_period=hydra_cfg.image_logging_period,
        weight_normalize=hydra_cfg.weight_normalize,
        loop=hydra_cfg.loop
    )

    model.cuda()

    if(hydra_cfg.checkpoint_file is not None):
        model.load_state_dict(torch.load(
            hydra_cfg.checkpoint_file)['state_dict'])
        log_info("Model Loaded from " + str(hydra_cfg.checkpoint_file))

    for iteration in range(hydra_cfg.start_iteration, hydra_cfg.num_iterations):

        torch.autograd.set_detect_anomaly(True)
        torch.cuda.empty_cache()
        # network = ResidualFlow_DiffEmb()

        # model = EquivarianceTrainingModule(
        #     network)
        # model.cuda()
        # checkpoint_file = "/home/exx/media/DataDrive/singularity_chuerp/equiv_pgraph_logs/train_test_mr_dcpflow_residual0_attn_trans0.1_rot10_meancenter_overfit/equiv_dcpflow/version_1/saved_checkpts/epoch=69-step=8750.ckpt"
        # checkpoint_file = hydra_cfg.checkpoint_file
        # model.load_state_dict(torch.load(checkpoint_file)['state_dict'])
        # log_info("Model Loaded from " + str(checkpoint_file))
        ##
        place_model = model
        log_info("type(place_model)")
        log_info(type(place_model))

        f = open(txt_file_name, "a")
        f.write("-----------------------{}-----------------------\n".format(iteration))
        f.close()
        # load a test object
        # obj_shapenet_id = random.sample(object_ids, 1)[0]
        # obj_shapenet_id = '34ae0b61b0d8aaf2d7b20fded0142d7a'
        obj_shapenet_id = hydra_cfg.obj_shapenet_id
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        log_info(id_str)

        f = open(txt_file_name, "a")
        f.write(id_str)
        f.write("\n")
        f.close()

        if obj_class in ['bottle', 'jar', 'bowl', 'mug']:
            upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(
            shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

        scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
        scale_default = cfg.MESH_SCALE_DEFAULT
        if hydra_cfg.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high -
                                                scale_low) + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3

        if hydra_cfg.any_pose:
            if obj_class in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                table_z]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(
                pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(util.list2pose_stamped(
                pose), util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(
                pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
        else:
            pos = [np.random.random() * (x_high - x_low) + x_low,
                   np.random.random() * (y_high - y_low) + y_low, table_z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(
                pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(
                pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(
                pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
            # log_info("pos")
            # log_info(pos)
            # log_info("ori")
            # log_info(ori)
        pose_w_yam_T = util.matrix_from_pose(pose_w_yaw)
        log_info("pose_w_yam_T")
        log_info(pose_w_yam_T)

        # convert mesh with vhacd
        if not osp.exists(obj_obj_file_dec):
            p.vhacd(
                obj_obj_file,
                obj_obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )

        robot.arm.go_home(ignore_physics=True)
        robot.arm.move_ee_xyz([0, 0, 0.2])

        if hydra_cfg.any_pose:
            robot.pb_client.set_step_sim(True)
        if obj_class in ['bowl']:
            robot.pb_client.set_step_sim(True)

        obj_id = robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=pos,
            base_ori=ori)
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)

        if obj_class == 'bowl':
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id,
                                    linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id,
                                    linkIndexA=-1, linkIndexB=shelf_link_id, enableCollision=False)
            robot.pb_client.set_step_sim(False)

        o_cid = None
        if hydra_cfg.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        cloud_points, cloud_colors, cloud_classes = get_object_clouds(cams)

        data_dir = hydra_cfg.data_dir
        save_dir = os.path.join(
            '/home/exx/Documents/ndf_robot/src/ndf_robot', data_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        mug_id = 0
        rack_id = 1
        gripper_id = 2

        mug_points = cloud_points[cloud_classes == mug_id]
        rack_points = cloud_points[cloud_classes == rack_id]
        gripper_points = cloud_points[cloud_classes == gripper_id]

        np.save('mug_pcd_obs.npy', mug_points)
        np.save('rack_pcd_obs.npy', rack_points)
        np.save('gripper_pcd_obs.npy', gripper_points)
        hide_link(table_id, rack_link_id)

        # get object point cloud
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []
        rack_pcd_pts = []

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(
            list(obj_pose_world[0]) + list(obj_pose_world[1]))

        for i, cam in enumerate(cams.cams):
            import matplotlib.pyplot as plt
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(
                get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(
                in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            table_inds = np.where(flat_seg == table_id)
            seg_depth = flat_depth[obj_inds[0]]

            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))

            table_pts = pts_raw[table_inds[0],
                                :][::int(table_inds[0].shape[0]/500)]
            table_pcd_pts.append(table_pts)

            if rack_link_id is not None:
                rack_val = table_id + ((rack_link_id+1) << 24)
                rack_inds = np.where(flat_seg == rack_val)
                if rack_inds[0].shape[0] > 0:
                    rack_pts = pts_raw[rack_inds[0], :]
                    rack_pcd_pts.append(rack_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        target_obj_pcd_obs = np.concatenate(
            obj_pcd_pts, axis=0)  # object shape point cloud
        print(len(table_pcd_pts))
        for i in range(len(table_pcd_pts)):
            print(table_pcd_pts[i].shape)
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(
            target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        if obj_class == 'mug':
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        if obj_class == 'bowl':
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id,
                                        linkIndexA=i, linkIndexB=rack_link_id, enableCollision=False)
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id,
                                        linkIndexA=i, linkIndexB=shelf_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id,
                                    linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id,
                                    linkIndexA=-1, linkIndexB=shelf_link_id, enableCollision=False)

        # optimize grasp pose
        pre_grasp_ee_pose_mats, best_idx = grasp_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True)
        pre_grasp_ee_pose = util.pose_stamped2list(
            util.pose_from_matrix(pre_grasp_ee_pose_mats[best_idx]))

        ########################### grasp post-process #############################
        new_grasp_pt = post_process_grasp_point(pre_grasp_ee_pose, target_obj_pcd_obs, thin_feature=(
            not hydra_cfg.non_thin_feature), grasp_viz=hydra_cfg.grasp_viz, grasp_dist_thresh=hydra_cfg.grasp_dist_thresh)
        pre_grasp_ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # optimize placement pose
        rack_pose_mats, best_rack_idx = place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False)
        # # reset object to placement pose to detect placement success
        # safeCollisionFilterPair(
        #     obj_id, table_id, -1, -1, enableCollision=False)
        # safeCollisionFilterPair(obj_id, table_id, -1,
        #                         placement_link_id, enableCollision=False)
        # robot.pb_client.set_step_sim(True)
        # safeRemoveConstraint(o_cid)
        # time.sleep(1.0)

        # # TODO: debug replace rack_relative_pose with model predicted
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, '%d_init.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        points_action, points_anchor = load_data(
            num_points=1024, clouds=obj_points, classes=obj_classes, action_class=0, anchor_class=1)

        ans = place_model.get_transform(
            points_action, points_anchor)  # 1, 4, 4
        pred_T_action_init = ans["pred_T_action"]
        pred_T_action_transformed = pred_T_action_init.transform_points(
            points_action)
        # for i in range(20):
        #     pred_T_action, pred_T_anchor, pred_T  = place_model(pred_T_action_transformed, points_anchor)
        #     pred_T_action_transformed = pred_T_action.transform_points(pred_T_action_transformed)

        pred_T_action_mat = pred_T_action_init.get_matrix()[
            0].T.detach().cpu().numpy()
        rack_relative_pose = util.pose_stamped2list(
            util.pose_from_matrix(pred_T_action_mat))

        # log_info("pred_T_action_mat")
        # log_info(pred_T_action_mat)
        # pred_T_action = torch.transpose(pred_T_action_mat,-1,-2).detach().cpu().numpy()
        # pred_T_action = pred_T_action.detach().cpu().numpy()

        # log_info("pred_T_action.shape")
        # log_info(pred_T_action.shape)
        # print(pred_T_action[0])

        # rack_relative_pose = util.pose_stamped2list(util.pose_from_matrix_debug(pred_T_action[0]))
        # rack_relative_pose = util.pose_stamped2list(util.pose_from_matrix(rack_pose_mats[best_rack_idx]))
        # del place_model
        # del model
        # torch.cuda.empty_cache()

        np.savez(f'{save_dir}/{iteration}_init_all_points.npz',
                 clouds=cloud_points, colors=cloud_colors, classes=cloud_classes, shapenet_id=obj_shapenet_id)

        np.savez(f'{save_dir}/{iteration}_init_obj_points.npz',
                 clouds=obj_points, colors=obj_colors, classes=obj_classes, shapenet_id=obj_shapenet_id, pred_T_action_mat=pred_T_action_mat)
        log_info("Saved point cloud data to:")
        log_info(f'{save_dir}/{iteration}_init_obj_points.npz')

        # T_z = np.eye(4)
        # T_z[2, 3] -= 0.12
        # pose_shift_z = util.pose_from_matrix(T_z)

        # T_pred = pred_T_action_init.get_matrix().detach().cpu().numpy()[0].T

        # rack_relative_pose = util.pose_stamped2list(
        #     util.transform_pose(pose_shift_z, util.pose_from_matrix(T_pred)))

        ee_end_pose = util.transform_pose(pose_source=util.list2pose_stamped(
            pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(rack_relative_pose))
        pre_ee_end_pose2 = util.transform_pose(
            pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
        pre_ee_end_pose1 = util.transform_pose(
            pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf)

        ee_end_pose_list = util.pose_stamped2list(ee_end_pose)
        pre_ee_end_pose1_list = util.pose_stamped2list(pre_ee_end_pose1)
        pre_ee_end_pose2_list = util.pose_stamped2list(pre_ee_end_pose2)

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(
            list(obj_pose_world[0]) + list(obj_pose_world[1]))
        obj_start_pose = obj_pose_world

        obj_end_pose = util.transform_pose(
            pose_source=obj_start_pose, pose_transform=util.list2pose_stamped(rack_relative_pose))
        obj_end_pose_list = util.pose_stamped2list(obj_end_pose)

        # reset object to placement pose to detect placement success
        safeCollisionFilterPair(
            obj_id, table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, table_id, -1,
                                placement_link_id, enableCollision=False)
        robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, '%d_rightbefore.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)

        robot.pb_client.reset_body(
            obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, '%d_rightafter.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)

        robot.pb_client.set_step_sim(True)

        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, '%d_rightafter_step.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)

        cloud_points, cloud_colors, cloud_classes = get_clouds(
            cams)
        obj_points, obj_colors, obj_classes = get_object_clouds(
            cams)

        robot.pb_client.set_step_sim(True)
        # teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        # cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        # obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        time.sleep(1.0)
        # teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        # cloud_points, cloud_colors, cloud_classes = get_clouds(cams)
        # obj_points, obj_colors, obj_classes = get_object_clouds(cams)

        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(
            eval_teleport_imgs_dir, '%d.png' % iteration)
        log_info("eval_teleport_imgs_dir")
        log_info(eval_teleport_imgs_dir)

        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        safeCollisionFilterPair(obj_id, table_id, -1,
                                placement_link_id, enableCollision=True)

        robot.pb_client.set_step_sim(False)
        time.sleep(1.0)

        np.savez(f'{save_dir}/{iteration}_teleport_all_points.npz',
                 clouds=cloud_points, colors=cloud_colors, classes=cloud_classes, shapenet_id=obj_shapenet_id)

        np.savez(f'{save_dir}/{iteration}_teleport_obj_points.npz',
                 clouds=obj_points, colors=obj_colors, classes=obj_classes, shapenet_id=obj_shapenet_id, pred_T_action_transformed=pred_T_action_transformed.detach().cpu().numpy())
        log_info("teleport_img_fname")
        log_info(teleport_img_fname)

        obj_surf_contacts = p.getContactPoints(
            obj_id, table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        place_success_teleport_list.append(place_success_teleport)
        if not place_success_teleport:
            place_fail_teleport_list.append(iteration)

        time.sleep(1.0)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        robot.pb_client.reset_body(obj_id, pos, ori)

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):

            # reset everything
            robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(
                obj_id, table_id, -1, -1, enableCollision=True)
            if hydra_cfg.any_pose:
                robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            if hydra_cfg.any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                robot.pb_client.set_step_sim(False)
            robot.arm.go_home(ignore_physics=True)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i,
                                        linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i,
                                        linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
            robot.arm.eetool.open()

            if jnt_pos is None or grasp_jnt_pos is None:
                jnt_pos = ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = ik_helper.get_feasible_ik(pre_grasp_ee_pose)

                if jnt_pos is None or grasp_jnt_pos is None:
                    jnt_pos = ik_helper.get_ik(pre_pre_grasp_ee_pose)
                    grasp_jnt_pos = ik_helper.get_ik(pre_grasp_ee_pose)

                    if jnt_pos is None or grasp_jnt_pos is None:
                        jnt_pos = robot.arm.compute_ik(
                            pre_pre_grasp_ee_pose[:3], pre_pre_grasp_ee_pose[3:])
                        # this is the pose that's at the grasp, where we just need to close the fingers
                        grasp_jnt_pos = robot.arm.compute_ik(
                            pre_grasp_ee_pose[:3], pre_grasp_ee_pose[3:])

            # both ik returns a grapable joint plan for pre_pre_grasp and pre_grasp
            if grasp_jnt_pos is not None and jnt_pos is not None:
                if g_idx == 0:
                    robot.pb_client.set_step_sim(True)
                    robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    robot.arm.eetool.close(ignore_physics=True)
                    time.sleep(0.2)
                    # grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                    # grasp_img_fname = osp.join(eval_grasp_imgs_dir, '%d.png' % iteration)
                    # np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    continue

                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    plan1 = ik_helper.plan_joint_motion(
                        robot.arm.get_jpos(), jnt_pos, file_name=txt_file_name)
                    plan2 = ik_helper.plan_joint_motion(
                        jnt_pos, grasp_jnt_pos, file_name=txt_file_name)

                    # f = open(txt_file_name, "a")
                    # if plan1 == None:
                    #     f.write("plan1:{}\n".format(plan1))
                    # else:
                    #     f.write("plan1: not NONE \n")
                    # if plan2 == None:
                    #     f.write("plan2:{} \n".format(plan2))
                    # else:
                    #     f.write("plan2: not NONE \n")
                    # f.close()
                    if plan1 is not None and plan2 is not None:
                        grasp_plan = plan1 + plan2

                        robot.arm.eetool.open()
                        for jnt in plan1:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.025)
                        robot.arm.set_jpos(plan1[-1], wait=True)
                        for jnt in plan2:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.04)
                        robot.arm.set_jpos(grasp_plan[-1], wait=True)

                        # get pose that's straight up
                        offset_pose = util.transform_pose(
                            pose_source=util.list2pose_stamped(
                                np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()),
                            pose_transform=util.list2pose_stamped(
                                [0, 0, 0.15, 0, 0, 0, 1])
                        )
                        offset_pose_list = util.pose_stamped2list(offset_pose)
                        offset_jnts = ik_helper.get_feasible_ik(
                            offset_pose_list)

                        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
                        grasped_img_fname = osp.join(
                            eval_grasp_imgs_dir, '{}_pre_grasped_no_collision.png'.format(iteration))
                        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)

                        # turn ON collisions between robot and object, and close fingers
                        for i in range(p.getNumJoints(robot.arm.robot_id)):
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i,
                                                    linkIndexB=-1, enableCollision=True, physicsClientId=robot.pb_client.get_client_id())
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i,
                                                    linkIndexB=rack_link_id, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())

                        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
                        grasped_img_fname = osp.join(
                            eval_grasp_imgs_dir, '{}_pre_grasped.png'.format(iteration))
                        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)

                        time.sleep(0.8)
                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[
                            0]
                        jnt_pos_before_grasp = robot.arm.get_jpos()
                        soft_grasp_close(robot, finger_joint_id, force=50)
                        safeRemoveConstraint(o_cid)
                        time.sleep(0.8)
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False)
                        time.sleep(0.8)

                        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
                        grasped_img_fname = osp.join(
                            eval_grasp_imgs_dir, '{}_post_grasped.png'.format(iteration))
                        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)

                        if g_idx == 1:
                            grasp_success = object_is_still_grasped(
                                robot, obj_id, right_pad_id, left_pad_id)

                            if grasp_success:
                                # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                                safeCollisionFilterPair(
                                    obj_id, table_id, -1, -1, enableCollision=True)
                                robot.arm.eetool.open()
                                p.resetBasePositionAndOrientation(
                                    obj_id, obj_pos_before_grasp, ori)
                                soft_grasp_close(
                                    robot, finger_joint_id, force=40)
                                robot.arm.set_jpos(
                                    jnt_pos_before_grasp, ignore_physics=True)
                                cid = constraint_grasp_close(robot, obj_id)
                        #########################################################################################################

                        if offset_jnts is not None:
                            offset_plan = ik_helper.plan_joint_motion(
                                robot.arm.get_jpos(), offset_jnts)

                            if offset_plan is not None:
                                for jnt in offset_plan:
                                    robot.arm.set_jpos(jnt, wait=False)
                                    time.sleep(0.04)
                                robot.arm.set_jpos(offset_plan[-1], wait=True)

                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False)
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, rack_link_id, enableCollision=False)
                        time.sleep(1.0)
        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
        grasped_img_fname = osp.join(
            eval_grasp_imgs_dir, '{}_grasped.png'.format(iteration))
        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)
        if grasp_success:
            ####################################### get place pose ###########################################

            pre_place_jnt_pos1 = ik_helper.get_feasible_ik(
                pre_ee_end_pose1_list)
            pre_place_jnt_pos2 = ik_helper.get_feasible_ik(
                pre_ee_end_pose2_list)
            place_jnt_pos = ik_helper.get_feasible_ik(ee_end_pose_list)

            if place_jnt_pos is not None and pre_place_jnt_pos2 is not None and pre_place_jnt_pos1 is not None:
                plan1 = ik_helper.plan_joint_motion(
                    robot.arm.get_jpos(), pre_place_jnt_pos1)
                plan2 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos1, pre_place_jnt_pos2)
                plan3 = ik_helper.plan_joint_motion(
                    pre_place_jnt_pos2, place_jnt_pos)

                if plan1 is not None and plan2 is not None and plan3 is not None:
                    place_plan = plan1 + plan2

                    for jnt in place_plan:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.035)
                    robot.arm.set_jpos(place_plan[-1], wait=True)

                ################################################################################################################

                    # turn ON collisions between object and rack, and open fingers
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=True)
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, rack_link_id, enableCollision=True)

                    for jnt in plan3:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.075)
                    robot.arm.set_jpos(plan3[-1], wait=True)

                    p.changeDynamics(
                        obj_id, -1, linearDamping=5, angularDamping=5)
                    constraint_grasp_open(cid)
                    robot.arm.eetool.open()

                    time.sleep(0.2)
                    for i in range(p.getNumJoints(robot.arm.robot_id)):
                        safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i,
                                                linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
                    robot.arm.move_ee_xyz([0, 0.075, 0.075])
                    safeCollisionFilterPair(
                        obj_id, table_id, -1, -1, enableCollision=False)
                    time.sleep(4.0)

                    # observe and record outcome
                    obj_surf_contacts = p.getContactPoints(
                        obj_id, table_id, -1, placement_link_id)
                    touching_surf = len(obj_surf_contacts) > 0
                    obj_floor_contacts = p.getContactPoints(
                        obj_id, robot.arm.floor_id, -1, -1)
                    touching_floor = len(obj_floor_contacts) > 0
                    place_success = touching_surf and not touching_floor

            placed_rgb = robot.cam.get_images(get_rgb=True)[0]

            placed_img_fname = osp.join(
                eval_place_imgs_dir, '{}_placed_cam_{}.png'.format(iteration, i))
            np2img(placed_rgb[i].astype(np.uint8), placed_img_fname)
        robot.arm.go_home(ignore_physics=True)

        place_success_list.append(place_success)
        grasp_success_list.append(grasp_success)
        if not place_success:
            place_fail_list.append(iteration)
        if not grasp_success:
            grasp_fail_list.append(iteration)
        log_str = 'Iteration: %d, ' % iteration
        kvs = {}

        kvs['Place Success Rate'] = sum(
            place_success_list) / float(len(place_success_list))
        kvs['Place [teleport] Success Rate'] = sum(
            place_success_teleport_list) / float(len(place_success_teleport_list))
        kvs['Grasp Success Rate'] = sum(
            grasp_success_list) / float(len(grasp_success_list))

        kvs['Place Success'] = place_success_list[-1]
        kvs['Place [teleport] Success'] = place_success_teleport_list[-1]
        kvs['Grasp Success'] = grasp_success_list[-1]

        # f = open(txt_file_name, "a")
        # f.write("{} \n".format("place success"))
        # f.write(str(bool(place_success_list[-1])))
        # f.write("\n")
        # f.write("{} \n".format("Place [teleport] Success"))
        # f.write(str(bool(place_success_teleport_list[-1])))
        # f.write("\n")
        # f.write("{} \n".format("Grasp Success"))
        # f.write(str(bool(grasp_success_list[-1])))
        # f.write("\n")
        # f.close()

        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        id_str = ', shapenet_id: %s' % obj_shapenet_id
        log_info(log_str + id_str)

        robot.pb_client.remove_body(obj_id)
    # f = open(txt_file_name, "a")
    # f.write("{} \n".format("RATE place success"))
    # f.write(str(sum(place_success_list) / float(len(place_success_list))))
    # f.write("\n")
    # f.write("{} \n".format("RATE Place [teleport] Success"))
    # f.write(str(sum(place_success_teleport_list) / float(len(place_success_teleport_list))))
    # f.write("\n")
    # f.write("{} \n".format("Rate Grasp Success"))
    # f.write(str(sum(grasp_success_list) / float(len(grasp_success_list))))
    # f.write("\n")
    # f.close()

    # f = open(txt_file_name, "a")
    # f.write("{} \n".format("place_fail_list"))
    # f.write(str(place_fail_list))
    # f.write("\n")
    # f.write("{} \n".format("place_fail_teleport_list"))
    # f.write(str(place_fail_teleport_list))
    # f.write("\n")
    # f.write("{} \n".format("grasp_fail_list"))
    # f.write(str(grasp_fail_list))
    # f.write("\n")
    # f.close()


if __name__ == "__main__":

    signal.signal(signal.SIGINT, util.signal_handler)

    main()
