import os, os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import shutil 
import pybullet as p
from PIL import Image

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from ndf_robot.utils.sim_utils import get_clouds, get_object_clouds
from airobot.utils.common import euler2quat

from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import np2img
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import path_util
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.eval_gen_utils import safeCollisionFilterPair


# posegraph imports
import sys
sys.path.insert(1, '/home/exx/Documents/equivariant_pose_graph/python')
import torch
from pytorch3d.ops import sample_farthest_points
from equivariant_pose_graph.training.flow_equivariance_training_module_centering import EquivarianceTrainingModule
from equivariant_pose_graph.dataset.point_cloud_data_module import MultiviewDataModule
from equivariant_pose_graph.models.transformer_flow import ResidualFlow, ResidualFlow_V1, \
    ResidualFlow_V2, ResidualFlow_V3,ResidualFlow_V4, ResidualFlow_Correspondence,\
    ResidualFlow_Identity, ResidualFlow_PE, ResidualFlow_DiffEmb, ResidualFlow_DiffEmbTransformer

def load_data(num_points, clouds, classes, action_class, anchor_class):
    points_raw_np = clouds
    classes_raw_np = classes

    points_action_np = points_raw_np[classes_raw_np == action_class].copy()
    points_action_mean_np = points_action_np.mean(axis=0)
    points_action_np = points_action_np - points_action_mean_np
    
    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
    points_anchor_np = points_anchor_np - points_action_mean_np

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)
    points_action, points_anchor = subsample(num_points,points_action, points_anchor)
    return points_action.cuda(), points_anchor.cuda()

def subsample(num_points,points_action,points_anchor):
    if(points_action.shape[1] > num_points):
        points_action, _ = sample_farthest_points(points_action, 
            K=num_points, random_start_point=True)
    elif(points_action.shape[1] < num_points):
        raise NotImplementedError(f'Action point cloud is smaller than cloud size ({points_action.shape[1]} < {num_points})')

    if(points_anchor.shape[1] > num_points):
        points_anchor, _ = sample_farthest_points(points_anchor, 
            K=num_points, random_start_point=True)
    elif(points_anchor.shape[1] < num_points):
        raise NotImplementedError(f'Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {num_points})')
    
    return points_action, points_anchor
# posegraph imports
# from ndf_robot.utils.equivariant_pose_graph_utils import get_model

def main(args):
    robot = Robot('franka', pb_cfg={'gui': True}, arm_cfg={'self_collision': False, 'seed': args.seed})
    ik_helper = FrankaIK(gui=False)
    
    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', args.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' % config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(path_util.get_ndf_config(), args.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    obj_class = 'mug'
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10
    p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
    p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)
    
    test_object_ids = ['2852b888abae54b0e3523e99fd841f4']
    
    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

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
    # '/home/bokorn/src/ndf_robot/src/ndf_robot/data/demos/mug/grasp_rim_hang_handle_gaussian_precise_w_shelf/grasp_demo_5c48d471200d2bf16e8a121e6886e18d.npz'
    # this is the URDF that was used in the demos -- make sure we load an identical one
    tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
    table_id = robot.pb_client.load_urdf(tmp_urdf_fname,
                            cfg.TABLE_POS,
                            table_ori,
                            scaling=cfg.TABLE_SCALING)

    rack_link_id = 0

    # load a test object
    obj_shapenet_id = '2852b888abae54b0e3523e99fd841f4'
    id_str = 'Shapenet ID: %s' % obj_shapenet_id
    print(id_str)

    mug_id = 0
    rack_id = 1
    gripper_id = 2
    
    # for testing, use the "normalized" object
    obj_obj_file = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
    obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

    scale_default = cfg.MESH_SCALE_DEFAULT
    mesh_scale=[scale_default] * 3

    obj_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
    obj_translation = [(x_high+x_low)/2., (y_high+y_low)/2., table_z]
    
    pose_base = util.list2pose_stamped(obj_translation + obj_orientation)

    T_init = np.array(
        [[ 0.3665,  0.9304,  0.0000,  0.0000],
         [-0.9304,  0.3665,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  1.0000,  0.0000],
         [ 0.2960, -0.8170,  0.0000,  1.0000]]
    ).T
    
    pose_init = util.transform_pose(pose_base, util.pose_from_matrix(T_init))
    pose_init_list = util.pose_stamped2list(pose_init)
    
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

    obj_id = robot.pb_client.load_geom(
        'mesh',
        mass=0.01,
        mesh_scale=mesh_scale,
        visualfile=obj_obj_file_dec,
        collifile=obj_obj_file_dec,
        base_pos=pose_init_list[:3],
        base_ori=pose_init_list[3:])
    p.changeDynamics(obj_id, -1, lateralFriction=0.5)
    
    safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
    p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
    time.sleep(1.5)



    pos_cur, ori_cur = robot.pb_client.get_body_state(obj_id)[:2]
    pose_cur_list = np.concatenate([pos_cur, ori_cur])
    pose_cur = util.list2pose_stamped(pose_cur_list)    
    pose_pre_transform = pose_cur

    cloud_points_init, cloud_colors_init, cloud_classes_init = get_object_clouds(cams)

    #############################
    ### ADD NETWORK CODE HERE ###
    #############################
    iteration = 0

    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache() 
    network = ResidualFlow_DiffEmbTransformer(emb_nn='dgcnn')
 
    place_model = EquivarianceTrainingModule(
        network)
    place_model.cuda()
    # checkpoint_file = '/home/exx/media/DataDrive/singularity_chuerp/equiv_pgraph_logs/train_test_mr_dcpflow_residual0_attn_trans0.1_rot180_diffembnntrans_dgcnn/equiv_dcpflow/version_1/checkpoints/epoch=83-step=10500.ckpt'
    checkpoint_file='/home/exx/media/DataDrive/singularity_chuerp/equiv_pgraph_logs/train_test_mr_dcpflow_residual0_attn_trans0.1_rot180_diffembnntrans_dgcnn/equiv_dcpflow/version_2/saved_ckpts/epoch=34-step=4375.ckpt'
    place_model.load_state_dict(torch.load(checkpoint_file)['state_dict'])
    log_info("Model Loaded from " + str(checkpoint_file))
    # robot.pb_client.set_step_sim(False)
    # time.sleep(10)

    
    points_action, points_anchor = load_data(num_points=1024, clouds = cloud_points_init ,classes = cloud_classes_init, action_class= 0, anchor_class= 1)
    pred_T_action_init, pred_T_anchor  = place_model(points_action, points_anchor) # 1, 4, 4
    T_pred = pred_T_action_init.get_matrix().detach().cpu().numpy()[0].T

    # T_pred = np.eye(4)
    # T_pred[0:2,-1] = 0.2

    data_dir = args.data_dir
    save_dir = os.path.join('/home/exx/Documents/ndf_robot/src/ndf_robot',data_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    log_info(f'{save_dir}/{iteration}_init_obj_points.npz')
    np.savez(f'{save_dir}/{iteration}_init_obj_points.npz', 
                 clouds = cloud_points_init, colors = cloud_colors_init, classes = cloud_classes_init, T_pred = T_pred)
    # np.savez(f'{save_dir}/{iteration}_init_obj_points.npz', 
    #              clouds = cloud_points_init, colors = cloud_colors_init, classes = cloud_classes_init, pred_T_action_init = pred_T_action_init.get_matrix().detach().cpu(), T_pred = T_pred)
    # T_pred = np.array(
    #     [[-0.2500, -0.7964, -0.5508,  0.0000],
    #      [ 0.4658, -0.5977,  0.6523,  0.0000],
    #      [-0.8491, -0.0934,  0.5200,  0.0000],
    #      [ 0.0442,  0.5593,  0.0954,  1.0000]]).T

    # T_pred = np.array(
    #     [[ 0.2141, -0.8560, -0.4719,  0.0000],
    #      [ 0.3320, -0.3906,  0.8589,  0.0000],
    #      [-0.9189, -0.3403,  0.2004,  0.0000],
    #      [ 0.0648,  0.5762,  0.0915,  1.0000]]).T
    pose_pred = util.transform_pose(pose_pre_transform, util.pose_from_matrix(T_pred))
    pose_pred_list = util.pose_stamped2list(pose_pred)

    robot.pb_client.set_step_sim(True)
    robot.pb_client.reset_body(obj_id, pose_pred_list[:3], pose_pred_list[3:])
    time.sleep(10)
 
    cloud_points_trans, cloud_colors_trans, cloud_classes_trans = get_object_clouds(cams)
 
    np.savez(f'{save_dir}/{iteration}_teleport_obj_points.npz', 
                 clouds = cloud_points_trans, colors = cloud_colors_trans, classes = cloud_classes_trans)
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--data_dir', type=str, default='place_test_0_my_model_may12_dgcnn')

    args = parser.parse_args()

    signal.signal(signal.SIGINT, util.signal_handler)

    main(args)#, global_dict)
