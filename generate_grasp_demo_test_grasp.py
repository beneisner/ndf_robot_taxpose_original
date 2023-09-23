from cgi import test
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

from ndf_robot.utils.sim_utils import get_object_clouds, get_clouds


def get_demo_pose_ori(grasp_data, get_jpos=False):
    data_dict = {}
    grasp_data_dict = dict(grasp_data)
    shapenet_id = grasp_data_dict['shapenet_id'].item()
    obj_pose_world = grasp_data_dict['obj_pose_world']
    pos = obj_pose_world[:3]
    ori = obj_pose_world[3:]
    data_dict['shapenet_id'] = shapenet_id
    data_dict['obj_pos'] = pos
    data_dict['obj_ori'] = ori

    if get_jpos:
        robot_joints = grasp_data_dict['robot_joints']
        data_dict['robot_joints'] = robot_joints
    return data_dict


def main(args, global_dict):
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={
                  'self_collision': False, 'seed': args.seed})
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_ndf_config(),
                            'eval_cfgs', args.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' %
                 config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(
        path_util.get_ndf_config(), args.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict['shapenet_obj_dir']
    obj_class = global_dict['object_class']
    eval_save_dir = global_dict['eval_save_dir']

    eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
    eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)

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

    if args.dgcnn:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='dgcnn',
            return_features=True,
            sigmoid=True,
            acts=args.acts).cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='pointnet',
            return_features=True,
            sigmoid=True).cuda()

    if not args.random:
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

    demo_shapenet_ids = []

    # get info from all demonstrations
    demo_target_info_list = []
    demo_rack_target_info_list = []

    if args.n_demos > 0:
        gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
        gp_fns = random.sample(gp_fns, args.n_demos)
        grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
        grasp_demo_filenames, place_demo_filenames = list(
            grasp_demo_filenames), list(place_demo_filenames)
        log_warn('USING ONLY %d DEMONSTRATIONS' % len(grasp_demo_filenames))
        print(grasp_demo_filenames, place_demo_filenames)
    else:
        log_warn('USING ALL %d DEMONSTRATIONS' % len(grasp_demo_filenames))

    grasp_demo_filenames = grasp_demo_filenames[:args.num_demo]
    place_demo_filenames = place_demo_filenames[:args.num_demo]

    grasp_data_list = []
    place_data_list = []
    demo_rel_mat_list = []

    # load all the demo data and look at objects to help decide on query points
    for i, fname in enumerate(grasp_demo_filenames):
        grasp_demo_fn = grasp_demo_filenames[i]
        place_demo_fn = place_demo_filenames[i]
        print('Loading demo from place_demo_fn: %s' % place_demo_fn)
        print('Loading demo from grasp_demo_fn: %s' % grasp_demo_fn)
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
        opt_iterations=args.opt_iterations)

    grasp_optimizer = OccNetOptimizer(
        model,
        query_pts=optimizer_gripper_pts,
        query_pts_real_shape=optimizer_gripper_pts_rs,
        opt_iterations=args.opt_iterations)
    grasp_optimizer.set_demo_info(demo_target_info_list)
    place_optimizer.set_demo_info(demo_rack_target_info_list)

    # get objects that we can use for testing
    test_object_ids = []
    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(
        shapenet_obj_dir)] if obj_class == 'mug' else os.listdir(shapenet_obj_dir)
    for s_id in shapenet_id_list:
        valid = s_id not in demo_shapenet_ids and s_id not in avoid_shapenet_ids

        if valid:
            test_object_ids.append(s_id)

    train_object_ids = demo_shapenet_ids

    if args.single_instance:
        test_object_ids = [demo_shapenet_ids[0]]

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

    log_info(f'CREATING DATA WITH SEED {args.seed}..........................')
    viz_data_list_test = []

    # Test ID
    # data_dir = 'test_data_ndf_mug_place_{}'.format(args.seed)
    # data_dir = 'test_data_ndf_mug_place_0'
    data_dir = 'bowl_test_new_{}_pregrasp'.format(args.seed)

    save_dir = os.path.join(os.getcwd(), data_dir, 'renders')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    failed_teleport_shapeids = []
    train_ids = []
    log_info("len(test_object_ids):{}".format(len(test_object_ids)))

    for iteration in range(len(place_demo_filenames)):
        place_demo_fn = place_demo_filenames[iteration]
        place_data = np.load(place_demo_fn, allow_pickle=True)
        demo_dict = get_demo_pose_ori(place_data, get_jpos=True)

        shapenet_id = demo_dict['shapenet_id']
        train_ids.append(shapenet_id)
    assert not set(train_ids).intersection(set(test_object_ids)
                                           ), "train_ids and test_object_ids should not intersect!"
    # test_object_ids = train_object_ids
    log_info("len(test_object_ids):{}".format(len(test_object_ids)))
    log_info("len(set(test_object_ids)):{}".format(len(set(test_object_ids))))

    rack_relative_pose_list_file = f'{data_dir}/rack_relative_pose_stacked.npz'
    if os.path.isfile(rack_relative_pose_list_file):
        rack_relative_pose_list = list(np.load(
            rack_relative_pose_list_file, allow_pickle=True)["rack_relative_pose_stacked"])
    else:
        rack_relative_pose_list = []

    for iteration in range(len(test_object_ids)):
        # for iteration in range(args.start_iteration, args.num_iterations):

        # load a test object
        # obj_shapenet_id = random.sample(test_object_ids, 1)[0]

        obj_shapenet_id = test_object_ids[iteration % len(test_object_ids)]
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        log_info(id_str)
        iteration = iteration

        viz_dict = {}  # will hold information that's useful for post-run visualizations
        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        if obj_class in ['bottle', 'jar', 'bowl', 'mug']:
            upright_orientation = common.euler2quat(
                [np.pi/2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(
            shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')

        obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

        scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
        scale_default = cfg.MESH_SCALE_DEFAULT
        if args.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high -
                                                scale_low) + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3

        if args.any_pose:
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

        viz_dict['shapenet_id'] = obj_shapenet_id
        viz_dict['obj_obj_file'] = obj_obj_file
        if 'normalized' not in shapenet_obj_dir:
            viz_dict['obj_obj_norm_file'] = osp.join(
                shapenet_obj_dir + '_normalized', obj_shapenet_id, 'models/model_normalized.obj')
        else:
            viz_dict['obj_obj_norm_file'] = osp.join(
                shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        viz_dict['obj_obj_file_dec'] = obj_obj_file_dec
        viz_dict['mesh_scale'] = mesh_scale

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

        if args.any_pose:
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
        if args.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(
            obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        cloud_points, cloud_colors, cloud_classes = get_clouds(
            cams, occlusion=args.occlusion)
        obj_points, obj_colors, obj_classes = get_object_clouds(
            cams, occlusion=args.occlusion)

        np.savez(f'{data_dir}/renders/{iteration}_init_all_points.npz',
                 clouds=cloud_points, colors=cloud_colors, classes=cloud_classes, shapenet_id=obj_shapenet_id)

        np.savez(f'{data_dir}/renders/{iteration}_init_obj_points.npz',
                 clouds=obj_points, colors=obj_colors, classes=obj_classes, shapenet_id=obj_shapenet_id)

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
        viz_dict['start_obj_pose'] = util.pose_stamped2list(obj_pose_world)

        cloud_segs = []
        cloud_points = []
        for i, cam in enumerate(cams.cams):
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
        # pre_grasp_ee_pose[2, -1] -= 0.005
        viz_dict['start_ee_pose'] = pre_grasp_ee_pose

        ########################### grasp post-process #############################
        new_grasp_pt = post_process_grasp_point(pre_grasp_ee_pose, target_obj_pcd_obs, thin_feature=(
            not args.non_thin_feature), grasp_viz=args.grasp_viz, grasp_dist_thresh=args.grasp_dist_thresh)
        pre_grasp_ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # optimize placement pose
        rack_pose_mats, best_rack_idx = place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False)
        rack_relative_pose = util.pose_stamped2list(
            util.pose_from_matrix(rack_pose_mats[best_rack_idx]))

        ee_end_pose = util.transform_pose(pose_source=util.list2pose_stamped(
            pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(rack_relative_pose))
        pre_ee_end_pose2 = util.transform_pose(
            pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
        pre_ee_end_pose1 = util.transform_pose(
            pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf)

        robot.pb_client.reset_body(obj_id, pos, ori)

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):

            # reset everything
            robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(
                obj_id, table_id, -1, -1, enableCollision=True)
            if args.any_pose:
                robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            if args.any_pose:
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

            if grasp_jnt_pos is not None and jnt_pos is not None:
                if g_idx == 0:
                    robot.pb_client.set_step_sim(True)
                    robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    robot.arm.eetool.close(ignore_physics=True)
                    time.sleep(0.2)
                    grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(
                        eval_grasp_imgs_dir, '%d.png' % iteration)
                    np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    continue

                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    plan1 = ik_helper.plan_joint_motion(
                        robot.arm.get_jpos(), jnt_pos)
                    plan2 = ik_helper.plan_joint_motion(jnt_pos, grasp_jnt_pos)
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

                        # turn ON collisions between robot and object, and close fingers
                        for i in range(p.getNumJoints(robot.arm.robot_id)):
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i,
                                                    linkIndexB=-1, enableCollision=True, physicsClientId=robot.pb_client.get_client_id())
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i,
                                                    linkIndexB=rack_link_id, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())

                        time.sleep(0.8)
                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[
                            0]
                        jnt_pos_before_grasp = robot.arm.get_jpos()
                        cloud_points, cloud_colors, cloud_classes = get_clouds(
                            cams)
                        obj_points, obj_colors, obj_classes = get_object_clouds(
                            cams)
                        soft_grasp_close(robot, finger_joint_id, force=50)
                        safeRemoveConstraint(o_cid)
                        time.sleep(0.8)
                        safeCollisionFilterPair(
                            obj_id, table_id, -1, -1, enableCollision=False)
                        time.sleep(0.8)

                        if g_idx == 1:
                            grasp_success = object_is_still_grasped(
                                robot, obj_id, right_pad_id, left_pad_id)

                            if grasp_success:

                                gripper_pose = robot.arm.get_ee_pose()
                                np.savez(f'{data_dir}/renders/{iteration}_pre_grasp_all_points.npz',
                                         clouds=cloud_points, colors=cloud_colors, classes=cloud_classes, shapenet_id=obj_shapenet_id, gripper_pose=gripper_pose)

                                np.savez(f'{data_dir}/renders/{iteration}_pre_grasp_obj_points.npz',
                                         clouds=obj_points, colors=obj_colors, classes=obj_classes, shapenet_id=obj_shapenet_id, gripper_pose=gripper_pose)
                            else:
                                failed_teleport_shapeids.append(
                                    obj_shapenet_id)
        grasp_success_list.append(grasp_success)

        robot.arm.go_home(ignore_physics=True)

        log_str = 'Iteration: %d, ' % iteration
        kvs = {}

        kvs['Place [teleport] Success Rate'] = sum(
            grasp_success_list) / float(len(grasp_success_list))
        kvs['Place [teleport] Success'] = grasp_success
        kvs['grasp_success'] = grasp_success
        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        id_str = ', shapenet_id: %s' % obj_shapenet_id
        log_info(log_str + id_str)

        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        if not osp.exists(eval_iter_dir):
            os.makedirs(eval_iter_dir)
        sample_fname = osp.join(
            eval_iter_dir, 'success_rate_eval_implicit.npz')
        np.savez(
            sample_fname,
            obj_shapenet_id=obj_shapenet_id,
            success=success_list,
            grasp_success=1,
            place_success=1,

            grasp_success_list=grasp_success_list,
            place_success_list=place_success_list,
            place_success_teleport_list=place_success_teleport_list,

            ee_transforms=pre_grasp_ee_pose_mats,
            obj_transforms=rack_pose_mats,
            mesh_file=obj_obj_file,
            distractor_info=None,
            args=args.__dict__,
            global_dict=global_dict,
            cfg=util.cn2dict(cfg),
            obj_cfg=util.cn2dict(obj_cfg)
        )

        robot.pb_client.remove_body(obj_id)
        log_info("failed_teleport_shapeids")
        log_info(failed_teleport_shapeids)
        # counter += 1
        # last_list_len = len(test_object_ids)
    # rack_relative_pose_stacked = np.stack(rack_relative_pose_list)
        # log_info("rack_relative_pose_stacked.shape")
        # log_info(rack_relative_pose_stacked.shape)
        # np.savez(f'{data_dir}/rack_relative_pose_stacked.npz',
        #         rack_relative_pose_stacked=rack_relative_pose_stacked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    # parser.add_argument('--demo_exp', type=str,
    #                     default='grasp_rim_hang_handle_gaussian_precise_w_shelf')
    parser.add_argument('--demo_exp', type=str,
                        default='grasp_rim_anywhere_place_shelf_all_methods_multi_instance')
    # parser.add_argument('--demo_exp', type=str,
    #                     default='grasp_side_place_shelf_start_upright_all_methods_multi_instance')
    parser.add_argument('--exp', type=str, default='debug_eval')
    # parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--object_class', type=str, default='bowl')
    parser.add_argument('--opt_iterations', type=int, default=250)
    parser.add_argument('--num_demo', type=int, default=12,
                        help='number of demos use')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--resume_iter', type=int, default=0)
    # parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--config', type=str, default='eval_bowl_gen')
    parser.add_argument('--model_path', type=str,
                        default='multi_category_weights')
    parser.add_argument('--save_vis_per_model', action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--noise_decay', type=float, default=0.75)
    parser.add_argument('--pybullet_viz', action='store_true', default=True)
    parser.add_argument('--dgcnn', action='store_true')
    parser.add_argument('--random', action='store_true',
                        help='utilize random weights')
    parser.add_argument('--early_weight', action='store_true',
                        help='utilize early weights')
    parser.add_argument('--late_weight', action='store_true',
                        help='utilize late weights')
    parser.add_argument('--rand_mesh_scale', action='store_true', default=True)
    parser.add_argument('--only_test_ids', action='store_true', default=True)
    parser.add_argument('--all_cat_model', action='store_true',
                        help='True if we want to use a model that was trained on multipl categories')
    parser.add_argument('--n_demos', type=int, default=0,
                        help='if some integer value greater than 0, we will only use that many demonstrations')
    parser.add_argument('--acts', type=str, default='all')
    parser.add_argument('--old_model', action='store_true',
                        help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    parser.add_argument('--save_all_opt_results', action='store_true',
                        help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--non_thin_feature',
                        action='store_true', default=False)
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--start_iteration', type=int, default=0)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--seed', type=int, default=3)
    # parser.add_argument('--num_samples', type=int, default=100)
    # parser.add_argument('--data_dir', type=str, default='data')
    # parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    # # parser.add_argument('--demo_exp', type=str,
    # #                     default='grasp_rim_hang_handle_gaussian_precise_w_shelf')
    # parser.add_argument('--demo_exp', type=str,
    #                     default='grasp_side_place_shelf_start_upright_all_methods_multi_instance')
    # parser.add_argument('--exp', type=str, default='debug_eval')
    # # parser.add_argument('--object_class', type=str, default='mug')
    # parser.add_argument('--object_class', type=str, default='bottle')
    # parser.add_argument('--opt_iterations', type=int, default=250)
    # parser.add_argument('--num_demo', type=int, default=12,
    #                     help='number of demos use')
    # parser.add_argument('--any_pose', action='store_true')
    # parser.add_argument('--num_iterations', type=int, default=100)
    # parser.add_argument('--resume_iter', type=int, default=0)
    # # parser.add_argument('--config', type=str, default='base_cfg')
    # parser.add_argument('--config', type=str, default='eval_bottle_gen')
    # parser.add_argument('--model_path', type=str,
    #                     default='multi_category_weights')
    # parser.add_argument('--save_vis_per_model', action='store_true')
    # parser.add_argument('--noise_scale', type=float, default=0.05)
    # parser.add_argument('--noise_decay', type=float, default=0.75)
    # parser.add_argument('--pybullet_viz', action='store_true', default=True)
    # parser.add_argument('--dgcnn', action='store_true')
    # parser.add_argument('--random', action='store_true',
    #                     help='utilize random weights')
    # parser.add_argument('--early_weight', action='store_true',
    #                     help='utilize early weights')
    # parser.add_argument('--late_weight', action='store_true',
    #                     help='utilize late weights')
    # parser.add_argument('--rand_mesh_scale', action='store_true', default=True)
    # parser.add_argument('--only_test_ids', action='store_true', default=True)
    # parser.add_argument('--all_cat_model', action='store_true',
    #                     help='True if we want to use a model that was trained on multipl categories')
    # parser.add_argument('--n_demos', type=int, default=0,
    #                     help='if some integer value greater than 0, we will only use that many demonstrations')
    # parser.add_argument('--acts', type=str, default='all')
    # parser.add_argument('--old_model', action='store_true',
    #                     help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    # parser.add_argument('--save_all_opt_results', action='store_true',
    #                     help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    # parser.add_argument('--grasp_viz', action='store_true')
    # parser.add_argument('--single_instance', action='store_true')
    # parser.add_argument('--non_thin_feature',
    #                     action='store_true', default=True)
    # parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    # # parser.add_argument('--start_iteration', type=int, default=0)
    parser.add_argument('--occlusion', action='store_true', default=False)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--num_samples', type=int, default=100)
    # parser.add_argument('--data_dir', type=str, default='data')
    # parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    # parser.add_argument('--demo_exp', type=str,
    #                     default='grasp_rim_hang_handle_gaussian_precise_w_shelf')
    # parser.add_argument('--exp', type=str, default='debug_eval')
    # parser.add_argument('--object_class', type=str, default='mug')
    # parser.add_argument('--opt_iterations', type=int, default=500)
    # parser.add_argument('--num_demo', type=int, default=12,
    #                     help='number of demos use')
    # parser.add_argument('--any_pose', action='store_true', default=True)
    # parser.add_argument('--num_iterations', type=int, default=30)
    # parser.add_argument('--resume_iter', type=int, default=0)
    # parser.add_argument('--config', type=str, default='base_cfg')
    # parser.add_argument('--model_path', type=str,
    #                     default='multi_category_weights')
    # parser.add_argument('--save_vis_per_model', action='store_true')
    # parser.add_argument('--noise_scale', type=float, default=0.05)
    # parser.add_argument('--noise_decay', type=float, default=0.75)
    # parser.add_argument('--pybullet_viz', action='store_true', default=True)
    # parser.add_argument('--dgcnn', action='store_true')
    # parser.add_argument('--random', action='store_true',
    #                     help='utilize random weights')
    # parser.add_argument('--early_weight', action='store_true',
    #                     help='utilize early weights')
    # parser.add_argument('--late_weight', action='store_true',
    #                     help='utilize late weights')
    # parser.add_argument('--rand_mesh_scale', action='store_true')
    # parser.add_argument('--only_test_ids', action='store_true', default=True)
    # parser.add_argument('--all_cat_model', action='store_true',
    #                     help='True if we want to use a model that was trained on multipl categories')
    # parser.add_argument('--n_demos', type=int, default=0,
    #                     help='if some integer value greater than 0, we will only use that many demonstrations')
    # parser.add_argument('--acts', type=str, default='all')
    # parser.add_argument('--old_model', action='store_true',
    #                     help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    # parser.add_argument('--save_all_opt_results', action='store_true',
    #                     help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    # parser.add_argument('--grasp_viz', action='store_true')
    # parser.add_argument('--single_instance', action='store_true')
    # parser.add_argument('--non_thin_feature', action='store_true')
    # parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    # parser.add_argument('--start_iteration', type=int, default=0)
    # parser.add_argument('--occlusion', action='store_true', default=False)

    args = parser.parse_args()

    signal.signal(signal.SIGINT, util.signal_handler)

    obj_class = args.object_class
    shapenet_obj_dir = osp.join(
        path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')
    log_info("shapenet_obj_dir")
    log_info(shapenet_obj_dir)
    demo_load_dir = osp.join(path_util.get_ndf_data(),
                             'demos', obj_class, args.demo_exp)

    print("demo_load_dir")
    print(demo_load_dir)

    expstr = 'exp--' + str(args.exp)
    modelstr = 'model--' + str(args.model_path)
    seedstr = 'seed--' + str(args.seed)
    full_experiment_name = '_'.join([expstr, modelstr, seedstr])
    eval_save_dir = osp.join(path_util.get_ndf_eval_data(
    ), args.eval_data_dir, full_experiment_name)
    util.safe_makedirs(eval_save_dir)

    vnn_model_path = osp.join(
        path_util.get_ndf_model_weights(), args.model_path + '.pth')

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path
    )

    main(args, global_dict)
