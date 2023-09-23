import os, os.path as osp
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

SEED = 1
obj_class = 'mug'



def make_video(image_folder,video_name):
    import cv2
    import os

    image_folder = image_folder
    video_name = os.path.join(image_folder,video_name)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
 
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def make_video_2(image_folder, video_name):
    import os
    import moviepy.video.io.ImageSequenceClip
    image_folder=image_folder
    fps=10

    image_files = [os.path.join(image_folder,img)
                for img in sorted(os.listdir(image_folder))
                if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)

def main(args):
    idx = 4
    set_log_level('info')

    robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': args.seed})
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    cfg = get_eval_cfg_defaults()
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(path_util.get_ndf_config(), args.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    # shapenet_obj_dir = global_dict['shapenet_obj_dir']
    obj_class = 'mug'
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')
    eval_save_dir = osp.join(path_util.get_ndf_eval_data(), args.eval_data_dir, 'custom_grasp')

    eval_grasp_imgs_dir = osp.join(eval_save_dir, str(idx),'grasp_imgs')
    eval_teleport_imgs_dir = osp.join(eval_save_dir, str(idx),'teleport_imgs')
    eval_place_imgs_dir = osp.join(eval_save_dir, str(idx),'place_imgs')
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)
    util.safe_makedirs(eval_place_imgs_dir)

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10
    p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
    p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)

    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    preplace_horizontal_tf_list = cfg.PREPLACE_HORIZONTAL_OFFSET_TF
    preplace_horizontal_tf = util.list2pose_stamped(cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
    preplace_offset_tf = util.list2pose_stamped(cfg.PREPLACE_OFFSET_TF)

    ## Graphics Loading Params ##
    load_shelf = False
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, args.demo_exp)
    # get filenames of all the demo files
    demo_filenames = os.listdir(demo_load_dir)
    assert len(demo_filenames), 'No demonstrations found in path: %s!' % demo_load_dir

    # strip the filenames to properly pair up each demo file
    grasp_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in demo_filenames if 'grasp_demo' in fn]  # use the grasp names as a reference
    place_demo_filenames = []
    grasp_demo_filenames = []
    for i, fname in enumerate(grasp_demo_filenames_orig):
        shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
        place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)

        if osp.exists(place_fname):
            grasp_demo_filenames.append(fname)
            place_demo_filenames.append(place_fname)
        else:
            log_warn('Could not find corresponding placement demo: %s, skipping ' % place_fname)

    if args.n_demos > 0:
        gp_fns = list(zip(grasp_demo_filenames, place_demo_filenames))
        gp_fns = random.sample(gp_fns, args.n_demos)
        grasp_demo_filenames, place_demo_filenames = zip(*gp_fns)
        grasp_demo_filenames, place_demo_filenames = list(grasp_demo_filenames), list(place_demo_filenames)
        log_warn('USING ONLY %d DEMONSTRATIONS' % len(grasp_demo_filenames))
        print(grasp_demo_filenames, place_demo_filenames)
    else:
        log_warn('USING ALL %d DEMONSTRATIONS' % len(grasp_demo_filenames))

    grasp_demo_filenames = grasp_demo_filenames[:args.num_demo]
    place_demo_filenames = place_demo_filenames[:args.num_demo]

    grasp_data_list = []
    place_data_list = []
    demo_rel_mat_list = []
    
    #########################################################3
 
    grasp_demo_fn = grasp_demo_filenames[idx]
    place_demo_fn = place_demo_filenames[idx]
    grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
    place_data = np.load(place_demo_fn, allow_pickle=True)
    print(grasp_data.files)
    print(place_data.files)
    # ['shapenet_id', 'ee_pose_world', 'robot_joints', 'obj_pose_world', 'obj_pose_camera',\
    # 'object_pointcloud', 'depth', 'seg', 'camera_poses', 'obj_model_file',\
    # 'obj_model_file_dec', 'gripper_pts', 'gripper_pts_gaussian', 'gripper_pts_uniform',\
    # 'gripper_contact_pose', 'table_urdf']

    #['shapenet_id', 'ee_pose_world', 'robot_joints', 'obj_pose_world', 'obj_pose_camera', \
    # 'object_pointcloud', 'depth', 'seg', 'camera_poses', 'obj_model_file', \
    # 'obj_model_file_dec', 'gripper_pts', 'rack_pointcloud_observed', 'rack_pointcloud_gt',\
    # 'rack_pointcloud_gaussian', 'rack_pointcloud_uniform', 'rack_pose_world', \
    # 'rack_contact_pose', 'shelf_pose_world', 'shelf_pointcloud_observed', \
    # 'shelf_pointcloud_uniform', 'shelf_pointcloud_gt', 'table_urdf']

    obj_pose = grasp_data['obj_pose_world'].tolist()
    start_ee_pose = grasp_data['gripper_contact_pose'].tolist()
    end_ee_pose = place_data['ee_pose_world'].tolist()
    place_rel_mat = util.get_transform(
        pose_frame_target=util.list2pose_stamped(end_ee_pose),
        pose_frame_source=util.list2pose_stamped(start_ee_pose)
    )
    # start_ee_pose = get_ee_offset(ee_pose=start_ee_pose)
    place_rel_mat = util.matrix_from_pose(place_rel_mat)
    demo_rel_mat_list.append(place_rel_mat)


    obj_shapenet_id = str(grasp_data['shapenet_id'])
    print(obj_shapenet_id)

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
    tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
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
    for iteration in range(args.start_iteration, args.num_iterations):
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        log_info(id_str)
        upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

        scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
        scale_default = cfg.MESH_SCALE_DEFAULT
        if args.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale=[scale_default] * 3

        pos, ori = obj_pose[:3], obj_pose[3:]
        viz_dict = {}
        viz_dict['shapenet_id'] = obj_shapenet_id
        viz_dict['obj_obj_file'] = obj_obj_file
        if 'normalized' not in shapenet_obj_dir:
            viz_dict['obj_obj_norm_file'] = osp.join(shapenet_obj_dir + '_normalized', obj_shapenet_id, 'models/model_normalized.obj')
        else:
            viz_dict['obj_obj_norm_file'] = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
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
        obj_id = robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=obj_pose[:3],
            base_ori=obj_pose[3:])
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)
        ik_helper.add_collision_bodies({'0':obj_id,'1:':table_id})
        
        o_cid = None

        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # Get point cloud pre grasp
        cloud_points, cloud_colors, cloud_classes = get_object_clouds(cams)
        
        mug_id = 0
        rack_id = 1
        gripper_id = 2

        mug_points = cloud_points[cloud_classes==mug_id]
        rack_points = cloud_points[cloud_classes==rack_id]
        gripper_points = cloud_points[cloud_classes==gripper_id]
        np.save('mug_pcd_obs_pregrasp.npy', mug_points)
        np.save('rack_pcd_obs_pregrasp.npy', rack_points)
        np.save('gripper_pcd_obs_pregrasp.npy', gripper_points)

        hide_link(table_id, rack_link_id)
        if obj_class == 'mug':
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        # reset everything
        robot.pb_client.set_step_sim(False)
        robot.pb_client.reset_body(obj_id, pos, ori)
        # safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        # safeRemoveConstraint(o_cid)
        # p.resetBasePositionAndOrientation(obj_id, pos, ori)
 
        time.sleep(0.5)
        robot.arm.go_home(ignore_physics=True)

        ## Open Gripper ##
        def get_ik(goal_pose):
            ee_pose = ik_helper.get_feasible_ik(goal_pose)
            if not ee_pose:
                ee_pose = ik_helper.get_ik(goal_pose)
                if not ee_pose:
                    ee_pose = robot.arm.compute_ik(goal_pose[:3], goal_pose[3:])
            return ee_pose
        grasp_pose = get_ik(start_ee_pose)
        place_pose = get_ik(end_ee_pose)
        assert grasp_pose, 'grasp_pose returned None'
        assert place_pose, 'place_pose returned None'

        # print(robot.arm.get_jpos())
        # robot.pb_client.set_step_sim(True)
        # robot.arm.set_jpos(robot.arm.get_jpos(), ignore_physics=True)
        # robot.arm.eetool.close(ignore_physics=True)
        # time.sleep(0.2)

        plan1 = ik_helper.plan_joint_motion(robot.arm.get_jpos(), grasp_pose)
        
        assert plan1, 'plan1 returned None'
        # plan3 = ik_helper.plan_joint_motion(grasp_offset_pose, grasp_pose)
        # assert plan3, 'plan1 returned None'
        robot.arm.eetool.open()
        # ### Execute ###
        # for i in range(p.getNumJoints(robot.arm.robot_id)):
        #     safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
        #     safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())

     
  
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        counter = 0
        for jnt in plan1:
            counter +=1
            robot.arm.set_jpos(jnt, wait=False)
            grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
            grasped_img_fname = osp.join(eval_teleport_imgs_dir, '{}_post_grasped_{}.png'.format(iteration, str(counter).zfill(6)))
            np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)
            time.sleep(0.025)
        robot.arm.set_jpos(plan1[-1], wait=True)

        # for jnt in plan3:
        #     robot.arm.set_jpos(jnt, wait=False)
        #     time.sleep(0.025)
        # robot.arm.set_jpos(plan3[-1], wait=True)

        ## Grasp - Close Gripper ##
        soft_grasp_close(robot, finger_joint_id, force=50)
        time.sleep(0.2)
        cid = constraint_grasp_close(robot, obj_id)
        # Get point cloud at grasp
        cloud_points, cloud_colors, cloud_classes = get_object_clouds(cams)
        
        mug_id = 0
        rack_id = 1
        gripper_id = 2

        mug_points = cloud_points[cloud_classes==mug_id]
        rack_points = cloud_points[cloud_classes==rack_id]
        gripper_points = cloud_points[cloud_classes==gripper_id]
        np.save('mug_pcd_obs_atgrasp.npy', mug_points)
        np.save('rack_pcd_obs_atgrasp.npy', rack_points)
        np.save('gripper_pcd_obs_atgrasp.npy', gripper_points)
        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
        grasped_img_fname = osp.join(eval_grasp_imgs_dir, '{}_post_grasped.png'.format(iteration))
        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)
        
        plan2 = ik_helper.plan_joint_motion(robot.arm.get_jpos(), place_pose)
        assert plan2, 'plan2 returned None'
        for jnt in plan2:
            counter +=1
            robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
            grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
            grasped_img_fname = osp.join(eval_teleport_imgs_dir, '{}_post_grasped_{}.png'.format(iteration, str(counter).zfill(6)))
            np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)
        robot.arm.set_jpos(plan2[-1], wait=True)
        # Get point cloud after place
        cloud_points, cloud_colors, cloud_classes = get_object_clouds(cams)
        
        mug_id = 0
        rack_id = 1
        gripper_id = 2

        mug_points = cloud_points[cloud_classes==mug_id]
        rack_points = cloud_points[cloud_classes==rack_id]
        gripper_points = cloud_points[cloud_classes==gripper_id]
        np.save('mug_pcd_obs_atplace.npy', mug_points)
        np.save('rack_pcd_obs_atplace.npy', rack_points)
        np.save('gripper_pcd_obs_atplace.npy', gripper_points)
        constraint_grasp_open(cid)
        robot.arm.eetool.open()
        time.sleep(0.2)
        counter +=1
        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
        grasped_img_fname = osp.join(eval_teleport_imgs_dir, '{}_post_grasped_{}.png'.format(iteration, str(counter).zfill(6)))
        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)

        # Get point cloud after place
        cloud_points, cloud_colors, cloud_classes = get_object_clouds(cams)
        
        mug_id = 0
        rack_id = 1
        gripper_id = 2

        mug_points = cloud_points[cloud_classes==mug_id]
        rack_points = cloud_points[cloud_classes==rack_id]
        gripper_points = cloud_points[cloud_classes==gripper_id]
        np.save('mug_pcd_obs_postplace.npy', mug_points)
        np.save('rack_pcd_obs_postplace.npy', rack_points)
        np.save('gripper_pcd_obs_postplace.npy', gripper_points)


        
        robot.arm.move_ee_xyz([0, 0.075, 0.075])
        time.sleep(4.0)


        counter +=1
        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
        grasped_img_fname = osp.join(eval_teleport_imgs_dir, '{}_post_grasped_{}.png'.format(iteration, str(counter).zfill(6)))
        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)


        # observe and record outcome
        obj_surf_contacts = p.getContactPoints(obj_id, table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        obj_floor_contacts = p.getContactPoints(obj_id, robot.arm.floor_id, -1, -1)
        touching_floor = len(obj_floor_contacts) > 0
        place_success = touching_surf and not touching_floor

        counter +=1
        grasped_rgb = robot.cam.get_images(get_rgb=True)[0]
        grasped_img_fname = osp.join(eval_teleport_imgs_dir, '{}_post_grasped_{}.png'.format(iteration, str(counter).zfill(6)))
        np2img(grasped_rgb.astype(np.uint8), grasped_img_fname)
 
        robot.arm.go_home(ignore_physics=True)

        make_video_2(eval_teleport_imgs_dir,'grasp_demo_{}_collision_on.mp4'.format(idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_data_dir', type=str, default='debug')
    parser.add_argument('--demo_exp', type=str, default='grasp_rim_hang_handle_gaussian_precise_w_shelf')
    parser.add_argument('--exp', type=str, default='debug_eval')
    parser.add_argument('--object_class', type=str, default='mug')
    parser.add_argument('--opt_iterations', type=int, default=250)
    parser.add_argument('--num_demo', type=int, default=12, help='number of demos use')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--save_vis_per_model', action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--noise_decay', type=float, default=0.75)
    parser.add_argument('--pybullet_viz', action='store_true', default=True)
    parser.add_argument('--dgcnn', action='store_true')
    parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--early_weight', action='store_true', help='utilize early weights')
    parser.add_argument('--late_weight', action='store_true', help='utilize late weights')
    parser.add_argument('--rand_mesh_scale', action='store_true')
    parser.add_argument('--only_test_ids', action='store_true')
    parser.add_argument('--all_cat_model', action='store_true', help='True if we want to use a model that was trained on multipl categories')
    parser.add_argument('--n_demos', type=int, default=0, help='if some integer value greater than 0, we will only use that many demonstrations')
    parser.add_argument('--acts', type=str, default='all')
    parser.add_argument('--old_model', action='store_true', help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    parser.add_argument('--save_all_opt_results', action='store_true', help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--start_iteration', type=int, default=0)

    args = parser.parse_args()
    main(args)