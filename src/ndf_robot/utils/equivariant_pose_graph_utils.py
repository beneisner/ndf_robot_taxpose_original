import sys
sys.path.insert(1, '/home/exx/Documents/equivariant_pose_graph/python')
 
from equivariant_pose_graph.training.flow_equivariance_training_module import EquivarianceTrainingModule
from equivariant_pose_graph.dataset.point_cloud_data_module import MultiviewDataModule
from equivariant_pose_graph.models.transformer_flow import ResidualFlow, ResidualFlow_V1, \
    ResidualFlow_V2, ResidualFlow_V3,ResidualFlow_V4, ResidualFlow_Correspondence,\
    ResidualFlow_Identity, ResidualFlow_PE
import os, torch

import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from equivariant_pose_graph.dataset.point_cloud_data_module import MultiviewDataModule
import hydra

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["PYOPENGL_PLATFORM"]="egl"
# chuerp conda env: pytorch3d_38
@hydra.main(config_path="../configs", config_name="exx_dcpflow_residual0") 

def get_model(cfg):
    pl.seed_everything(cfg.seed)

    if cfg.flow_compute_type == 0:
        network = ResidualFlow()
    elif cfg.flow_compute_type == 1:
        network = ResidualFlow_V1()
    elif cfg.flow_compute_type == 2:
        network = ResidualFlow_V2()
    elif cfg.flow_compute_type == 3:
        network = ResidualFlow_V3()
    elif cfg.flow_compute_type == 4:
        network = ResidualFlow_V4()
    elif cfg.flow_compute_type == 5:
        network = ResidualFlow_Correspondence()
    elif cfg.flow_compute_type == 6:
        network = ResidualFlow_Identity()
    elif cfg.flow_compute_type == 'pe':
        network = ResidualFlow_PE()
    model = EquivarianceTrainingModule(
        network, 
        lr = cfg.lr, 
        image_log_period = cfg.image_logging_period)
    model.cuda()
    model.train()
 

    if(cfg.checkpoint_file is not None):
        model.load_state_dict(torch.load(cfg.checkpoint_file)['state_dict'])
    return model
 
# if __name__ == '__main__':
#     model = get_model()
 