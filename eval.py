
import os
import torch
import parser
import logging
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
import network
import warnings

import numpy as np

import sys
sys.path.append('../../../..')
from uranus.data import build_data_loader, build_dataset
from uranus.data.generator.registry import build_matcher_data_generator
from uranus.utils.config import read_yaml_cfg
from uranus.utils.matcher_utils import (compute_epipolar_error, compute_pose_error, estimate_pose)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
test_ds_name,_ = os.path.splitext(os.path.basename(args.eval_dataset_cfg_file_path))
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S_')+ test_ds_name)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

args.features_dim = 14*768
if args.eval_dataset_name.startswith("pitts"):     # set infer_batch_size = 8 for pitts30k/pitts250k
    args.infer_batch_size = args.infer_batch_size // 2
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.CricaVPRNet()
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
# test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")

cfg_file = args.eval_dataset_cfg_file_path
# cfg_file = "/mnt/nas/share-all/kecen/code/Uranus/config/dataset/retrieval/inside_car_dense_det.yaml"
cfg = read_yaml_cfg(cfg_file)
test_ds = build_dataset(cfg.data_cfg.test.dataset_cfg)
logging.info(f"Test set: {test_ds_name}")
######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
logging.info(f"Recalls on {test_ds_name}: {recalls_str}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
