
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
ds_name,_ = os.path.splitext(os.path.basename(args.eval_dataset_cfg_file_path))
args.save_dir = join("test", args.save_dir, f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{ds_name}_{args.model}")
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

args.features_dim = 14*768
if args.eval_dataset_name.startswith("pitts"):     # set infer_batch_size = 8 for pitts30k/pitts250k
    args.infer_batch_size = args.infer_batch_size // 2
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

cfg_file = args.eval_dataset_cfg_file_path
cfg = read_yaml_cfg(cfg_file)

######################################### MODEL #########################################
# Choose model based on args.model
if args.model == "CricaVPR":
    model = network.CricaVPRNet()
    model = model.to(args.device)
    if args.resume is not None:
        logging.info(f"Resuming model from {args.resume}")
        model = util.resume_model(args, model)
elif args.model == "Cosplace":
    model = util.load_cosplace(dim=1024)
    model = model.to(args.device)
    args.features_dim = 1024
    logging.info(f"Resuming model from {args.model}")
else:
    raise ValueError(f"Unknown model type: {args.model}")

model = model.to(args.device)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim

    # first, load PCA if it is pre-computed
    pca_model_save_dir = join("test", "model")
    pca_model_path = os.path.join(pca_model_save_dir, f'{ds_name}_pca_model.pkl')
    # Create the directory if it doesn't exist
    if not os.path.exists(pca_model_save_dir):
        os.makedirs(pca_model_save_dir)

    if os.path.exists(pca_model_path):
        pca = util.load_pca(args, pca_model_path)
        logging.info(f"Loaded pre-computed PCA model from {pca_model_path}")
    else:
        # if not, compute and save the PCA model
        pca_ds = build_dataset(cfg.data_cfg.test.dataset_cfg)
        logging.info(f"PCA set: {ds_name}/{cfg.data_cfg.test.dataset_cfg.data_split}")
        pca = util.compute_pca(args, model, pca_ds, full_features_dim)
        util.save_pca(args, pca, pca_model_path)
        logging.info(f"Computed and saved PCA model to {pca_model_path}")

######################################### DATASETS #########################################
# test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
# cfg_file = "/mnt/nas/share-all/kecen/code/Uranus/config/dataset/retrieval/inside_car_dense_det.yaml"
test_ds = build_dataset(cfg.data_cfg.test.dataset_cfg)
logging.info(f"Test set: {ds_name}/{cfg.data_cfg.test.dataset_cfg.data_split}")
######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
logging.info(f"Recalls on {ds_name}: {recalls_str}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
