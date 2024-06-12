
import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm

import datasets_ws

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


# def compute_pca(args, model, pca_dataset_folder, full_features_dim):
#     model = model.eval()
#     pca_ds = datasets_ws.PCADataset(args, args.eval_datasets_folder, pca_dataset_folder)
#     dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
#     pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
#     with torch.no_grad():
#         for i, images in enumerate(dl):
#             if i*args.infer_batch_size >= len(pca_features):
#                 break
#             features = model(images).cpu().numpy()
#             pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
#     pca = PCA(args.pca_dim)
#     pca.fit(pca_features)
#     return pca


def compute_pca(args, model, pca_dataset, full_features_dim):
    model = model.eval()
    pca_ds = pca_dataset
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, data in enumerate(tqdm(dl, desc="Computing PCA features")):
            images, _, _ = data
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca

def save_pca(args, pca_model, saved_path):
    with open(saved_path,'wb') as pickle_file:
        pickle.dump(pca_model,pickle_file)

def load_pca(args, model_path):
    with open(model_path,'rb') as pickle_file:
        pca = pickle.load(pickle_file)
    return pca

def load_cosplace(dim):
    model = torch.hub.load("gmberton/cosplace",
                           "get_trained_model",
                           backbone="ResNet50",
                           fc_output_dim=dim)
    return model