import torch
from datasets.camera_pose_dataset import CameraPoseDatasetDeprecated
import pandas as pd
from utils import utils
import numpy as np
from os.path import join
import os
import argparse
import yaml

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="path to configuration file", default="train_apr_config.yaml")
    args = arg_parser.parse_args()


    out_path = "eigen/test/"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    device_id = "cuda:0"
      # Read configuration
    with open(args.config, "r") as read_file:
        config = yaml.safe_load(read_file)
    encoder_name = config["apr_params"]["encoder_name"]
    encoder_params = config["apr_params"][encoder_name]
    output_dim = encoder_params["output_dim"]
    # Create and load the model
    x = torch.cuda.is_available()
    device = torch.device(device_id)
    encoder = torch.hub.load("gmberton/eigenplaces", "get_trained_model", 
                         backbone=encoder_params["backbone"], fc_output_dim=output_dim).to(device)
    encoder.eval()

    # Create the dataset and loader
    dataset = CameraPoseDatasetDeprecated(config["dataset"])
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': 1}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
    labels_file = config["dataset"]["labels_file"]

    # Generate encodings and save them
    n = len(dataloader.dataset)
    netvlad_paths = []
    with torch.no_grad():  # This is essential to avoid fradients accumulating on the GPU, which will cause memory blow-up
        for i, minibatch in enumerate(dataloader):
            img_path = dataloader.dataset.img_paths[i]
            print("Encoding image {}/{} at {} ".format(i, n, img_path))
            img = minibatch.get('img').to(device)
            global_desc = encoder(img).squeeze(0).cpu().numpy()*100000000
            outfile_name = img_path.replace(config["dataset"]["dataset_path"], "").replace("/", "_").\
                replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace("\\","")
            outfile_name = join(out_path, outfile_name)
            np.savez(outfile_name, global_desc)
            netvlad_paths.append(outfile_name.replace(out_path, "")+".npz")
            print("Encoding saved to {}".format(outfile_name))

    df = pd.read_csv(labels_file)
    df["eigen_path"] = netvlad_paths
    df.to_csv(labels_file+"_with_eigen.csv", index=False)