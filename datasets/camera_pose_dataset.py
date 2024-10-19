from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import cv2
from utils import utils

class CameraPoseDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, config):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :param sample_pose: (bool) whether to return a pose guess 
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        dataset_path = config["dataset_path"]
        labels_file = config["labels_file"]
        resi_pose = config["resi_pose"]
        self.img_paths, self.poses, self.clusters = read_labels_file_clusters(labels_file, dataset_path, resi_pose)
        self.dataset_size = self.poses.shape[0]
        
        self.transform = None
        if config["transform"]:
            self.transform = utils.transforms[config["transform"]]
        
        self.sample_poses_probabilities = config.get("pose_sample_probs")
        #assert not self.sample_poses_probabilities or len(self.sample_poses_probabilities) == 2

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):        
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx], 
                                              cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        pose = self.poses[idx]
        cluster_idx = self.clusters[idx]
        if self.transform:
            img = self.transform(img)
            
        sample = {"img": img, "pose": pose, "cluster_idx": cluster_idx}
        return sample


class CameraPoseDatasetDeprecated(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, config):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :param sample_pose: (bool) whether to return a pose guess 
        :return: an instance of the class
        """
        super(CameraPoseDatasetDeprecated, self).__init__()
        dataset_path = config["dataset_path"]
        labels_file = config["labels_file"]
        self.img_paths, self.poses = read_labels_file(labels_file, dataset_path)
        self.dataset_size = self.poses.shape[0]
        
        self.transform = None
        if config["transform"]:
            self.transform = utils.transforms[config["transform"]]
        
        self.sample_poses_probabilities = config.get("pose_sample_probs")
        #assert not self.sample_poses_probabilities or len(self.sample_poses_probabilities) == 2

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):        
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx], 
                                              cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        pose = self.poses[idx]
        if self.transform:
            img = self.transform(img)
            
        idx_range = np.arange(np.max([idx-5, 0]), np.min([idx+5, self.dataset_size]))
        
        sample = {'img': img, 'pose': pose}
        if self.sample_poses_probabilities:
            input_pose = np.zeros(7) # fill from the actual estimate
            #p = np.random.rand()
            
            #if p < self.sample_poses_probabilities[0]:
            #    input_pose = self.poses[np.random.randint(self.dataset_size)]
            #elif p < self.sample_poses_probabilities[1] and p >= self.sample_poses_probabilities[0]:
            #    input_pose = pose
        else:
            input_pose = pose
        sample['input_pose'] = input_pose
            
        return sample


def read_labels_file_clusters(labels_file, dataset_path, resi_pose):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    clusters = df['cluster_idx'].values
    n = df.shape[0]
    poses = np.zeros((n, 7))
    if resi_pose:
        poses[:, 0] = df['x_rel_0'].values
        poses[:, 1] = df['x_rel_1'].values
        poses[:, 2] = df['x_rel_2'].values
        poses[:, 3] = df['q_rel_0'].values
        poses[:, 4] = df['q_rel_1'].values
        poses[:, 5] = df['q_rel_2'].values
        poses[:, 6] = df['q_rel_3'].values
    else:
        poses[:, 0] = df['t1'].values
        poses[:, 1] = df['t2'].values
        poses[:, 2] = df['t3'].values
        poses[:, 3] = df['q1'].values
        poses[:, 4] = df['q2'].values
        poses[:, 5] = df['q3'].values
        poses[:, 6] = df['q4'].values
    return imgs_paths, poses, clusters

def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]    
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses


    