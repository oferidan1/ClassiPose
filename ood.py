import pandas as pd 
from os.path import join
import numpy as np  
from sklearn import metrics
from sklearn.cluster import KMeans
from glob import glob 
import os
import torch

def load_features(files_path, is_normalize=False):    
    files_npy = glob(files_path + '/*.npy')
    feats = []
    names = []
    for f in files_npy:
        feat = np.load(f)
        #feat = np.expand_dims(feat, axis=0)
        filename = os.path.splitext(os.path.basename(f))[0]
        feats.append(feat)
        names.append(filename)        
    feats = np.array(feats)
    if is_normalize:
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)    
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))
        feats = prepos_feat(feats)
    return feats, names

def get_cluster_members(c_id, cluster_ids, train_features, train_poses):
    # get the indices of cluster and retrieve cluster poses and features 
    c_indices = np.where(cluster_ids == c_id)[0]
    c_poses = train_poses[c_indices]
    c_features = train_features[c_indices] # TODO we might want to not hold all train features
    return c_indices, c_poses, c_features 

def get_within_cluster_stats(x, metric='l2'):
    # compute pairwise distances
    m = metrics.pairwise_distances(x, metric=metric)
    # get element in the upper triangle (excluding diagonal)
    distances = m[np.triu_indices(m.shape[0], k = 1)] 
    return {"mean": np.mean(distances), 
            "std": np.std(distances),
            "q95": np.quantile(distances, 0.95)}

def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    #imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]    
    imgs_paths = df['filename'].values
    x_error = df['x_error'].values
    ang_error = df['ang_error'].values
    pose_predict = []
    pose_gt = []    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for i, row in df.iterrows():        
        pose_x = row['predict_pose_x']
        pose_q = row['predict_pose_q']
        pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in pose_x.split(',')]))
        pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in pose_q.split(',')]))
        pose = np.concatenate((pose_x, pose_q), axis=0)
        pose_predict.append(pose)
        
        pose_x = row['gt_pose_x']
        pose_q = row['gt_pose_q']
        pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in pose_x.split(',')]))
        pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in pose_q.split(',')]))
        pose = np.concatenate((pose_x, pose_q), axis=0)
        pose_gt.append(pose)
        
    pose_predict = np.array(pose_predict)
    pose_gt = np.array(pose_gt)    
    return imgs_paths, pose_predict, pose_gt, x_error, ang_error


scene = "shop"
model = "dfnet"
thr_x = 1
thr_q = 6
dataset = {}
for split in ["train", "test"]:
    dataset[split] = {"dataset_path": "F:/CambridgeLandmarks/"+scene,
                  #"labels_file": "F:/CambridgeLandmarks/"+scene+"/GT_"+scene+"_"+split+".csv",
                  "labels_file": "D:/ofer/pose_ood/features/"+scene+"/"+model+"_features_"+split+"/dfnet_res.csv",
                   "features_path":"D:/ofer/pose_ood/features/"+scene+"/"+model+"_features_"+split }

# offline: train set 
train_img_paths, train_pose_predict, train_pose_gt, _, _ = read_labels_file(dataset["train"]["labels_file"], 
                                                dataset["train"]["dataset_path"])
is_normalize = True
train_features, train_names = load_features(dataset["train"]["features_path"], is_normalize) # consider normalization 

# cluster poses 
# TODO need to have a more robust clustering method - automatic way to evaluate or refine clusters
k = 10
k_means = KMeans(n_clusters=k, random_state=0, n_init="auto")
k_means.fit(train_pose_gt)
centers = k_means.cluster_centers_
cluster_ids = k_means.predict(train_pose_gt)
clusters_stats = {}
for c_id in range(k):
    # get cluster members
    _, c_poses, c_features = get_cluster_members(c_id, cluster_ids, 
                                                 train_features, train_pose_gt)
    # compute centroids
    pose_centroid = centers[c_id]
    feature_centroid = np.mean(c_features, axis=0)
    
    # compute stats of within-cluster distances 
    c_posit_stats = get_within_cluster_stats(c_poses[:, :3], metric='l2')
    c_orient_stats = get_within_cluster_stats(c_poses[:, 3:], metric='l2')
    c_features_stats = get_within_cluster_stats(c_features, metric='l2')
    clusters_stats[c_id] = {"pose_centroid": pose_centroid,
                           "feature_centroid": feature_centroid,
                           "c_posit_stats": c_posit_stats,
                           "c_orient_stats": c_orient_stats,
                           "c_features_stats": c_features_stats}
    # TODO distances to centroid
    # TODO between cluster stats



# ONLINE: OOD Analysis
correct = 0
total = 0
test_img_paths, test_pose_predict, test_pose_gt, test_x_error, test_ang_error = read_labels_file(dataset["test"]["labels_file"], 
                                                dataset["test"]["dataset_path"])
test_features, test_names = load_features(dataset["test"]["features_path"], is_normalize) # consider normalization 
for i, est_pose in enumerate(test_pose_predict):
    feature_vec = test_features[i]
    cid = k_means.predict(np.expand_dims(est_pose, axis=0))[0] # TODO get the cluster id by pose estimate or feature? 
    # retrieve cluster centroid, stats, features and poses

    # TODO maybe it's enough to compute distance only to the centroid and see if it is > mean/0.95 quantile
    _, c_pose, c_feature = get_cluster_members(c_id, cluster_ids, 
                                                 train_features, train_pose_gt)
    c_stats = clusters_stats[cid]
    c_feature_centroid = c_stats["feature_centroid"]
    c_pose_centroid = c_stats["pose_centroid"]    
    #d_feat = # distance between feature_vec and centroid    
    # calc distance to centroid and comapre to quantile95
    c_posit_stats = c_stats["c_posit_stats"]
    c_orient_stats = c_stats["c_orient_stats"]    
    est_ood = 0
    posit_dist = abs((est_pose[:3] - c_pose_centroid[:3]).mean())
    orient_dist = abs((est_pose[3:] - c_pose_centroid[3:]).mean())
    if posit_dist > c_posit_stats["q95"] or orient_dist > c_orient_stats["q95"]:
        est_ood = 1        
            
    # compute if real ood
    real_ood = 0
    if test_x_error[i] > thr_x or test_ang_error[i] > thr_q:
        real_ood = 1 
        
    correct += (est_ood == real_ood)
    total += 1

accuracy = correct/total
print("accuracy: ", accuracy)
    



# online test set 
# for each image
# get feature vector, pose estimate and ground truth pose
# compute 