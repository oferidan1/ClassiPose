import pandas as pd
import numpy as np
from os.path import join, basename
import argparse
from sklearn.cluster import KMeans
import pickle
import transforms3d as t3d

def get_knn_indices(query, db):
    distances = np.linalg.norm(db-query, axis=1)
    return np.argsort(distances)

def read_labels_file(labels_file):
    df = pd.read_csv(labels_file)
    imgs_paths = df['img_path'].values
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

def compute_rel_pose(p1, p2):
    t1 = p1[:3]
    q1 = p1[3:]
    rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

    t2 = p2[:3]
    q2 = p2[3:]
    rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

    t_rel = t2 - t1
    rot_rel = np.dot(np.linalg.inv(rot1), rot2)
    q_rel = t3d.quaternions.mat2quat(rot_rel)
    return t_rel, q_rel

def get_nn_from_train(poses, train_clusters_poses, train_cluster_imgs):
    clusters = []
    cluster_imgs = []
    x_rels = []
    q_rels = []
    for i in range(len(poses)):
        pose = poses[i]
        #distance from clusters
        min_dist = 1000000        
        cluster_idx = -1
        for j in range(len(train_clusters_poses)):
            train_pose = train_clusters_poses[j]
            dist = ((pose[:3] - train_pose[:3])**2).mean() + 400*((pose[3:] - train_pose[3:])**2).mean()                        
            if dist < min_dist:
                min_dist = dist
                cluster_idx = j
                min_train_pose = train_pose
        #calc pose diff from nn
        x_rel, q_rel = compute_rel_pose(pose, min_train_pose)
        clusters.append(cluster_idx)
        cluster_imgs.append(train_cluster_imgs[cluster_idx])
        x_rels.append(x_rel)
        q_rels.append(q_rel)
    clusters = np.array(clusters)
    x_rels = np.array(x_rels)
    q_rels = np.array(q_rels)
    return clusters, cluster_imgs, x_rels, q_rels    

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path_db", help="path where netvlads are", default="D:/yoli/classipose/eigen/train/")
    arg_parser.add_argument("--db_labels_file", default="F:\CambridgeLandmarks\ShopFacade\GT_ShopFacade_train.csv_with_eigen.csv")
    arg_parser.add_argument("--train_labels_file", help="labels", default="F:\CambridgeLandmarks\ShopFacade\GT_ShopFacade_train.csv")    
    arg_parser.add_argument("--test_labels_file", help="labels", default="F:\CambridgeLandmarks\ShopFacade\GT_ShopFacade_test.csv")    
    arg_parser.add_argument("--is_train", help="is_train", type=int, default=1)
    arg_parser.add_argument("--k", help="k", type=int, default=10)
    # arg_parser.add_argument("--data_path_db", help="path where netvlads are", default="D:/yoli/classipose/eigen/test/")
    # arg_parser.add_argument("--db_labels_file", default="F:\CambridgeLandmarks\ShopFacade\GT_ShopFacade_test.csv_with_eigen.csv")
    # arg_parser.add_argument("--labels_file", help="labels", default="F:\CambridgeLandmarks\ShopFacade\GT_ShopFacade_test.csv")        
    # arg_parser.add_argument("--is_train", help="is_train", type=int, default=0)

    args = arg_parser.parse_args()
    
    db_df = pd.read_csv(args.db_labels_file)
    eigen_df = db_df["eigen_path"]    

    dim = 2048
    num_neighbors = 100
    start_index = 0

    n = db_df.shape[0]    
    # read db into memory
    db = np.zeros((n, dim))    
    for i, path in enumerate(eigen_df):
        path = path.replace("\\", "")
        vec = np.load(join(args.data_path_db, path))['arr_0']/100000000
        db[i, :] = vec
        
    kmeans_path = "out/kmeans_model_" + str(args.k) + ".pkl"
    if args.is_train:
        #k = n//10
        k = args.k
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(db)        
        #idx = kmeans.labels_                 
        #find closest image in train to centroid
        centers = kmeans.cluster_centers_    
        sim = np.dot(centers, db.T)
        train_idx = np.argmax(sim, axis=1)            
        
        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
    else:
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)
        #idx = kmeans.predict(db)
    #print(idx)
        
    imgs_paths, train_poses = read_labels_file(args.train_labels_file)
    train_cluster_poses = []
    train_cluster_imgs = []
    for i in train_idx:
        train_cluster_poses.append(train_poses[i])    
        train_cluster_imgs.append(imgs_paths[i])
    clusters, cluster_imgs, x_rels, q_rels = get_nn_from_train(train_poses, train_cluster_poses, train_cluster_imgs)
    df = pd.read_csv(args.train_labels_file)
    df["cluster_idx"] = clusters
    df["x_rel_0"] = x_rels[:,0]
    df["x_rel_1"] = x_rels[:,1]
    df["x_rel_2"] = x_rels[:,2]
    df["q_rel_0"] = q_rels[:,0]
    df["q_rel_1"] = q_rels[:,1]
    df["q_rel_2"] = q_rels[:,2]
    df["q_rel_3"] = q_rels[:,3]
    df["cluster img"] = cluster_imgs
    df.to_csv(args.train_labels_file+"_with_cluster_idx.csv", index=False)

    imgs_paths, test_poses = read_labels_file(args.test_labels_file)    
    clusters, cluster_imgs, x_rels, q_rels = get_nn_from_train(test_poses, train_cluster_poses, train_cluster_imgs)
    df = pd.read_csv(args.test_labels_file)
    df["cluster_idx"] = clusters
    df["x_rel_0"] = x_rels[:,0]
    df["x_rel_1"] = x_rels[:,1]
    df["x_rel_2"] = x_rels[:,2]
    df["q_rel_0"] = q_rels[:,0]
    df["q_rel_1"] = q_rels[:,1]
    df["q_rel_2"] = q_rels[:,2]
    df["q_rel_3"] = q_rels[:,3]
    df["cluster img"] = cluster_imgs
    df.to_csv(args.test_labels_file+"_with_cluster_idx.csv", index=False)

    f.close()