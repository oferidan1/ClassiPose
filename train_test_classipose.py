"""
Entry point training and testing a classifier which takes 
a vector and pose and classifies their match
"""
import argparse
import torch
import torch.nn as nn 
import numpy as np
import logging
from utils import utils
import time
from datasets.camera_pose_dataset import CameraPoseDataset
from models.model import PoseNet, ClassiPose
from os.path import join
import yaml
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


def main(args):
    utils.init_logger()

    # Read configuration
    with open(args.config, "r") as read_file:
        config = yaml.safe_load(read_file)
    with open(args.apr_config, "r") as read_file:
        apr_config = yaml.safe_load(read_file)
     
    config["apr_params"] = apr_config["apr_params"]
    config["apr_checkpoint_path"] = apr_config["apr_checkpoint_path"]
    config["x_thr"] = apr_config["x_thr"]
    config["q_thr"] = apr_config["q_thr"]
    
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))
    

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)
    
    # Initialize the APR model
    apr = PoseNet(config["apr_params"]).to(device)
    # Load the checkpoint if needed
    if config["apr_checkpoint_path"]:
        apr.load_state_dict(torch.load(config["apr_checkpoint_path"], map_location=device_id))
        logging.info("Initializing APR from checkpoint: {}".format(config["apr_checkpoint_path"]))
    
    
    config["classifier_params"]["model_dim"] = config["apr_params"]["mlp_dim"]
    model = ClassiPose(config["classifier_params"]).to(device)
    if config["classifier_checkpoint_path"]:
        model.load_state_dict(torch.load(config["classifier_checkpoint_path"], map_location=device_id))
        logging.info("Initializing classifier from checkpoint: {}".format(config["classifier_checkpoint_path"]))
 

    if config["mode"] == 'train':
        # Set to train mode
        model.train()

        # Set the losses
        train_config = config["train_params"]
        loss_fn = nn.NLLLoss()
        
        # Set the optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(),
                                  lr=train_config['lr'],
                                  weight_decay=train_config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=train_config['lr_scheduler_step_size'],
                                                   gamma=train_config['lr_scheduler_gamma'])

        # Set the dataset and data loader
        dataset = CameraPoseDataset(config["dataset"])
        loader_params = {'batch_size': train_config['batch_size'],
                                  'shuffle': True,
                                  'num_workers': train_config.get('num_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = train_config["n_freq_print"]
        n_freq_checkpoint = train_config["n_freq_checkpoint"]
        n_epochs = train_config["n_epochs"]

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        logging.info("Start training")
        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in enumerate(dataloader):
                # TODO connect to tensor board or wandb 
                # TODO read align poses from config
                res = train_step(data, apr, model, optimizer, device, loss_fn, 
                                 config["x_thr"],
                                 config["q_thr"])

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("[Batch-{}/Epoch-{}] NLL loss: {:.6f} Prop Positive x: {:.3f} Prop Positive q: {:.3f}".format(
                                                                        batch_idx+1, epoch+1, 
                                                                        res["loss"], res["positives_x"], res["positives_q"]))
                    
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_cls_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()
        
        torch.save(model.state_dict(), checkpoint_prefix + '_cls_last.pth')


    else: # Test
        # Set the dataset and data loader
        dataset = CameraPoseDataset(config["dataset"])
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 4}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        pred_x = np.zeros(len(dataloader.dataset))
        pred_q = np.zeros(len(dataloader.dataset))
        cls_x = np.zeros(len(dataloader.dataset))
        cls_q = np.zeros(len(dataloader.dataset))
        labels_x = np.zeros(len(dataloader.dataset))
        labels_q = np.zeros(len(dataloader.dataset))
        runtime = np.zeros(len(dataloader.dataset))
        
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader, 0)):                
                #TODO change to batch mode 
                pred_x_i, pred_q_i, cls_x_i, cls_q_i, posit_err, orient_err, labels_x_i, labels_q_i, runtime_i = test_step(data, apr, model, device, config["x_thr"],
                                 config["q_thr"])
                # Collect statistics
                pred_x[i] = pred_x_i.item()
                pred_q[i] = pred_q_i.item()
                cls_x[i] = cls_x_i.item()
                cls_q[i] = cls_q_i.item()
                labels_x[i] = labels_x_i.item()
                labels_q[i] = labels_q_i.item()
                runtime[i] = runtime_i
                
                logging.info("posit err: {:.3f}, probability x < {}: [{:.3f}] (class: {}), orient err: {:.3f}, probability q < {}: [{:.3f}] (class {})".format(posit_err.item(), config["x_thr"], pred_x[i], cls_x[i],
                                                                                                                    orient_err.item(), config["q_thr"], pred_q[i], cls_q[i]))
              
        # Record overall statistics
        # roc auc, precision, recall, etc.
        precision_x, recall_x, f1_x, _ = precision_recall_fscore_support(labels_x, cls_x, average="binary")
        precision_q, recall_q, f1_q, _  = precision_recall_fscore_support(labels_q, cls_q, average="binary")
        logging.info("Position error classification: precision: {:.3f}, recall:{:.3f}, f1 score:{:.3f}".format(precision_x, recall_x, f1_x))
        logging.info("Orientation error classification: precision: {:.3f}, recall:{:.3f}, f1 score:{:.3f}".format(precision_q, recall_q, f1_q))
        
def train_step(data, apr, classifier, optimizer, device, loss_fn, 
                                 x_thr,
                                 q_thr):
    apr.eval() 
    classifier.train()
    gt_poses = data['pose'].to(device).to(dtype=torch.float32)
    input_poses = data['input_pose'].to(device).to(dtype=torch.float32)
    imgs = data['img'].to(device)
    batch_size = gt_poses.shape[0]

    # compute the input pose 
    with torch.no_grad():
        res = apr(imgs)
        features =  res['features']
        est_poses = res['pose']
        #fill_indices = torch.where(input_poses.sum(axis=1)==0)
        #input_poses[fill_indices] = est_poses[fill_indices]
        input_poses = est_poses
        labels_x, labels_q, posit_err, orient_err, = utils.get_labels(input_poses, gt_poses, x_thr, q_thr)
           
    # Zero the gradients
    optimizer.zero_grad()
    res = classifier(features, input_poses) 
    criterion = loss_fn(res["log_px"], labels_x) + loss_fn(res["log_pq"], labels_q)
    # Back prop
    criterion.backward()
    optimizer.step()
    # Return value for logging
    ret_val = {"batch_size":batch_size,
            "loss":criterion.item(),
            "posit_err": posit_err,
            "orient_err": orient_err,
            "positives_x": torch.sum(labels_x)/batch_size,
            "positives_q": torch.sum(labels_q)/batch_size}
    return ret_val  

def test_step(data, apr, classifier, device, x_thr, q_thr):
    apr.eval()
    classifier.eval()
    gt_poses = data.get('pose').to(dtype=torch.float32).to(device)
    imgs = data.get('img').to(device)
    tic = time.time()
    
    with torch.no_grad():
        res = apr(imgs)
        features =  res['features']
        est_poses = res['pose']
        res = classifier(features, est_poses)
        px = res["log_px"].exp()[:,1]
        pq  = res["log_pq"].exp()[:,1]
        clsx = (px > 0.5).to(dtype=torch.int64)
        clsq = (pq > 0.5).to(dtype=torch.int64)
    
    labels_x, labels_q, posit_err, orient_err = utils.get_labels(est_poses, gt_poses, x_thr, q_thr)
    toc = time.time()
    
    return px, pq, clsx, clsq, posit_err, orient_err, labels_x, labels_q, (toc-tic)*1000

    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="path to configuration file", default="train_classipose_config.yaml")
    arg_parser.add_argument("--apr_config", help="path to apr configuration file", default="apr_config.yaml")
    
    args = arg_parser.parse_args()
    main(args)