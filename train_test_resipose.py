"""
Entry point training and testing a baseline APR
"""
import argparse
import torch
import torch.nn as nn 
import numpy as np
import logging
from utils import utils
import time
from datasets.camera_pose_dataset import CameraPoseDataset
from models.losses import CameraPoseLoss
from models.model import ResiPoseNet
from os.path import join
import yaml


def main(args):
    utils.init_logger()

    # Read configuration
    with open(args.config, "r") as read_file:
        config = yaml.safe_load(read_file)
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
    model = ResiPoseNet(config["apr_params"]).to(device)
    # Load the checkpoint if needed
    if config["apr_checkpoint_path"]:
        model.load_state_dict(torch.load(config["apr_checkpoint_path"], map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(config["apr_checkpoint_path"]))

    if config['mode'] == 'train':
        # Set to train mode
        model.train()

        # Set the losses
        train_config = config["train_params"]
        pose_loss = CameraPoseLoss(train_config["pose_loss_params"]).to(device)
        loss_fn = nn.NLLLoss()
        
        # Set the optimizer and scheduler
        params = list(model.parameters())#list(model.parameters()) + list(pose_loss.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
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
        for epoch in range(n_epochs):
            for batch_idx, data in enumerate(dataloader):
                # TODO connect to tensor board or wandb 
                # TODO read align poses from config
                res = train_step(data, model, optimizer, device, loss_fn, pose_loss)                

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    batch_loss = res["loss"]
                    accuracy = res["accuracy"]      
                    posit_err = res["posit_err"]
                    orient_err = res["orient_err"]
                    
                    logging.info("[Batch-{}/Epoch-{}] loss: {:.3f}"
                                    ", accuracy: {:.2f}, pos err: {:.3f}, orient err: {:.3f},".format(
                                                                        batch_idx+1, epoch+1, 
                                                                        batch_loss,                                                                         
                                                                        accuracy,
                                                                        posit_err,
                                                                        orient_err))
                    
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_apr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()
        
        torch.save(model.state_dict(), checkpoint_prefix + '_apr_last.pth')
    elif config['mode'] == 'test':
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        dataset = CameraPoseDataset(config["dataset"])
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 4}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 5))

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):                
                
                accuracy, posit_err, orient_err, H, runtime = test_step(data, model, device)
                # Collect statistics
                stats[i, 0] = accuracy              
                stats[i, 1] = posit_err              
                stats[i, 2] = orient_err              
                stats[i, 3] = H
                stats[i, 4] = runtime

                logging.info("accuracy: {:.2f}, pos err: {:.3f}, orient err: {:.3f}, entrophy: {:.3f}, inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]))

        # Record overall statistics
        # TODO revisit nan issue
        logging.info("Mean accuracy: {:.3f}".format(np.mean(stats[:, 0])))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 1]), np.nanmedian(stats[:, 2])))
        logging.info("Mean entrophy:{:.2f}".format(np.mean(stats[:, 3])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 4])))


def train_step(data, model, optimizer, device, loss_fn, pose_loss):
    cluster_idx = data['cluster_idx'].type(torch.LongTensor).to(device)
    pose = data['pose'].to(device).to(dtype=torch.float32)
    imgs = data['img'].to(device)
    batch_size = cluster_idx.shape[0]

    # Zero the gradients
    optimizer.zero_grad()
    res = model(imgs)
    #est_poses = res.get('pose')
    logits = res.get("logits")
    est_pose = res.get("p")
    criterion = 10*loss_fn(logits, cluster_idx) + pose_loss(est_pose, pose)
    # Back prop
    criterion.backward()
    optimizer.step()    
    accuracy = torch.sum(logits.max(dim=1).indices==cluster_idx)/batch_size
    posit_err, orient_err = utils.pose_err(est_pose, pose)
    # Return value for logging
    ret_val = {"batch_size":batch_size,
            "loss":criterion.item(),
            "accuracy": accuracy,
            "posit_err": posit_err.mean().item(),
            "orient_err": orient_err.mean().item()}
    '''
    ret_val = {"batch_size":batch_size,
            "est_poses": est_poses.detach(),
            "poses":poses.detach(),
            "pose_loss":pose_criterion.item()}
    '''
    return ret_val  

def test_step(data, model, device):
    cluster_idx = data.get('cluster_idx').type(torch.LongTensor).to(device)
    pose = data['pose'].to(device).to(dtype=torch.float32)
    imgs = data.get('img').to(device)
    tic = time.time()
    batch_size = cluster_idx.shape[0]
    res = model(imgs)
    logits = res.get("logits")
    est_pose = res.get("p")
    toc = time.time()

    # Evaluate error
    # TODO metrics of classification 
    accuracy = torch.sum(logits.max(dim=1).indices==cluster_idx)/batch_size
    p = torch.exp(logits)
    H = -torch.sum(p*torch.log(p)) 
    posit_err, orient_err = utils.pose_err(est_pose, pose)    
    return accuracy, posit_err, orient_err, H, toc-tic
    
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="path to configuration file", default="train_apr_config.yaml")
    args = arg_parser.parse_args()
    main(args)