import torch
import torch.nn as nn

def init_linear_layers(layers, init_func = torch.nn.init.kaiming_normal_):
    """
    Initialize all the fully connected layers
    :param layers: (list<nn.Module>) a list of layers
    :param init_func: (callable) torch.nn.init.kaiming_normal_by default or some other init. method
    """
    for m in layers:
        if isinstance(m, nn.Linear):
            init_func(m.weight)

class Encoder(nn.Module):
    def __init__(self, encoder_params):
        super(Encoder, self).__init__()
        orig_backbone = torch.hub.load("pytorch/vision", encoder_params["backbone"], 
                                       weights="IMAGENET1K_V2")
        self.backbone = nn.Sequential(*list(orig_backbone.children())[:-2])
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        
        
    def forward(self, x):
        out = self.backbone(x)  # NxDxW'XH'
        out = self.avg_pooling_2d(out)  # output shape NxDx1
        out = out.view(out.size(0), -1)  # output shape NxD
        return out
    
class ResiPoseNet(nn.Module):
    def __init__(self, config):
        super(ResiPoseNet, self).__init__()
        self.config = config
        encoder_name = self.config["encoder_name"]
        encoder_params = self.config[encoder_name]
        output_dim = encoder_params["output_dim"]
        if encoder_name == "eigenplaces":
            self.encoder = torch.hub.load("gmberton/eigenplaces", "get_trained_model", 
                        backbone=encoder_params["backbone"], fc_output_dim=output_dim)
        elif encoder_name == "imagenet":
            self.encoder = Encoder(encoder_params)
        else:
            raise NotImplementedError("{} is not supported for encoder type".format(encoder_name))
        ori_repr = self.config["orientation_representation"]
        q_dim = 4
        if ori_repr == "quat":
            q_dim = 4
        elif ori_repr == "6d":
            q_dim = 6
        else: 
            raise Exception("{} not supported for orientation representation".format(ori_repr))
        mlp_dim = config["mlp_dim"] 
        k = config["num_clusters"] 
        self.cluster_classifier = nn.Sequential(nn.Linear(output_dim, mlp_dim), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(config["dropout"]),
                                 nn.Linear(mlp_dim, k),
                                 nn.LogSoftmax(dim=1))
        self.mlp = nn.Sequential(nn.Linear(output_dim, mlp_dim), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(config["dropout"]),
                                 nn.Linear(mlp_dim, mlp_dim))
        self.fc_x_res = nn.Linear(mlp_dim, 3)
        self.fc_q_res = nn.Linear(mlp_dim, q_dim)
        init_linear_layers(list(self.modules()))
        
    def forward(self, img):
        h = self.encoder(img)
        h_mlp = self.mlp(h)
        res_x = self.fc_x_res(h_mlp)
        res_q = self.fc_q_res(h_mlp)
        p = torch.cat([res_x,res_q], dim=1)        
        return {"logits":self.cluster_classifier(h), "p":p}


class ClsPose(nn.Module):
    def __init__(self, config):
        super(ClsPose, self).__init__()
        self.config = config
        encoder_name = self.config["encoder_name"]
        encoder_params = self.config[encoder_name]
        output_dim = encoder_params["output_dim"]
        
        self.position_encoder = nn.Linear(3, output_dim)
        self.orientation_encoder = nn.Linear(4, output_dim)

        
        if encoder_name == "eigenplaces":
            self.encoder = torch.hub.load("gmberton/eigenplaces", "get_trained_model", 
                        backbone=encoder_params["backbone"], fc_output_dim=output_dim)
        elif encoder_name == "imagenet":
            self.encoder = Encoder(encoder_params)
        else:
            raise NotImplementedError("{} is not supported for encoder type".format(encoder_name))
        
        self.cls_position = nn.Sequential(nn.Linear(output_dim*2, 256), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(config["dropout"]),
                                 nn.Linear(256, 2), nn.LogSoftmax(dim=1))
        self.cls_orientation = nn.Sequential(nn.Linear(output_dim*2, 256), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(config["dropout"]),
                                 nn.Linear(256, 2), nn.LogSoftmax(dim=1))
        
        mlp_dim = config["mlp_dim"] 
        self.mlp = nn.Sequential(nn.Linear(output_dim, mlp_dim), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(config["dropout"]),
                                 nn.Linear(mlp_dim, mlp_dim))
        self.fc_x = nn.Linear(mlp_dim, 3)
        self.fc_q = nn.Linear(mlp_dim, 4)
        
        init_linear_layers(list(self.modules()))
        
    def forward(self, img, pose):
        h = self.encoder(img)
        h_position = torch.cat((h, self.position_encoder(pose[:, :3])), dim=1)
        h_orientation = torch.cat((h, self.orientation_encoder(pose[:, 3:])), dim=1)
        h_p = self.mlp(h)
        x = self.fc_x(h_p)
        q = self.fc_q(h_p)
        p = torch.cat([x,q], dim=1)
        return {"theta_x":self.cls_position(h_position), 
                "theta_q":self.cls_orientation(h_orientation),
                "p": p}
        

class PoseNet(nn.Module):
    def __init__(self, config):
        super(PoseNet, self).__init__()
        self.config = config
        encoder_name = self.config["encoder_name"]
        encoder_params = self.config[encoder_name]
        output_dim = encoder_params["output_dim"]
        if encoder_name == "eigenplaces":
            self.encoder = torch.hub.load("gmberton/eigenplaces", "get_trained_model", 
                        backbone=encoder_params["backbone"], fc_output_dim=output_dim)
        elif encoder_name == "imagenet":
            self.encoder = Encoder(encoder_params)
        else:
            raise NotImplementedError("{} is not supported for encoder type".format(encoder_name))
        ori_repr = self.config["orientation_representation"]
        q_dim = 4
        if ori_repr == "quat":
            q_dim = 4
        elif ori_repr == "6d":
            q_dim = 6
        else: 
            raise Exception("{} not supported for orientation representation".format(ori_repr))
        mlp_dim = config["mlp_dim"] 
        self.mlp = nn.Sequential(nn.Linear(output_dim, mlp_dim), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(config["dropout"]),
                                 nn.Linear(mlp_dim, mlp_dim))
        self.fc_x = nn.Linear(mlp_dim, 3)
        self.fc_q = nn.Linear(mlp_dim, q_dim)
        init_linear_layers(list(self.modules()))
        
    def forward(self, features):
        features = self.mlp(self.encoder(features))
        x = self.fc_x(features)
        q = self.fc_q(features)
        return {"features":features,
                "pose": torch.cat([x,q], dim=1)}
        
class ClassiPose(nn.Module):
    def __init__(self, config):
        super(ClassiPose, self).__init__()
        self.config = config
        dim = self.config["model_dim"]
        self.pose_encoder = nn.Linear(7, dim)
        self.classifier_x = nn.Sequential(nn.Linear(dim, dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Dropout(config["dropout"]),
                                        nn.Linear(dim,2),
                                        nn.LogSoftmax(dim=1))
        self.classifier_q = nn.Sequential(nn.Linear(dim, dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Dropout(config["dropout"]),
                                        nn.Linear(dim, 2),
                                        nn.LogSoftmax(dim=1))
    def forward(self, features, pose):
        #features = torch.cat((features, self.pose_encoder(pose)), dim=1)
        return {"log_px":self.classifier_x(features), 
                "log_pq":self.classifier_q(features)}

        
        
        

 