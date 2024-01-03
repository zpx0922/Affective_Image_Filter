import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal,normal_style
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from torchvision import transforms
import sys
from torch.hub import tqdm, load_state_dict_from_url as load_url
from torchvision import models
from torchvision.transforms import Resize


s_mean = (0.52, 0.465, 0.40)
s_std = (0.22, 0.21,0.19)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x

class Discriminator(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.text_linear = MLP(40,1)

        self.bce_loss = nn.BCELoss()

        self.fc_0 = torch.nn.Sequential(*[ torch.nn.Linear(64+64+768, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()])

        self.fc_1 = torch.nn.Sequential(*[ torch.nn.Linear(128+128+768, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()])

        self.fc_2 = torch.nn.Sequential(*[ torch.nn.Linear(256+256+768, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()])
                     
        self.fc_3 = torch.nn.Sequential(*[ torch.nn.Linear(512+512+768, 512), nn.BatchNorm1d(512), torch.nn.ReLU(),
                                    torch.nn.Linear(512, 128), nn.BatchNorm1d(128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Sigmoid()]) 

        self.fc_4 = torch.nn.Sequential(*[ torch.nn.Linear(512+512+768, 512), nn.BatchNorm1d(512), torch.nn.ReLU(),
                                    torch.nn.Linear(512, 128), nn.BatchNorm1d(128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Sigmoid()]) 

        self.fc_un_0 = torch.nn.Sequential(*[ torch.nn.Linear(128, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 1), torch.nn.Sigmoid()])
        self.fc_un_1 = torch.nn.Sequential(*[ torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()])
        self.fc_un_2 = torch.nn.Sequential(*[ torch.nn.Linear(512, 128), nn.BatchNorm1d(128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 32), nn.BatchNorm1d(32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 1), torch.nn.Sigmoid()])
        self.fc_un_3 = torch.nn.Sequential(*[ torch.nn.Linear(1024, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()])
        self.fc_un_4 = torch.nn.Sequential(*[ torch.nn.Linear(1024, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()])

    def cal_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std


    def forward(self, feats, ins, flag):
        ins = ins.permute(0,2,1)
        ins = self.text_linear(ins).squeeze()
        mean_0, std_0 = self.cal_mean_std(feats[0])
        mean_1, std_1 = self.cal_mean_std(feats[1])
        mean_2, std_2 = self.cal_mean_std(feats[2])
        mean_3, std_3 = self.cal_mean_std(feats[3])
        mean_4, std_4 = self.cal_mean_std(feats[4])
        features_0 = torch.cat([mean_0,std_0,ins],dim=1)
        features_1 = torch.cat([mean_1,std_1,ins],dim=1)
        features_2 = torch.cat([mean_2,std_2,ins],dim=1)
        features_3 = torch.cat([mean_3,std_3,ins],dim=1)
        features_4 = torch.cat([mean_4,std_4,ins],dim=1)
        out_0 = self.fc_0(features_0)
        out_1 = self.fc_1(features_1)
        out_2 = self.fc_2(features_2)
        out_3 = self.fc_3(features_3)
        out_4 = self.fc_4(features_4)
        features_un_0 = torch.cat([mean_0,std_0],dim=1)
        features_un_1 = torch.cat([mean_1,std_1],dim=1)
        features_un_2 = torch.cat([mean_2,std_2],dim=1)
        features_un_3 = torch.cat([mean_3,std_3],dim=1)
        features_un_4 = torch.cat([mean_4,std_4],dim=1)
        out_un_0 = self.fc_un_0(features_un_0)
        out_un_1 = self.fc_un_1(features_un_1)
        out_un_2 = self.fc_un_2(features_un_2)
        out_un_3 = self.fc_un_3(features_un_3)
        out_un_4 = self.fc_un_4(features_un_4)
        loss_0 = self.bce_loss(out_0, torch.ones(out_0.shape).cuda())
        loss_1 = self.bce_loss(out_1, torch.ones(out_1.shape).cuda())
        loss_2 = self.bce_loss(out_2, torch.ones(out_2.shape).cuda())
        loss_3 = self.bce_loss(out_3, torch.ones(out_3.shape).cuda())
        loss_4 = self.bce_loss(out_4, torch.ones(out_4.shape).cuda())
        loss_un_0 = self.bce_loss(out_un_0, torch.ones(out_un_0.shape).cuda())
        loss_un_1 = self.bce_loss(out_un_1, torch.ones(out_un_1.shape).cuda())
        loss_un_2 = self.bce_loss(out_un_2, torch.ones(out_un_2.shape).cuda())
        loss_un_3 = self.bce_loss(out_un_3, torch.ones(out_un_3.shape).cuda())
        loss_un_4 = self.bce_loss(out_un_4, torch.ones(out_un_4.shape).cuda())
        loss = 0.1 * (loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_un_0 + loss_un_1 + loss_un_2 + loss_un_3 + loss_un_4)

        return loss


class classify_label(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.fc_0 = torch.nn.Sequential(*[ torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 16)])
        self.fc_1 = torch.nn.Sequential(*[ torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 16)])
        self.fc_2 = torch.nn.Sequential(*[ torch.nn.Linear(512, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 16)])
        self.fc_3 = torch.nn.Sequential(*[ torch.nn.Linear(1024, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 16)])
        self.fc_4 = torch.nn.Sequential(*[ torch.nn.Linear(1024, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 16)])

        self.linear_label = torch.nn.Sequential(*[torch.nn.Linear(80, 40), torch.nn.ReLU(),
                                    torch.nn.Linear(40, 8)])

        self.kl_loss = nn.KLDivLoss()
    
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def cal_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std

    def forward(self,image,label):
        img_feats = self.encode_with_intermediate(image)
        mean_0, std_0 = self.cal_mean_std(img_feats[0])
        mean_1, std_1 = self.cal_mean_std(img_feats[1])
        mean_2, std_2 = self.cal_mean_std(img_feats[2])
        mean_3, std_3 = self.cal_mean_std(img_feats[3])
        mean_4, std_4 = self.cal_mean_std(img_feats[4])
        features_0 = torch.cat([mean_0,std_0],dim=1)
        features_1 = torch.cat([mean_1,std_1],dim=1)
        features_2 = torch.cat([mean_2,std_2],dim=1)
        features_3 = torch.cat([mean_3,std_3],dim=1)
        features_4 = torch.cat([mean_4,std_4],dim=1)
        out_0 = self.fc_0(features_0)
        out_1 = self.fc_1(features_1)
        out_2 = self.fc_2(features_2)
        out_3 = self.fc_3(features_3)
        out_4 = self.fc_4(features_4)
        out_all = torch.cat([out_0,out_1,out_2,out_3,out_4],dim=1)
        label_predict = self.linear_label(out_all)

        label=torch.tensor(label,dtype=torch.float32).cuda()
        label_predict=torch.tensor(label_predict,dtype=torch.float32).cuda()
        loss_mean = self.kl_loss(F.log_softmax(label_predict, dim=1), F.softmax(label, dim=1))

        return loss_mean


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class MLP(nn.Module): 
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, num_i, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_o)

    def forward(self, x):
        x = self.linear1(x)
        return x


class AIFTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,encoder,decoder,PatchEmbed, transformer, conv, device, conv1, conv2, conv3, conv4, conv5, label_model):

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed
        self.device = device
        self.conv_1x1_encoder = conv
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            for param in getattr(self,name).parameters():
                param.requires_grad = False
        self.distance_l2 = torch.nn.PairwiseDistance(p=2).to(device)
        self.label_model = label_model
        for param in label_model.parameters():
            param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def TV_loss(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def cal_SV(self,content_feats):
        gram_pre_1 = self.conv1(content_feats[0]).flatten(2)
        gram_pre_2 = self.conv2(content_feats[1]).flatten(2)
        gram_pre_3 = self.conv3(content_feats[2]).flatten(2)
        gram_pre_4 = self.conv4(content_feats[3]).flatten(2)
        gram_pre_5 = self.conv5(content_feats[4]).flatten(2)
        gram_pre_1_T = gram_pre_1.transpose(1, 2)
        gram_pre_2_T = gram_pre_2.transpose(1, 2)
        gram_pre_3_T = gram_pre_3.transpose(1, 2)
        gram_pre_4_T = gram_pre_4.transpose(1, 2)
        gram_pre_5_T = gram_pre_5.transpose(1, 2)
        gram_1 = gram_pre_1.bmm(gram_pre_1_T).reshape(-1,256)
        gram_2 = gram_pre_2.bmm(gram_pre_2_T).reshape(-1,256)
        gram_3 = gram_pre_3.bmm(gram_pre_3_T).reshape(-1,256)
        gram_4 = gram_pre_4.bmm(gram_pre_4_T).reshape(-1,256)
        gram_5 = gram_pre_5.bmm(gram_pre_5_T).reshape(-1,256)
        gram_content_1 = torch.nn.functional.normalize(gram_1, p=2, dim=1)
        gram_content_2 = torch.nn.functional.normalize(gram_2, p=2, dim=1)
        gram_content_3 = torch.nn.functional.normalize(gram_3, p=2, dim=1)
        gram_content_4 = torch.nn.functional.normalize(gram_4, p=2, dim=1)
        gram_content_5 = torch.nn.functional.normalize(gram_5, p=2, dim=1)
        gram_content = torch.cat([gram_content_1,gram_content_2,gram_content_3,gram_content_4,gram_content_5],dim=0)
        return gram_content

    def cal_homo_loss(self,content_feats,style_feats):
        content_SV = self.cal_SV(content_feats)
        style_SV = self.cal_SV(style_feats)

        gram_loss = self.distance_l2(content_SV,style_SV)

        return gram_loss.sum()
    
    def Triplet_loss(self, anchor, pos, neg, rel, margin1 = 0.2, margin2 = 0.1):
        dis_ap = self.distance_l2(anchor,pos)
        dis_ar = 0.5 * self.distance_l2(anchor,rel)
        dis_an = 0.2 * self.distance_l2(anchor,neg)
        loss_1 = dis_ap - dis_ar + margin1
        loss_2 = dis_ar - dis_an + margin2
        zeros = torch.zeros(loss_1.shape).cuda()
        loss_1 = torch.maximum(loss_1,zeros)
        loss_2 = torch.maximum(loss_2,zeros)
        loss_1 = loss_1.sum()
        loss_2 = loss_2.sum()
        return loss_1 + loss_2


    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor, style_label, style_utterance_tensor,anchor_PE,\
                pos_images, pos_label, pos_utterance_tensor, pos_PE, neg_images, neg_label, neg_utterance_tensor, neg_PE, rel_images, rel_label, rel_utterance_tensor, rel_PE):
        """
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s) 

        content_feats = self.encode_with_intermediate(samples_c.tensors)  # extract content relu1_1-4_1
        a_style_feats = self.encode_with_intermediate(samples_s.tensors)  # extract style relu1_1-4_1

        style_encoder = style_utterance_tensor
        style_encoder = torch.cat([style_encoder,anchor_PE],dim=2)
        style_encoder = style_encoder.permute(0,2,1)
        style_encoder = style_encoder.unsqueeze(3)
        a_style = self.conv_1x1_encoder(style_encoder)

        content = self.embedding(samples_c.tensors)

        a_style_embedding = self.embedding(samples_s.tensors)
        pos_s = None
        pos_c = None
        mask = None
        a_hs = self.transformer(a_style, mask, content, pos_c, pos_s) 
        a_Ics = self.decode(a_hs)
        a_Ics_feats = self.encode_with_intermediate(a_Ics)
        a_loss_c = self.calc_content_loss(normal(a_Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(a_Ics_feats[-2]), normal(content_feats[-2]))
        a_loss_s = self.calc_style_loss(a_Ics_feats[0], a_style_feats[0])
        for i in range(1, 5):
            a_loss_s += self.calc_style_loss(a_Ics_feats[i], a_style_feats[i])
        a_Iss = self.decode(self.transformer(a_style, mask , a_style_embedding, pos_s, pos_s))      
        a_loss_lambda1 = self.calc_content_loss(a_Iss, samples_s.tensors)
        a_Iss_feats=self.encode_with_intermediate(a_Iss)
        a_loss_lambda2 = self.calc_content_loss(a_Iss_feats[0], a_style_feats[0])
        for i in range(1, 5):
            a_loss_lambda2 += self.calc_content_loss(a_Iss_feats[i], a_style_feats[i])
        a_loss_tv = self.TV_loss(a_Ics)
        std_s = torch.tensor(s_std).view(1, -1, 1, 1).cuda()
        mean_s = torch.tensor(s_mean).view(1, -1, 1, 1).cuda()
        a_Ics_denorm = a_Ics * std_s + mean_s
        a_style_denorm = samples_s.tensors * std_s + mean_s
        a_Ics_feats_denorm = self.encode_with_intermediate(a_Ics_denorm)
        a_style_feats_denorm = self.encode_with_intermediate(a_style_denorm)
        a_loss_homo = self.cal_homo_loss(a_Ics_feats_denorm,a_style_feats_denorm)
        a_loss_label = self.label_model(a_Ics_denorm,style_label)

        ins = style_utterance_tensor.detach()
        G_thing = [a_Ics_feats,ins]
        D_thing = [a_style_feats,a_Ics_feats,ins]


        p_style_feats = self.encode_with_intermediate(pos_images)
        style_encoder = pos_utterance_tensor
        style_encoder = torch.cat([style_encoder,pos_PE],dim=2)
        style_encoder = style_encoder.permute(0,2,1)
        style_encoder = style_encoder.unsqueeze(3)
        p_style = self.conv_1x1_encoder(style_encoder)
        p_style_embedding = self.embedding(pos_images)
        pos_s = None
        pos_c = None
        mask = None
        p_hs = self.transformer(p_style, mask, content, pos_c, pos_s) 
        p_Ics = self.decode(p_hs)
        p_Ics_feats = self.encode_with_intermediate(p_Ics)
        p_loss_c = self.calc_content_loss(normal(p_Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(p_Ics_feats[-2]), normal(content_feats[-2]))  # [1]
        p_loss_s = self.calc_style_loss(p_Ics_feats[0], p_style_feats[0])
        for i in range(1, 5):
            p_loss_s += self.calc_style_loss(p_Ics_feats[i], p_style_feats[i])
        p_Iss = self.decode(self.transformer(p_style, mask , p_style_embedding, pos_s, pos_s))       
        p_loss_lambda1 = self.calc_content_loss(p_Iss, pos_images)
        p_Iss_feats=self.encode_with_intermediate(p_Iss)
        p_loss_lambda2 = self.calc_content_loss(p_Iss_feats[0], p_style_feats[0])
        for i in range(1, 5):
            p_loss_lambda2 += self.calc_content_loss(p_Iss_feats[i], p_style_feats[i])
        p_loss_tv = self.TV_loss(p_Ics)
        p_Ics_denorm = p_Ics * std_s + mean_s
        p_style_denorm = pos_images * std_s + mean_s
        p_Ics_feats_denorm = self.encode_with_intermediate(p_Ics_denorm)
        p_style_feats_denorm = self.encode_with_intermediate(p_style_denorm)
        p_loss_homo = self.cal_homo_loss(p_Ics_feats_denorm,p_style_feats_denorm)
        p_loss_label = self.label_model(p_Ics_denorm,pos_label)


        n_style_feats = self.encode_with_intermediate(neg_images)
        style_encoder = neg_utterance_tensor
        style_encoder = torch.cat([style_encoder,neg_PE],dim=2)
        style_encoder = style_encoder.permute(0,2,1)
        style_encoder = style_encoder.unsqueeze(3)
        n_style = self.conv_1x1_encoder(style_encoder)
        n_style_embedding = self.embedding(neg_images)
        pos_s = None
        pos_c = None
        mask = None
        n_hs = self.transformer(n_style, mask, content, pos_c, pos_s) 
        n_Ics = self.decode(n_hs)
        n_Ics_feats = self.encode_with_intermediate(n_Ics)
        n_loss_c = self.calc_content_loss(normal(n_Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(n_Ics_feats[-2]), normal(content_feats[-2]))
        n_loss_s = self.calc_style_loss(n_Ics_feats[0], n_style_feats[0])
        for i in range(1, 5):
            n_loss_s += self.calc_style_loss(n_Ics_feats[i], n_style_feats[i])
        n_Iss = self.decode(self.transformer(n_style, mask , n_style_embedding, pos_s, pos_s))     
        n_loss_lambda1 = self.calc_content_loss(n_Iss, neg_images)
        n_Iss_feats=self.encode_with_intermediate(n_Iss)
        n_loss_lambda2 = self.calc_content_loss(n_Iss_feats[0], n_style_feats[0])
        for i in range(1, 5):
            n_loss_lambda2 += self.calc_content_loss(n_Iss_feats[i], n_style_feats[i])
        n_loss_tv = self.TV_loss(n_Ics)
        n_Ics_denorm = n_Ics * std_s + mean_s
        n_style_denorm = neg_images * std_s + mean_s
        n_Ics_feats_denorm = self.encode_with_intermediate(n_Ics_denorm)
        n_style_feats_denorm = self.encode_with_intermediate(n_style_denorm)
        n_loss_homo = self.cal_homo_loss(n_Ics_feats_denorm,n_style_feats_denorm)
        n_loss_label = self.label_model(n_Ics_denorm,neg_label)

        r_style_feats = self.encode_with_intermediate(rel_images)
        style_encoder = rel_utterance_tensor
        style_encoder = torch.cat([style_encoder,rel_PE],dim=2)
        style_encoder = style_encoder.permute(0,2,1)
        style_encoder = style_encoder.unsqueeze(3)
        r_style = self.conv_1x1_encoder(style_encoder)
        r_style_embedding = self.embedding(rel_images)
        pos_s = None
        pos_c = None
        mask = None
        r_hs = self.transformer(r_style, mask, content, pos_c, pos_s) 
        r_Ics = self.decode(r_hs)
        r_Ics_feats = self.encode_with_intermediate(r_Ics)
        r_loss_c = self.calc_content_loss(normal(r_Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(r_Ics_feats[-2]), normal(content_feats[-2]))
        r_loss_s = self.calc_style_loss(r_Ics_feats[0], r_style_feats[0])
        for i in range(1, 5):
            r_loss_s += self.calc_style_loss(r_Ics_feats[i], r_style_feats[i])
        r_Iss = self.decode(self.transformer(r_style, mask , r_style_embedding, pos_s, pos_s))       
        r_loss_lambda1 = self.calc_content_loss(r_Iss, rel_images)
        r_Iss_feats=self.encode_with_intermediate(r_Iss)
        r_loss_lambda2 = self.calc_content_loss(r_Iss_feats[0], r_style_feats[0])
        for i in range(1, 5):
            r_loss_lambda2 += self.calc_content_loss(r_Iss_feats[i], r_style_feats[i])
        r_loss_tv = self.TV_loss(r_Ics)
        r_Ics_denorm = r_Ics * std_s + mean_s
        r_style_denorm = rel_images * std_s + mean_s
        r_Ics_feats_denorm = self.encode_with_intermediate(r_Ics_denorm)
        r_style_feats_denorm = self.encode_with_intermediate(r_style_denorm)
        r_loss_homo = self.cal_homo_loss(r_Ics_feats_denorm,r_style_feats_denorm)
        r_loss_label = self.label_model(r_Ics_denorm,rel_label)


        anchor_SV = self.cal_SV(a_Ics_feats_denorm)
        pos_SV = self.cal_SV(p_Ics_feats_denorm)
        neg_SV = self.cal_SV(n_Ics_feats_denorm)
        rel_SV = self.cal_SV(r_Ics_feats_denorm)
        triplet_loss = self.Triplet_loss(anchor_SV,pos_SV,neg_SV,rel_SV,margin1=0.02,margin2=0.01)

        loss_c = a_loss_c + p_loss_c + n_loss_c + r_loss_c
        loss_s = a_loss_s + p_loss_s + n_loss_s + r_loss_s
        loss_lambda1 = a_loss_lambda1 + p_loss_lambda1 + n_loss_lambda1 + r_loss_lambda1
        loss_lambda2 = a_loss_lambda2 + p_loss_lambda2 + n_loss_lambda2 + r_loss_lambda2
        loss_tv = a_loss_tv + p_loss_tv + n_loss_tv + r_loss_tv
        loss_homo = 20 * (a_loss_homo + p_loss_homo + n_loss_homo + r_loss_homo) + triplet_loss
        loss_label = a_loss_label + p_loss_label + n_loss_label + r_loss_label


        return a_Ics_denorm, loss_c, loss_s, loss_lambda1, loss_lambda2, loss_tv, loss_homo, loss_label, G_thing, D_thing



class AIFormer_test(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,encoder,decoder,PatchEmbed, transformer, conv, device, args):

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed
        self.device = device
        self.conv_1x1_encoder = conv
        self.args = args


    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


    def forward(self, samples_c: NestedTensor, utterance_tensor,img_PE):
        """
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 


        style_encoder = utterance_tensor
        style_encoder = torch.cat([style_encoder,img_PE],dim=2)
        style_encoder = style_encoder.permute(0,2,1)
        style_encoder = style_encoder.unsqueeze(3)
        style = self.conv_1x1_encoder(style_encoder) 

        # ##Linear projection
        content = self.embedding(samples_c.tensors)
        
        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None
        mask = None
        
        hs = self.transformer(style, mask, content, pos_c, pos_s) 
        Ics = self.decode(hs)

        return Ics