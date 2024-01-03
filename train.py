import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile,ImageDraw,ImageFont
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.AIFormer as AIF_model
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random
from transformers import BertTokenizer, BertModel
from transformers import logging
from collections import defaultdict
import re

# norm
c_mean = (0.5,0.5,0.5)
c_std = (0.5,0.5,0.5)
s_mean = (0.52, 0.465, 0.40)
s_std = (0.22, 0.21,0.19)

# build emotion_dictionary
emotion_dict = {
    "amusement":torch.tensor([1.,0,0,0,0,0,0,0]),
    "contentment":torch.tensor([0,1.,0,0,0,0,0,0]),
    "awe":torch.tensor([0,0,1.,0,0,0,0,0]),
    "excitement":torch.tensor([0,0,0,1.,0,0,0,0]),
    "fear":torch.tensor([0,0,0,0,1.,0,0,0]),
    "sadness":torch.tensor([0,0,0,0,0,1.,0,0]),
    "disgust":torch.tensor([0,0,0,0,0,0,1.,0]),
    "anger":torch.tensor([0,0,0,0,0,0,0,1.])
    }

# define seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# read AIFormer dataset
def build_label_utterance(csv_dir):
    df = pd.read_csv(csv_dir, encoding='utf-8')
    label_utterance_dict = defaultdict(list)
    data = df.values
    for i in range(len(data)):
        key = '/' + data[i][0] + '/' + data[i][1] + '.jpg' 
        value = [data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7],data[i][9],data[i][10],data[i][11],data[i][13],data[i][14],data[i][15]]
        label_utterance_dict[key].append(value)
    return label_utterance_dict

# build position embedding
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

# read VAD dictionary
def build_VAD_word(csv_dir):
    df = pd.read_csv(csv_dir, encoding='utf-8')
    VAD_dict = defaultdict(list)
    data = df.values
    for i in range(len(data)):
        key = data[i][0]
        value = [1000*data[i][1],1000*data[i][2],1000*data[i][3]]
        VAD_dict.update({key:value})
    return VAD_dict

# build position embedding
def make_PE(style_utterance,batch_size):
    anchor_PE = torch.empty((0,40,192))
    for v in range(batch_size):
        text = style_utterance[v]
        words = text.split(' ')
        sentence_PE = torch.empty((0,64*3))
        if len(words) < 40:
            for word_num in range(len(words)):
                word_PE = torch.tensor([])
                if VAD_word_dict.__contains__(words[word_num].lower()) == True:
                    word_VAD = VAD_word_dict[words[word_num].lower()]
                    for vad_num in range(3):
                        position = int(word_VAD[vad_num])
                        word_PE_one = PE_sheet[0][position]
                        word_PE = torch.cat([word_PE,word_PE_one],dim=0)
                    word_PE = word_PE.unsqueeze(0)
                    sentence_PE = torch.cat([sentence_PE,word_PE],dim=0)
                else:
                    for vad_num in range(3):
                        position = 500
                        word_PE_one = PE_sheet[0][position]
                        word_PE = torch.cat([word_PE,word_PE_one],dim=0)
                    word_PE = word_PE.unsqueeze(0)
                    sentence_PE = torch.cat([sentence_PE,word_PE],dim=0)
            for word_rest in range(40-len(words)):
                word_PE = torch.tensor([])
                for vad_num in range(3):
                    position = 500
                    word_PE_one = PE_sheet[0][position]
                    word_PE = torch.cat([word_PE,word_PE_one],dim=0)
                word_PE = word_PE.unsqueeze(0)
                sentence_PE = torch.cat([sentence_PE,word_PE],dim=0)
        else:
             for word_num in range(40):
                word_PE = torch.tensor([])
                if VAD_word_dict.__contains__(words[word_num].lower()) == True:
                    word_VAD = VAD_word_dict[words[word_num].lower()]
                    for vad_num in range(3):
                        position = int(word_VAD[vad_num])
                        word_PE_one = PE_sheet[0][position]
                        word_PE = torch.cat([word_PE,word_PE_one],dim=0)
                    word_PE = word_PE.unsqueeze(0)
                    sentence_PE = torch.cat([sentence_PE,word_PE],dim=0)
                else:
                    for vad_num in range(3):
                        position = 500
                        word_PE_one = PE_sheet[0][position]
                        word_PE = torch.cat([word_PE,word_PE_one],dim=0)
                    word_PE = word_PE.unsqueeze(0)
                    sentence_PE = torch.cat([sentence_PE,word_PE],dim=0)
        sentence_PE = sentence_PE.unsqueeze(0)
        anchor_PE = torch.cat([anchor_PE,sentence_PE],dim=0)
    return anchor_PE

# resize content picture
def content_transform():
    c_mean = (0.5,0.5,0.5)
    c_std = (0.5,0.5,0.5)
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=c_mean, std=c_std)
    ]
    return transforms.Compose(transform_list)
    
# resize style picture
def style_transform():
    s_mean = (0.52, 0.465, 0.40)
    s_std = (0.22, 0.21,0.19)
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=s_mean, std=s_std)
    ]
    return transforms.Compose(transform_list)


# build content dataset
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        try:
            path = self.paths[index]
            img = Image.open(str(path)).convert('RGB')
            image = self.transform(img)
        except:
            path = self.paths[0]
            img = Image.open(str(path)).convert('RGB')
            image = self.transform(img)
        return image
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

# build style dataset
class FlatFolderDataset_withnum(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset_withnum, self).__init__()
        self.root = root
        print(self.root)
        self.paths = list(Path(self.root).glob('**/*.*'))
        self.transform = transform

    def __getitem__(self, index):
        try:
            path = self.paths[index]
            img = Image.open(str(path)).convert('RGB')
            image = self.transform(img)
            path = str(path)[len(str(args.style_dir)):]
            pic_path = path
            utterance_num = random.randint(0,len(label_utterance_dict[path])-1)
            label = label_utterance_dict[path][utterance_num][0]
            labels_tensor = emotion_dict[label]
            utterance = label_utterance_dict[path][utterance_num][1]
            utterances_list = utterance
            key_word = label_utterance_dict[path][utterance_num][2]
            
            pos_path = label_utterance_dict[path][utterance_num][3]
            pos_path = args.style_dir + pos_path
            pos_img = Image.open(str(pos_path)).convert('RGB')
            pos_image = self.transform(pos_img)
            pos_label = label_utterance_dict[path][utterance_num][4]
            pos_label_tensor = emotion_dict[pos_label]
            pos_utterance = label_utterance_dict[path][utterance_num][5]

            neg_path = label_utterance_dict[path][utterance_num][6]
            neg_path = args.style_dir + neg_path
            neg_img = Image.open(str(neg_path)).convert('RGB')
            neg_image = self.transform(neg_img)
            neg_label = label_utterance_dict[path][utterance_num][7]
            neg_label_tensor = emotion_dict[neg_label]
            neg_utterance = label_utterance_dict[path][utterance_num][8]

            rel_path = label_utterance_dict[path][utterance_num][9]
            rel_path = args.style_dir + rel_path
            rel_img = Image.open(str(rel_path)).convert('RGB')
            rel_image = self.transform(rel_img)
            rel_label = label_utterance_dict[path][utterance_num][10]
            rel_label_tensor = emotion_dict[rel_label]
            rel_utterance = label_utterance_dict[path][utterance_num][11]

        except:
            path = self.paths[1]
            img = Image.open(str(path)).convert('RGB')
            image = self.transform(img)
            path = str(path)[len(str(args.style_dir)):]
            pic_path = path
            utterance_num = random.randint(0,len(label_utterance_dict[path])-1)
            label = label_utterance_dict[path][utterance_num][0]
            labels_tensor = emotion_dict[label]
            utterance = label_utterance_dict[path][utterance_num][1]
            utterances_list = utterance
            key_word = label_utterance_dict[path][utterance_num][2]

            pos_path = label_utterance_dict[path][utterance_num][3]
            pos_path = args.style_dir + pos_path
            pos_img = Image.open(str(pos_path)).convert('RGB')
            pos_image = self.transform(pos_img)
            pos_label = label_utterance_dict[path][utterance_num][4]
            pos_label_tensor = emotion_dict[pos_label]
            pos_utterance = label_utterance_dict[path][utterance_num][5]

            neg_path = label_utterance_dict[path][utterance_num][6]
            neg_path = args.style_dir + neg_path
            neg_img = Image.open(str(neg_path)).convert('RGB')
            neg_image = self.transform(neg_img)
            neg_label = label_utterance_dict[path][utterance_num][7]
            neg_label_tensor = emotion_dict[neg_label]
            neg_utterance = label_utterance_dict[path][utterance_num][8]

            rel_path = label_utterance_dict[path][utterance_num][9]
            rel_path = args.style_dir + rel_path
            rel_img = Image.open(str(rel_path)).convert('RGB')
            rel_image = self.transform(rel_img)
            rel_label = label_utterance_dict[path][utterance_num][10]
            rel_label_tensor = emotion_dict[rel_label]
            rel_utterance = label_utterance_dict[path][utterance_num][11]

        return image, labels_tensor, utterances_list, pic_path, key_word, pos_image, pos_label_tensor, pos_utterance, neg_image, neg_label_tensor, neg_utterance, rel_image, rel_label_tensor, rel_utterance  
        
    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

# adjust learning rate1
def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# adjust learning rate2
def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--affective_ArtEmis', default='', type=str,
                    help='Directory path to name and gener of style images ')
parser.add_argument('--VAD_csv', default='', type=str,
                    help='Directory path to name and gener of style images ')
parser.add_argument('--vgg', type=str, default='')  #run the train.py, please download the pretrained vgg checkpoint
parser.add_argument('--embedding', type=str, default=None)  #run the train.py, please download the pretrained embedding checkpoint
parser.add_argument('--Trans', type=str, default=None)  #run the train.py, please download the pretrained Trans checkpoint
parser.add_argument('--decoder', type=str, default=None)  #run the train.py, please download the pretrained decoder checkpoint
parser.add_argument('--VAD_emb', type=str, default=None)  #run the train.py, please download the pretrained VAD_emb checkpoint
parser.add_argument('--D', type=str, default=None)
parser.add_argument('--SV_1', type=str, default='')
parser.add_argument('--SV_2', type=str, default='')
parser.add_argument('--SV_3', type=str, default='')
parser.add_argument('--SV_4', type=str, default='')
parser.add_argument('--SV_5', type=str, default='')
parser.add_argument('--label', type=str, default='')
parser.add_argument('--save_dir', default='',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='',
                    help='Directory to save the log')

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--show_pic', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--device_ids', type=list, default=[0,1])
parser.add_argument('--style_weight', type=float, default=0.3)
parser.add_argument('--content_weight', type=float, default=5)
parser.add_argument('--l_identity1', type=float, default=2.0)
parser.add_argument('--l_identity2', type=float, default=0.01)
parser.add_argument('--tv_loss', type=float, default=1)
parser.add_argument('--homo_loss', type=float, default=30.0)
parser.add_argument('--label_loss', type=float, default=140.0)
parser.add_argument('--G_loss', type=float, default=3.0)
parser.add_argument('--D_loss', type=float, default=3.0)
parser.add_argument('--seed', type=int, default=922)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()


setup_seed(args.seed)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")

logging.set_verbosity_error()
BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
BERT_model.to(device)

vgg = AIF_model.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = AIF_model.decoder
if args.decoder is not None:
    print('decoder loaded')
    decoder.load_state_dict(torch.load(args.decoder))
embedding = AIF_model.PatchEmbed()
if args.decoder is not None:
    print('embedding loaded')
    embedding.load_state_dict(torch.load(args.embedding))
Trans = transformer.Transformer()
if args.Trans is not None:
    print('Trans loaded')
    Trans.load_state_dict(torch.load(args.Trans))
VAD_emb = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=960,out_channels=512)
if args.VAD_emb is not None:
    print('VAD_emb loaded')
    VAD_emb.load_state_dict(torch.load(args.VAD_emb))
D = AIF_model.Discriminator()
if args.D is not None:
    print('D loaded')
    D.load_state_dict(torch.load(args.D))
D.to(device)

emo_1 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=64,out_channels=16)
emo_2 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=128,out_channels=16)
emo_3 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=256,out_channels=16)
emo_4 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=16)
emo_5 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=16)
emo_1.load_state_dict(torch.load(args.SV_1))
emo_2.load_state_dict(torch.load(args.SV_2))
emo_3.load_state_dict(torch.load(args.SV_3))
emo_4.load_state_dict(torch.load(args.SV_4))
emo_5.load_state_dict(torch.load(args.SV_5))

label_model = AIF_model.classify_label(vgg)
label_model.load_state_dict(torch.load(args.label))

label_utterance_dict = build_label_utterance(args.affective_ArtEmis)
VAD_word_dict = build_VAD_word(args.VAD_csv)
PE_sheet = get_sinusoid_encoding_table(1001,64)


with torch.no_grad():
    network = AIF_model.AIFTrans(vgg, decoder, embedding, Trans, VAD_emb, device, emo_1, emo_2, emo_3, emo_4, emo_5, label_model)
network.train()
network.to(device)

# DP
network = nn.DataParallel(network, device_ids=args.device_ids)
c_tf = content_transform()
s_tf = style_transform()

# read content dataset
content_dataset = FlatFolderDataset(args.content_dir, c_tf)
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

# read style dataset
style_dataset = FlatFolderDataset_withnum(args.style_dir, s_tf)
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# define Adam optimizer
optimizer = torch.optim.Adam([
                              {'params': network.module.transformer.parameters()},
                              {'params': network.module.decode.parameters()},
                              {'params': network.module.embedding.parameters()},
                              {'params': network.module.conv_1x1_encoder.parameters()}
                              ], lr=args.lr)
optim_D = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.9, 0.98))

# training begin
for i in tqdm(range(args.max_iter)):
    # define learning rate
    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)
    print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))

    content_images = next(content_iter).to(device)
    style_all = next(style_iter) 

    style_images = style_all[0].to(device)
    style_label = style_all[1].to(device)
    style_utterance = style_all[2]
    style_path = style_all[3]
    key_word = style_all[4]
    pos_images = style_all[5].to(device)
    pos_label = style_all[6].to(device)
    pos_utterance = style_all[7]
    neg_images = style_all[8].to(device)
    neg_label = style_all[9].to(device)
    neg_utterance = style_all[10]
    rel_images = style_all[11].to(device)
    rel_label = style_all[12].to(device)
    rel_utterance = style_all[13]
    anchor_PE = make_PE(style_utterance,args.batch_size)
    pos_PE = make_PE(pos_utterance,args.batch_size)
    neg_PE = make_PE(neg_utterance,args.batch_size)
    rel_PE = make_PE(rel_utterance,args.batch_size)
    style_utterance_tensor = torch.tensor([]).cuda()
    pos_utterance_tensor = torch.tensor([]).cuda()
    neg_utterance_tensor = torch.tensor([]).cuda()
    rel_utterance_tensor = torch.tensor([]).cuda()
    with torch.no_grad():
        for m in range(args.batch_size):
            text = style_utterance[m]
            encoded_input = BERT_tokenizer(text, add_special_tokens = True, max_length = 40, pad_to_max_length = True, return_tensors='pt').to(device)
            output = BERT_model(**encoded_input)
            genre_of_style_encode = output[0]
            style_utterance_tensor = torch.cat((style_utterance_tensor, genre_of_style_encode), dim=0)
            text = pos_utterance[m]
            encoded_input = BERT_tokenizer(text, add_special_tokens = True, max_length = 40, pad_to_max_length = True, return_tensors='pt').to(device)
            output = BERT_model(**encoded_input)
            genre_of_style_encode = output[0]
            pos_utterance_tensor = torch.cat((pos_utterance_tensor, genre_of_style_encode), dim=0)
            text = neg_utterance[m]
            encoded_input = BERT_tokenizer(text, add_special_tokens = True, max_length = 40, pad_to_max_length = True, return_tensors='pt').to(device)
            output = BERT_model(**encoded_input)
            genre_of_style_encode = output[0]
            neg_utterance_tensor = torch.cat((neg_utterance_tensor, genre_of_style_encode), dim=0)
            text = rel_utterance[m]
            encoded_input = BERT_tokenizer(text, add_special_tokens = True, max_length = 40, pad_to_max_length = True, return_tensors='pt').to(device)
            output = BERT_model(**encoded_input)
            genre_of_style_encode = output[0]
            rel_utterance_tensor = torch.cat((rel_utterance_tensor, genre_of_style_encode), dim=0)

    out, loss_c, loss_s, l_identity1, l_identity2, loss_tv, loss_homo, loss_label, G_things, D_things = network(content_images, style_images, style_label, style_utterance_tensor, anchor_PE, \
                    pos_images, pos_label, pos_utterance_tensor, pos_PE, neg_images, neg_label, neg_utterance_tensor, neg_PE, rel_images, rel_label, rel_utterance_tensor, rel_PE)

    optim_D.zero_grad()

    style_feats = D_things[0]
    Ics_feats = D_things[1]
    for style_i in range(len(style_feats)):
        style_feats[style_i] = style_feats[style_i].detach()
    for Ics_i in range(len(Ics_feats)):
        Ics_feats[Ics_i] = Ics_feats[Ics_i].detach()
    ins = D_things[2].detach()
    loss_D = D(style_feats,ins,True) + D(Ics_feats,ins,False)
    loss_D = args.D_loss * loss_D
    print('D',loss_D.sum().cpu().detach().numpy())

    loss_D.sum().backward()
    optim_D.step()
    Ics_feats_G = G_things[0]
    ins_G = G_things[1]
    loss_G = D(Ics_feats_G,ins_G,True)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_psd = args.G_loss * loss_G
    l_identity1 = l_identity1 * args.l_identity1
    l_identity2 = l_identity2 * args.l_identity2
    loss_tv = args.tv_loss * loss_tv
    loss_homo = args.homo_loss * loss_homo
    loss_label = args.label_loss * loss_label
    loss = loss_c + loss_s + loss_psd + l_identity1 + l_identity2 + loss_tv + loss_homo + loss_label
    print("-G",loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy(),"-psd:",loss_psd.sum().cpu().detach().numpy()
              ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy(),"-TV:",loss_tv.sum().cpu().detach().numpy()
              ,"-loss_homo:",loss_homo.sum().cpu().detach().numpy(),"-loss_label:",loss_label.sum().cpu().detach().numpy()
              )
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()


    if i % args.show_pic == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(i),".jpg"
                    )
        white_images = torch.ones(min(4,args.batch_size),3,256,256).to(device)
        white_images = white_images * 255

        std_c = torch.tensor(c_std).view(1, -1, 1, 1).cuda()
        mean_c = torch.tensor(c_mean).view(1, -1, 1, 1).cuda()
        content_images = content_images * std_c + mean_c
        content_images = content_images[:min(4,args.batch_size), :, :, :]
        out = out[:min(4,args.batch_size), :, :, :]
        out = torch.cat((content_images,out),0)
        out = torch.cat((white_images,out),0)
        save_image(out, output_name)

        img = Image.open(output_name)
        draw = ImageDraw.Draw(img)
        typeface = ImageFont.truetype("Arial.ttf", 18)
        for i in range(min(4,args.batch_size)):
            miaoshu = style_utterance[i]
            miaoshu = re.sub(r"(.{30})", "\\1\n", miaoshu)
            draw.text((i * 258, 10), miaoshu, fill=(120, 0, 60), font=typeface)
        img.save(output_name)


    # save model
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:

        state_dict = network.module.conv_1x1_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                    '{:s}/VAD_emb_{:d}.pth'.format(args.save_dir,
                                                               i + 1))

        state_dict = network.module.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.module.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.module.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = D.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/D_{:d}.pth'.format(args.save_dir,
                                                           i + 1))



