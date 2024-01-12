import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.AIFormer as AIF_model
from function import normal
import numpy as np
import time
from transformers import BertTokenizer, BertModel
from transformers import logging
import pandas as pd
import torch.nn.functional as F
from collections import defaultdict

c_mean = (0.5,0.5,0.5)
c_std = (0.5,0.5,0.5)
s_mean = (0.52, 0.465, 0.40)
s_std = (0.22, 0.21,0.19)
std_s = torch.tensor(s_std).view(1, -1, 1, 1).cuda()
mean_s = torch.tensor(s_mean).view(1, -1, 1, 1).cuda()

def content_transform():

    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ]
    transform_list.append(transforms.Normalize(mean=c_mean, std=c_std))
    transform = transforms.Compose(transform_list)
    return transform


def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

def build_VAD_word(csv_dir):
    df = pd.read_csv(csv_dir, encoding='utf-8')
    VAD_dict = defaultdict(list)
    data = df.values
    for i in range(len(data)):
        key = data[i][0]
        value = [1000*data[i][1],1000*data[i][2],1000*data[i][3]]
        VAD_dict.update({key:value})
    return VAD_dict

def make_PE(style_utterance):  
    words = style_utterance.split(' ')
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
    anchor_PE = sentence_PE.unsqueeze(0)
    return anchor_PE

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='content_pic',
                    help='./test_pic/content/golden_gate.jpg')
parser.add_argument('--description_dir', type=str, default='utterance.txt',
                    help='./test_pic/style/1.txt')
parser.add_argument('--output', type=str, default='output',
                    help='./output')
# parser.add_argument('--vgg', type=str, default='')
# parser.add_argument('--decoder', type=str, default='')
# parser.add_argument('--Trans', type=str, default='')
# parser.add_argument('--embedding', type=str, default='')
# parser.add_argument('--VAD_emb', type=str, default='')
# parser.add_argument('--VAD_dic', default='', type=str,
#                     help='Directory path to name and gener of style images ')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='experiments/decoder_iter_40000.pth')
parser.add_argument('--Trans', type=str, default='experiments/transformer_iter_40000.pth')
parser.add_argument('--embedding', type=str, default='experiments/embedding_iter_40000.pth')
parser.add_argument('--VAD_emb', type=str, default='experiments/conv_40000.pth')
parser.add_argument('--VAD_dic', default='experiments/affective_ArtEmis.csv', type=str,
                    help='Directory path to name and gener of style images ')
parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()


# Advanced options
content_size=256
style_size=256
crop='store_true'
save_ext='.png'
output_path=args.output
preserve_color='store_true'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(args.content_dir)
content_dir = Path(args.content_dir)
content_paths = [f for f in content_dir.glob('*')]


if not os.path.exists(output_path):
    os.mkdir(output_path)


vgg = AIF_model.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

logging.set_verbosity_error()
# BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# BERT_model = BertModel.from_pretrained('bert-base-uncased')
BERT_tokenizer = BertTokenizer.from_pretrained('/home/shuchenweng/zpx/ICCV2023/bert-base-uncased')   # 'bert-base-uncased'
BERT_model = BertModel.from_pretrained('/home/shuchenweng/zpx/ICCV2023/bert-base-uncased')   # 'bert-base-uncased'
BERT_model.to(device)

decoder = AIF_model.decoder
Trans = transformer.Transformer()
embedding = AIF_model.PatchEmbed()

decoder.eval()
embedding.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict

new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

conv = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=960,out_channels=512)
conv.load_state_dict(torch.load(args.VAD_emb))

network = AIF_model.AIFormer_test(vgg,decoder,embedding,Trans, conv, device, args)
network.eval()
network.to(device)

VAD_word_dict = build_VAD_word(args.VAD_dic)
PE_sheet = get_sinusoid_encoding_table(1001,64)

style_dir = args.description_dir
paths = list(Path(style_dir).glob('**/*.*'))
paths.sort()

content_tf = content_transform()

file_name = args.description_dir
utterances = []
with open(file_name, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        utterances.append(line)


for content_path in content_paths:
    for utterance in utterances:
        content = content_tf(Image.open(content_path).convert("RGB"))
        h,w,c=np.shape(content)    
        img_PE = make_PE(utterance)
        img_PE = img_PE.cuda()
        text = utterance
        encoded_input = BERT_tokenizer(text, add_special_tokens = True, max_length = 40,pad_to_max_length = True,return_tensors='pt').to(device)
        output = BERT_model(**encoded_input)
        style = output[0]
        content = content.to(device).unsqueeze(0)
        
        with torch.no_grad():
            output= network(content,style,img_PE)
        output = output * std_s + mean_s
        output = output.cpu()
        output_path = args.output

        output_name = '{:s}/{:s}_{:s}{:s}'.format(
            output_path, utterance,
            splitext(basename(content_path))[0], save_ext
        )
        save_image(output, output_name)



