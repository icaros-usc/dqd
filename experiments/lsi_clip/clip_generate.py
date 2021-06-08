import os
import argparse

import torch
import torchvision
import clip
import numpy as np
from PIL import Image

from stylegan_models import g_all, g_synthesis, g_mapping
from opt import AdamOpt

torch.manual_seed(20)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_path',
    type=str,
    default='./generations',
    help='',
)
parser.add_argument(
    '--ref_img_path',
    type=str,
    default=None,
    help='',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Batch Size',
)
parser.add_argument(
    '--prompt',
    type=str,
    default='Elon Musk with short hair.',
    help='',
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='',
)
parser.add_argument(
    '--img_save_freq',
    type=int,
    default=5,
    help='',
)

args = parser.parse_args()

output_path = args.output_path
batch_size = args.batch_size
prompt = args.prompt
lr = args.lr
img_save_freq = args.img_save_freq
ref_img_path = args.ref_img_path

f_prompt = prompt.replace(' ', '')
output_dir = os.path.join(output_path, f'{f_prompt}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

g_synthesis.eval()
g_synthesis.to(device)
for p in g_synthesis.parameters():
    p.requires_grad_(False)
for p in clip_model.parameters():
    p.requires_grad_(False)

latent_shape = (batch_size, 1, 512)

latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
latents = torch.nn.Parameter(latents_init, requires_grad=True)

theta_init = latents.cpu().detach().numpy()
opt = AdamOpt(theta_init, lr, betas=(0.9, 0.999))

def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img

def compute_clip_loss(img, text):
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)

    img_logits, _text_logits = clip_model(img, tokenized_text)

    return 1/img_logits * 100

counter = 0
while True:
    latents.data = torch.Tensor(opt.theta).to(device)
    dlatents = latents.repeat(1,18,1)
    img = g_synthesis(dlatents)
    
    loss = compute_clip_loss(img, args.prompt)
    
    if latents.grad is not None:
        latents.grad.zero_()
    loss.backward()
    
    np_grad = latents.grad.cpu().detach().numpy()
    opt.step(np_grad)

    if counter % args.img_save_freq == 0:
        img = tensor_to_pil_img(img)
        img.save(os.path.join(output_dir, f'{counter}.png'))

        print(f'Step {counter}')
        print(f'Loss {loss.data.cpu().numpy()[0][0]}')

    counter += 1
