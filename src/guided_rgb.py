import PIL
import clip
import math
import torch
import streamlit as st
import torch.nn.functional as F
from io import BytesIO
from tqdm import trange
from IPython.display import display
from torchvision.transforms import *

from src.utils import frames_to_gif
from src.settings import CLIP_MODEL_PATH, DEVICE


def get_sizes(sz, min_sz=32):
    szs = [sz]
    while True:
        if sz <= min_sz:
            return sorted(szs)

        if sz % 2 == 0:
            sz = sz // 2
            szs.append(sz)
        else:
            return sorted(szs)


def make_crop(img, ratio, max_cut=224, min_cut=0.2):
    w, h = img.shape[2:]
    min_sz = min(w, h)
    if min_cut < 1:
        min_cut = int(min_sz * min_cut)
    crop_size = int(min(max(ratio * min_sz, min_cut), max_cut))

    w_offset = int(torch.rand(1) * (w - crop_size))
    h_offset = int(torch.rand(1) * (h - crop_size))

    cropped = img[:, :, w_offset:w_offset + crop_size, h_offset:h_offset + crop_size]
    return f(cropped)


def get_crops(img, ratios, max_cut, min_cut):
    return torch.cat([make_crop(img, ratio.item(), max_cut, min_cut) for ratio in ratios])


def show_img(t, streamlit=False, display_el=None):
    img = PIL.Image.fromarray((t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy(),
                              'RGB')
    if streamlit:
        if display_el:
            with display_el:
                st.image(img)
        else:
            st.image(img)
    else:
        display(img)


def fit(model, t, y, size, steps=1000, ncut=8, max_sz=224, min_sz=32, use_weighted_ratios=True, streamlit=False, save_every=None, show_every=500):
    if streamlit:
        progress = st.progress(0.)
        img_per_col = 4
        num_imgs = math.ceil(steps / show_every)
        num_rows = math.ceil(num_imgs / img_per_col)
        display_cols = []
        for i in range(num_rows):
            display_cols.extend(st.columns(img_per_col))

    z2 = F.interpolate(t, (size, size), mode='bicubic')
    t = z2.detach().clone().requires_grad_(True)
    show_img(t)
    opt = torch.optim.Adam([t], lr=lr)
    saved_frames = []
    for i in trange(steps):
        opt.zero_grad()

        ratios = [torch.ones(1).cuda()]
        for j in range(ncut):
            ratios.append(torch.rand(1).cuda())
        ratios = torch.cat(ratios)
        crops = get_crops(t, ratios, max_sz, min_sz)
        loss_avg = 0.
        loss = 0.
        weighted_ratios = ratios / ratios.sum() if use_weighted_ratios else torch.ones_like(ratios).to(DEVICE)

        embeds = model.encode_image(crops)

        for embed, ratio in zip(embeds, weighted_ratios):
            x = F.normalize(embed, dim=-1)
            loss += torch.sqrt(criterion(x, y)) * ratio

        loss.backward()
        opt.step()
        loss_avg = loss if loss_avg == 0. else (loss_avg * loss_lerp + loss * (1 - loss_lerp))
        if i % 100 == 0:
            print(loss_avg.item())
            if streamlit:
                progress.progress(i / steps)
        if i % show_every == 0:
            show_img(t, streamlit, display_cols.pop(0))
        if save_every and i % save_every == 0:
            byte = BytesIO()
            saved_frames.append(byte)
            PIL.Image.fromarray(
                (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy(),
                'RGB'
            ).save(byte, format='GIF')

    show_img(t, streamlit)
    if not save_every:
        return t, None
    return t, frames_to_gif(saved_frames)


criterion = torch.nn.MSELoss()
f = Compose([Resize(224),
             Lambda(lambda x: torch.clamp((x + 1) / 2, 0, 1)),
             RandomGrayscale(p=.2),
             Lambda(lambda x: x + torch.randn_like(x) * 0.01)])

# set parameters and train
prompt = 'a landscape containing knights riding on the horizon by Greg Rutkowski'  # text prompt
seed = 0
torch.manual_seed(seed)

szs = get_sizes(1024, 64)
print(szs)  # getting sizes
steps = [2000] * len(szs)  # getting number of steps per size
cuts = [8, 8, 8, 16, 24]  # number of image cuts for CLIP loss per iteration
max_szs = [64] * len(szs)  # max cut size (pixels)
min_szs = [0.2] * len(szs)  # min cut size (pixel or image size ratio)

lr = 1e-2
loss_lerp = 0.6  # used for display only

# encoded_prompt = model.encode_text(clip.tokenize(prompt).to(DEVICE))
# y = F.normalize(encoded_prompt, dim=-1)

# init image, can be replaced with a photo
# z = torch.rand((1, 3, szs[0], szs[0]), device=DEVICE, requires_grad=True)

# for size, step, cut, max_sz, min_sz in zip(szs, steps, cuts, max_szs, min_szs):
#     print(size, step, cut)
#     z = fit(z, size, steps=step, ncut=cut, max_sz=max_sz, min_sz=min_sz)
