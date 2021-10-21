import PIL
import torch
import streamlit as st
import torch.nn.functional as F
from io import BytesIO
from tqdm import trange
from stqdm import stqdm
from IPython.display import display
from torchvision.transforms import *

from src.utils import frames_to_gif, create_display_cols
from src.settings import DEVICE


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


def fit(model, t, y, size, steps=1000, ncut=8, max_sz=224, min_sz=32,
        starting_learning_rate=1e-2, epochs=1, loss_lerp=0.6,
        use_weighted_ratios=True, streamlit=False, save_every=None, show_every=500, ):
    z2 = F.interpolate(t, (size, size), mode='bicubic')
    t = z2.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([t], lr=starting_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    saved_frames = []
    if streamlit:
        display_cols = create_display_cols(show_every, steps * epochs)

    epoch_iterator = stqdm(range(epochs), desc=f'Epoch') if streamlit else trange(epochs)
    for e in epoch_iterator:
        if streamlit:
            iterator = stqdm(range(steps), desc=f'Resolution: {size}')
        else:
            iterator = trange(steps)

        for i in iterator:
            opt.zero_grad()

            ratios = torch.cat([torch.ones(1), torch.rand(ncut)]).cuda()

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
            if i % show_every == 0:
                show_img(t, streamlit, display_cols.pop(0) if streamlit else None)
            if save_every and i % save_every == 0:
                byte = BytesIO()
                saved_frames.append(byte)
                PIL.Image.fromarray(
                    (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy(),
                    'RGB'
                ).save(byte, format='GIF')
        scheduler.step()

    show_img(t, streamlit)
    if not save_every:
        return t, None
    return t, frames_to_gif(saved_frames)


criterion = torch.nn.MSELoss()
f = Compose([Resize(224),
             Lambda(lambda x: torch.clamp((x + 1) / 2, 0, 1)),
             RandomGrayscale(p=.2),
             Lambda(lambda x: x + torch.randn_like(x) * 0.01)])
