import clip
import streamlit as st
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

from src.settings import CLIP_MODEL_PATH, DEVICE


@st.cache(allow_output_mutation=True)
def get_model():
    return clip.load(str(CLIP_MODEL_PATH), jit=False)[0].eval().requires_grad_(False).to(DEVICE)


def encode_prompt(model, prompt):
    encoded_prompt = model.encode_text(clip.tokenize(prompt).to(DEVICE))
    return F.normalize(encoded_prompt, dim=-1)[0]


def image_to_jpg(image):
    return image.convert('RGB')


def frames_to_gif(byteframes, duration=5):
    gifs = [Image.open(byteframe) for byteframe in byteframes]
    temp = BytesIO()
    length_per_frame = (duration / len(gifs)) * 10
    gifs[0].save(temp, append_images=gifs[1:], save_all=True, format='GIF', optimize=False, loop=0, duration=length_per_frame)
    return temp
