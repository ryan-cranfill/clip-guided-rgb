import torch
from PIL import Image
import streamlit as st
from torchvision.transforms import ToTensor

from src.settings import DEVICE
from src.guided_rgb import fit, get_sizes
from src.utils import get_model, encode_prompt, image_to_jpg


st.set_page_config(layout='wide')

st.header('Cool App For CLIP Things!')

model = get_model()

prompt = st.text_input('Prompt', 'Shrek is love, Shrek is life')
y = encode_prompt(model, prompt)
# seed = 0
# torch.manual_seed(seed)
uploaded_file = st.file_uploader("Upload Base Image (Optional)")
max_dim = st.number_input('Max Dimension', 64, 2048, 1024)
min_size = st.number_input('Min size', 32, 2048, 64)
default_steps = st.number_input('Default number of steps per epoch', 1, value=2000, step=100)
learning_rate = st.number_input('Starting learning Rate', 0., value=1e-2)
epochs = st.number_input('Epochs', 1, value=1)


save_frames_every = st.sidebar.number_input('Save a frame every n iter (0 for no progress video)', 0, value=25)
show_every = st.sidebar.number_input('Show progress every n interations', 1, value=500, step=50)

sizes = get_sizes(max_dim, min_size)
# TODO: Is this appropriate for cuts??
cuts = [8, 8, 8, 16, 24, 32, 40][:len(sizes)]  # number of image cuts for CLIP loss per iteration
max_szs = [64] * len(sizes)  # max cut size (pixels)
min_szs = [0.2] * len(sizes)  # min cut size (pixel or image size ratio)


steps = []
with st.expander('Steps per resolution'):
    for size in sizes:
        steps.append(st.number_input(f'Iterations for size: {size}', 1, value=default_steps, step=100))
    steps = list(map(int, steps))  # IDK why this keeps reverting to floats but here we are....

if uploaded_file is not None:
    image = image_to_jpg(Image.open(uploaded_file)).resize((max_dim, max_dim))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    z = ToTensor()(image).unsqueeze(0).to(DEVICE)
else:
    z = torch.rand((1, 3, sizes[0], sizes[0]), device=DEVICE, requires_grad=True)

if st.button('Go!'):
    st.text('Resolution Steps')
    progress = st.progress(0.)
    for i, (size, step, cut, max_sz, min_sz) in enumerate(zip(sizes, steps, cuts, max_szs, min_szs)):
        print(size, step, cut)
        progress.progress(i / len(steps))
        st.text(f'Size: {size}')
        z, gif = fit(
            model,
            z,
            y,
            size,
            steps=step,
            ncut=cut,
            max_sz=max_sz,
            min_sz=min_sz,
            starting_learning_rate=learning_rate,
            epochs=epochs,
            streamlit=True,
            save_every=save_frames_every,
            show_every=show_every
        )

        if gif:
            st.image(gif, output_format='GIF')
    st.subheader('DONE')
