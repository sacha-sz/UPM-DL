import streamlit as st
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import io
import os
import cv2
import torch

from PIL import Image
from torch import cat
from skimage.color import lab2rgb
from torchvision.io import read_image, ImageReadMode

from streamlit_image_comparison import image_comparison

class Network(nn.Module):
    def __init__(self):
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)

        # Dilation layers.
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(128)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(64)
        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(32)
        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        x_1 = F.relu(self.conv1_bn(self.conv1(x)))
        x_2 = F.relu(self.conv2_bn(self.conv2(x_1)))
        x_3 = F.relu(self.conv3_bn(self.conv3(x_2)))
        x_4 = F.relu(self.conv4_bn(self.conv4(x_3)))

        # Dilation layers.
        x_5 = F.relu(self.conv5_bn(self.conv5(x_4)))
        x_5_d = F.relu(self.conv6_bn(self.conv6(x_5)))

        x_6 = F.relu(self.t_conv1_bn(self.t_conv1(x_5_d)))
        x_6 = cat((x_6, x_3), 1)
        x_7 = F.relu(self.t_conv2_bn(self.t_conv2(x_6)))
        x_7 = cat((x_7, x_2), 1)
        x_8 = F.relu(self.t_conv3_bn(self.t_conv3(x_7)))
        x_8 = cat((x_8, x_1), 1)
        x_9 = F.relu(self.t_conv4(x_8))
        x_9 = cat((x_9, x), 1)
        x = self.output(x_9)
        return x

def colorize_image(gray, model=None, device=None):
    """
    Colorizes the given grayscale image using the provided model and device.
    :param gray: The grayscale image.
    :param model: The model to use for colorization.
    :param device: The device to use for colorization.
    :return: The colorized image in rgb format
    """
    if model is None or device is None:
        raise ValueError("Model and device must be provided.")

    with torch.no_grad():
        output = model(gray.unsqueeze(0).float()).int()
        img = cat((gray, output[0]), 0)
        img = lab2rgb(img.permute(1, 2, 0).cpu().numpy())
        img = (img * 255).astype(np.uint8)
    return img

def main():
    st.set_page_config(
        page_title="Colorization of images", 
        page_icon="🎨", 
        layout="centered", 
        initial_sidebar_state="auto", 
        menu_items= {
            'Get help': "https://github.com/sacha-sz/UPM-DL/issues",
            'Report a bug': "https://github.com/sacha-sz/UPM-DL/issues",
            'About': "This application colorizes images using a convolutional neural network. The model was trained on a custom dataset of images from NASA's Astronomy Picture of the Day."
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    model.load_state_dict(torch.load("cnn_new_model.pth", map_location=device, weights_only=True)["model_state_dict"])

    model.eval()
    st.title("Colorization of images")
    st.write("Please upload one or more images, and we will resize and colorize them!")

    
    st.sidebar.title("Instructions")
    st.sidebar.write("Upload one or more images, and we will resize and colorize them using a convolutional neural network.")
    
    st.sidebar.title("Authors")
    st.sidebar.write("This application was created by [Benjamin](https://github.com/baneboll11), [Simon](https://github.com/Sim089n) and [Sacha](https://github.com/sacha-sz).")

    st.sidebar.title("Source code")
    st.sidebar.write("The source code for this application can be found on [GitHub](https://github.com/sacha-sz/UPM-DL)")

    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/c7/Chill_guy_original_artwork.jpg/220px-Chill_guy_original_artwork.jpg", caption="Chill guy repo", use_container_width=True)

    uploaded_files = st.file_uploader(
        "Choose one or more images...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.status("Colorizing images...", expanded=True) as status:
            results = []
            for uploaded_file in uploaded_files:
                try:
                    st.write(f"Processing {uploaded_file.name}...")
                    image = Image.open(uploaded_file)
                    image = np.array(image)

                    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if image_cv.shape[0] > 400 or image_cv.shape[1] > 400:
                        image_resized = cv2.resize(image_cv, (400, 400), interpolation=cv2.INTER_AREA) # Downscale
                    else:
                        image_resized = cv2.resize(image_cv, (400, 400), interpolation=cv2.INTER_CUBIC) # Upscale

                    cv2.imwrite(f"temp-{uploaded_file.name.split('.')[:-1]}.png", image_resized)
                    gray_image = read_image(f"temp-{uploaded_file.name.split('.')[:-1]}.png", ImageReadMode.GRAY).to(device) / 2.5
                    os.remove(f"temp-{uploaded_file.name.split('.')[:-1]}.png")
                    colorized_array = colorize_image(gray_image, model, device)
                    col_img = Image.fromarray(colorized_array)
                    results.append((uploaded_file.name, image_resized, col_img))

                except Exception as e:
                    st.error(f"An error occurred with {uploaded_file.name}: {e}")

            status.update(
                label="Colorization complete!", state="complete", expanded=False
            )

        filename_are_unique = len(set([file_name for file_name, _, _ in results])) == len(results)
            
        compteur = 0
        for file_name, original, colorized in results:
            if not filename_are_unique:
                compteur += 1

            st.write(f"Comparison for {file_name}:")
            image_comparison(original, np.array(colorized))

            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    buffered = io.BytesIO()
                    colorized.save(buffered, format="PNG")
                    buffered.seek(0)
                    st.download_button(
                        label="Download colorized image",
                        data=buffered,
                        file_name=f"colorized_{file_name}",
                        mime="image/png",
                        key=f"download_colorized_{file_name}" if filename_are_unique else f"download_colorized_{compteur}_{file_name}"
                    )

                with col2:
                    with st.popover("Show the original image"):
                        st.image(original, caption="Original image")

                with col3:
                    with st.popover("Show the colorized image"):
                        st.image(colorized, caption="Colorized image")
    else:
        st.markdown("You can upload a `.jpg` or `.png` file.")

if __name__ == "__main__":
    main()
