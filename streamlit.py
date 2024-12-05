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
    :return: The colorized image.
    """
    if model is None or device is None:
        raise ValueError("Model and device must be provided.")

    with torch.no_grad():
        output = model(gray.unsqueeze(0))
        output = output.squeeze(0).cpu().numpy()
        output = np.moveaxis(output, 0, -1)
        output = np.concatenate((gray[0].cpu().numpy()[..., np.newaxis], output), axis=-1)
        output = lab2rgb(output)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return output



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    model.load_state_dict(torch.load("cnn_new_model.pth", map_location=device, weights_only=True)["model_state_dict"])

    model.eval()
    st.title("Colorization of images")
    st.write("Please upload an image, and we will resize it and colorize it!")

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=False
    )

    if uploaded_file:
        st.subheader("Original image")
        
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_cv.shape[0] > 400 or image_cv.shape[1] > 400:
            image_resized = cv2.resize(image_cv, (400, 400), interpolation = cv2.INTER_AREA) # Downscale the image
        else:
            image_resized = cv2.resize(image_cv, (400, 400), interpolation = cv2.INTER_CUBIC) # Upscale the image

        st.image(image_resized, caption=f"Original image: {uploaded_file.name}")
        cv2.imwrite("resized_image.png", image_resized)        
        gray_image = read_image("resized_image.png", ImageReadMode.GRAY).to(device).float()
        # Delete the resized image
        os.remove("resized_image.png")
        

        try:
            colorized_array = colorize_image(gray_image, model, device)
            colorized_image = Image.fromarray(colorized_array)

            st.subheader("Colorized image")
            st.image(colorized_image, caption="Colorized image")

            buffered = io.BytesIO()
            colorized_image.save(buffered, format="PNG")
            buffered.seek(0)
            st.download_button(
                label="Download colorized image",
                data=buffered,
                file_name=f"colorized_{uploaded_file.name}",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"An error occurred during colorization: {e}")

if __name__ == "__main__":
    main()
