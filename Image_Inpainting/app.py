import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import transforms
import numpy as np

IMG_HEIGHT = 256
IMG_WIDTH = 256
class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.concat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class II_Unet(nn.Module):
    def __init__(
            self, n_channels=3, n_classes=3, features=[64, 128, 256, 512],
    ):
        super(II_Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ConvBlock(64, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = Decoder(1024, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)

        self.out_conv = FinalOutput(64, n_classes)


    def forward(self, x):
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)
        return x
    

def proces_img(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    img = transforms.functional.to_tensor(img)
    img = img*2 - 1
    img = img.type(torch.float32)
    img = img.unsqueeze(0)
    return img

IP_unet_model = II_Unet()# load model 
IP_unet_model.load_state_dict(torch.load('IP_unet_model_phase2.pt'))
IP_unet_model.eval()

st.title('Pytorch Image Inpainting')

img = st.sidebar.selectbox(
    'Select Image',
    ("0.png",  "1.png",  "2.png",  "3.png",  "4.png",  "5.png")
)

# model= "saved_models/" + style_name + ".pth"
input_image = "imgs/input/" + img
output_image = "imgs/target/" + img

st.write('### Source image:')
image = Image.open(input_image)
st.image(image, width=200) # image: numpy array

clicked = st.button('Inferece')

if clicked:
    st.write('### Image Inpainting:')
    img = proces_img(input_image)
    sr_image = IP_unet_model(img)
    sr_image = sr_image.squeeze(0).detach().numpy()
    sr_image = np.transpose(sr_image, (1, 2, 0))
    sr_image = Image.fromarray(((sr_image + 1) * 127.5).astype(np.uint8))
    st.image(sr_image, width=400)

    st.write('### Target image:')
    image = Image.open(output_image)
    st.image(image, width=400)
