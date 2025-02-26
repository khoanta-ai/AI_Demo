# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


imsize = 256
img_transforms = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def image_loader(image):
    """Loads image and applies transformations."""
    image = Image.open(image)
    image = img_transforms(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(tensor):
    """Calculates the gram matrix."""
    a, b, c, d = tensor.size()
    tensor = tensor.view(a * b, c * d)
    G = torch.mm(tensor, tensor.t())
    return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(torch.tensor([0.485, 0.456, 0.406]).to(device)).view(-1, 1, 1)
        self.std = torch.tensor(torch.tensor([0.229, 0.224, 0.225]).to(device)).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

normalization = Normalization().to(device)

VGG19_pretrained = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
VGG19_pretrained.to(device)

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_features(pretrained_model, image):
    """Extracts features from pretrained VGG model."""
    layers = {
        '0': 'conv_1',
        '5': 'conv_2',
        '16': 'conv_3',
        '25': 'conv_4',
        '34': 'conv_5'
    }
    features = {}
    x = image
    x = normalization(x)
    for name, pretrained_layer in pretrained_model._modules.items():
        x = pretrained_layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def rot_style_features(style_features, style_layers):
    """Rotates style features to create style."""
    final_rot_style_features = {}
    for layer in style_layers:
        sf = style_features[layer].clone()
        rot90 = torch.rot90(sf.clone(), 1, (2, 3))
        rot180 = torch.rot90(rot90.clone(), 1, (2, 3))
        final_rot = sf + (rot90 - rot180)
        final_rot_style_features[layer] = final_rot
    return final_rot_style_features

ContentLoss = nn.MSELoss()
StyleLoss = nn.MSELoss()

def style_tranfer_(model, optimizer, target_img,
                    content_features, style_features,
                    style_layers, content_weight, style_weight):
    """Performs one step of style transfer."""

    optimizer.zero_grad()
    with torch.no_grad():
        target_img.clamp_(0, 1)
    target_features = get_features(model, target_img)

    content_loss = ContentLoss(content_features['conv_4'], target_features['conv_4'])

    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += StyleLoss(style_gram, target_gram)

    total_loss = content_loss*content_weight + style_loss*style_weight
    total_loss.backward(retain_graph=True)
    optimizer.step()
    return total_loss, content_loss, style_loss

# --- Streamlit App ---
st.title("Neural Style Transfer")

st.sidebar.header("Upload Images")
style_image_file = st.sidebar.file_uploader("Choose Style Image", type=["jpg", "jpeg", "png"])
content_image_file = st.sidebar.file_uploader("Choose Content Image", type=["jpg", "jpeg", "png"])

content_weight = 1.0
style_weight = 1e6
steps = st.sidebar.number_input("Number of Steps", min_value=100, max_value=1000, value=100, step=100)

st.header("Uploaded Images")
cols = st.columns(2)
if style_image_file:
    style_img_pil = Image.open(style_image_file)
    cols[0].image(style_img_pil, caption="Style Image", use_column_width=True)
if content_image_file:
    content_img_pil = Image.open(content_image_file)
    cols[1].image(content_img_pil, caption="Content Image", use_column_width=True)

if st.button("Run Style Transfer"):
    if not style_image_file or not content_image_file:
        st.error("Please upload style and content images.")
    else:
                style_img = image_loader(style_image_file)
                content_img = image_loader(content_image_file)

                content_features = get_features(VGG19_pretrained, content_img)
                style_features1 = get_features(VGG19_pretrained, style_img)
                final_rot_style_features = rot_style_features(style_features1, style_layers)

                target_img1 = content_img.clone().requires_grad_(True).to(device)
                target_img2 = content_img.clone().requires_grad_(True).to(device)
                optimizer1 = optim.Adam([target_img1], lr=0.02)
                optimizer2 = optim.Adam([target_img2], lr=0.02)

                for step in range(steps):
                    total_loss1, content_loss1, style_loss1 = style_tranfer_(VGG19_pretrained, optimizer1, target_img1,
                                                                            content_features, style_features1,
                                                                            style_layers, content_weight, style_weight)

                    total_loss2, content_loss2, style_loss2 = style_tranfer_(VGG19_pretrained, optimizer2, target_img2,
                                                                            content_features, final_rot_style_features,
                                                                            style_layers, content_weight, style_weight)

                    if step % 100 == 99:
                        st.write(f"Epoch [{step+1}/{steps}] - Style 1 - Total loss: {total_loss1.item():.6f} - Content loss: {content_loss1.item():.6f} - Style loss: {style_loss1.item():.6f}")
                        st.write(f"Epoch [{step+1}/{steps}] - Style 2 - Total loss: {total_loss2.item():.6f} - Content loss: {content_loss2.item():.6f} - Style loss: {style_loss2.item():.6f}")

                with torch.no_grad():
                    target_img1.clamp_(0, 1)
                    target_img2.clamp_(0, 1)

                output_image1 = target_img1.cpu().clone()
                output_image1 = output_image1.squeeze(0)
                output_image1 = transforms.ToPILImage()(output_image1)

                output_image2 = target_img2.cpu().clone()
                output_image2 = output_image2.squeeze(0)
                output_image2 = transforms.ToPILImage()(output_image2)


                st.header("Output Images")
                output_cols = st.columns(2)
                output_cols[0].image(output_image1, caption="Output Image 1 (Original Style)", use_column_width=True)
                output_cols[1].image(output_image2, caption="Output Image 2 (Rotated Style)", use_column_width=True)
                st.success("Style transfer complete!")


