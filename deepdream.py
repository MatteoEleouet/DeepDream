import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import timm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Unable to find the image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def add_text(img, text, position, font_scale=1, font_color=(0, 0, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Adjust the y position to move the text 5 pixels lower
    adjusted_position = (position[0], position[1] + 5)
    return cv2.putText(img, text, adjusted_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Function to pad the top of the image
def pad_image_top(img, pad_height=40, pad_color=(255, 255, 255)):
    padded_img = np.full((pad_height, img.shape[1], 3), pad_color, dtype=np.uint8)
    return np.vstack((padded_img, img))


def deep_dream(image, model, iterations, lr):
    transformed = image.clone().detach().to(device)
    transformed.requires_grad = True

    optimizer = torch.optim.Adam([transformed], lr=lr)

    for i in range(iterations):
        optimizer.zero_grad()
        out = model(transformed)
        loss = -out.norm()
        loss.backward()
        optimizer.step()

        # Optional: Clip the values to maintain image quality
        transformed.data.clamp_(0, 1)

    return transformed


# Load the models and transfer them to the device
models = {
    'vgg19': models.vgg19(pretrained=True).features.to(device).eval(),
    'vit': timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval(),  # Load ViT model
    'inception': models.inception_v3(pretrained=True, transform_input=False).to(device).eval()
}

# Disable gradients for all models
for _, model in models.items():
    for param in model.parameters():
        param.requires_grad = False

# Load your image
img_path = 'wallpaper1.jpg'  # Update with your image path
original_image = load_image(img_path)

# Apply DeepDream
model_iterations = 200  # Number of iterations per model
iteration_results = {model_name: [] for model_name in models}

for model_name, model in models.items():
    print(f"Processing with {model_name}")
    processed_image = original_image.clone()
    for iteration in range(model_iterations):
        processed_image = deep_dream(
            processed_image, model, iterations=15, lr=0.02)

        # Store each iteration's result
        image_np = processed_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        iteration_results[model_name].append(image_np)

        print(f"Iteration {iteration + 1} complete for {model_name}")

# Create frames by combining the results side by side
frames = []
for iteration in range(model_iterations):
    combined_image = None
    for model_name in models.keys():
        if combined_image is None:
            combined_image = iteration_results[model_name][iteration]
        else:
            combined_image = np.hstack(
                (combined_image, iteration_results[model_name][iteration]))

    # Pad the top of the image
    combined_image = pad_image_top(combined_image, pad_height=40)

    # Calculate individual column width
    col_width = combined_image.shape[1] // 3

    # Add text labels at the top of each column
    combined_image = add_text(combined_image, 'VGG19', (col_width//2 - 50, 25))  # y coordinate changed to 25
    combined_image = add_text(combined_image, 'ViT', (col_width + col_width//2 - 80, 25))  # y coordinate changed to 25
    combined_image = add_text(combined_image, 'InceptionV3', (2 * col_width + col_width//2 - 80, 25))  # y coordinate changed to 25

    frames.append(combined_image)
# Create a video from frames
clip = ImageSequenceClip(frames, fps=30)
clip.write_videofile('deepdream_video.mp4', codec='libx264')
