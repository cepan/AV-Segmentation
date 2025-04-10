import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import glob
import torchvision.transforms.functional as TF


def crop_and_pad_to_square(image):
    """
    Crops an image to the bounding box of non-zero regions and pads it to make it square.
    The image is assumed to be in the shape (C, H, W) with values in the range [0, 1].
    """

    # Convert the image to grayscale
    gray = TF.rgb_to_grayscale(image)
    gray_uint8 = (gray * 255).to(torch.uint8)

    # Create a binary mask
    binary_mask = (gray_uint8 > 25).float()  # Use a threshold of 25 for uint8

    # Find the non-zero elements
    non_zero_indices = torch.nonzero(binary_mask[0], as_tuple=False)

    if non_zero_indices.size(0) == 0:
        # If the image is completely black or has no non-zero region, return it as is
        return image

    # Get the bounding box of the non-zero region
    top_left = torch.min(non_zero_indices, dim=0).values
    bottom_right = torch.max(non_zero_indices, dim=0).values

    # Crop the image to the bounding box
    cropped_image = image[:, top_left[0]                          :bottom_right[0], top_left[1]:bottom_right[1]]
    cropped_height = bottom_right[0] - top_left[0]
    cropped_width = bottom_right[1] - top_left[1]

    # Pad the image to make it square
    if cropped_width > cropped_height:
        padded_image = TF.pad(cropped_image, [
                              0, (cropped_width - cropped_height) // 2, 0, (cropped_width - cropped_height + 1) // 2])
    elif cropped_height > cropped_width:
        padded_image = TF.pad(cropped_image, [(
            cropped_height - cropped_width) // 2, 0, (cropped_height - cropped_width + 1) // 2, 0])
    else:
        padded_image = cropped_image  # Already square

    return padded_image


def extract_av_mask(av, threshold=0.2):
    """
    Extract artery, vein, and vessel masks from the AV image.
    Args:
        av (torch.Tensor): AV image tensor with shape (3, H, W).
        threshold (float): Threshold for determining artery and vein masks.
    Returns:
        tuple: Artery mask, vein mask, vessel mask, AV mask tensors.
    """
    # Split the image into R, G, B channels
    R = av[0]  # Red channel
    B = av[2]  # Blue channel
    diff = R - B

    # Create masks for arteries and veins
    artery_mask = (diff > threshold).float()
    vein_mask = (diff < -threshold).float()
    ves_mask = (artery_mask + vein_mask).clamp(0, 1)

    # Create an empty AV mask
    H, W = artery_mask.shape
    av_mask = torch.zeros(3, H, W)

    # Assign colors: Red for artery, Blue for vein, Green for overlap
    av_mask[0] = artery_mask  # Red
    av_mask[2] = vein_mask    # Blue
    av_mask[1] = (artery_mask * vein_mask)  # Green for overlap

    return artery_mask.unsqueeze(0), vein_mask.unsqueeze(0), ves_mask.unsqueeze(0), av_mask


def normalize_image(image, tolerance=1e-5):
    """
    Normalize an image to the range [0, 1].
    If the image is already in the range [0, 1], it will remain unchanged.
    Args:
        image (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Normalized image tensor in the range [0, 1].
    """

    if image.max() > 1.0 + tolerance:
        image = image / 255.0
    return image.clamp(0, 1)


def process_image(img_path, target_size=512, is_rgb=True):
    """
    Process an image using the same pipeline as in dataset.py

    Args:
        img_path: Path to the image
        target_size: Target image size
        is_rgb: Whether to process as RGB or grayscale

    Returns:
        Processed image tensor
        """
    # Open the image
    img = Image.open(img_path)
    if is_rgb:
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    # Apply the same preprocessing pipeline as in dataset.py
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        crop_and_pad_to_square,
        transforms.Resize(target_size),
        normalize_image,
    ])

    img = preprocess(img)
    return img


def save_image(img_tensor, save_path):
    """Save a tensor as an image"""
    # Convert tensor to PIL image and save
    pil_img = transforms.ToPILImage()(img_tensor)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_img.save(save_path)


def process_dataset(dataset_name):
    """
    Process a specific dataset and organize its data into the new structure.

    Args:
        dataset_name (str): Name of the dataset (RITE, FIVES, or RMHAS)
    """
    source_dir = os.path.join("../data", dataset_name)
    target_dir = os.path.join("../processed_data", dataset_name)

    # Define expected subdirectories
    image_dir = os.path.join(source_dir, "image")
    vessel_dir = os.path.join(source_dir, "vessel")
    av_dir = os.path.join(source_dir, "av_prediction")

    # Collect all file names
    image_files = sorted(os.listdir(image_dir))
    vessel_files = sorted(os.listdir(vessel_dir))
    av_files = sorted(os.listdir(av_dir))

    # Build base-name to file mappings
    image_map = {f.rsplit('.', 1)[0]: f for f in image_files}
    vessel_map = {f.rsplit('.', 1)[0]: f for f in vessel_files}
    av_map = {f.rsplit('.', 1)[0]: f for f in av_files}

    # Find common image base names
    common_keys = sorted(set(image_map) & set(vessel_map) & set(av_map))
    print(f"Found {len(common_keys)} common images in {dataset_name}.")

    # Prepare output folders
    for subfolder in ['image', 'vessel', 'artery', 'vein']:
        os.makedirs(os.path.join(target_dir, subfolder), exist_ok=True)

    for key in common_keys:
        img_path = os.path.join(image_dir, image_map[key])
        vessel_path = os.path.join(vessel_dir, vessel_map[key])
        av_path = os.path.join(av_dir, av_map[key])

        # Process raw images
        img_tensor = process_image(img_path, target_size=512, is_rgb=True)
        vessel_tensor = process_image(
            vessel_path, target_size=512, is_rgb=False)
        av_tensor = process_image(av_path, target_size=512, is_rgb=True)

        # Extract artery, vein, vessel masks
        artery_mask, vein_mask, vessel_mask, _ = extract_av_mask(av_tensor)

        # Save all processed tensors as images
        save_image(img_tensor, os.path.join(target_dir, "image", f"{key}.png"))
        save_image(vessel_tensor, os.path.join(
            target_dir, "vessel", f"{key}.png"))
        save_image(artery_mask, os.path.join(
            target_dir, "artery", f"{key}.png"))
        save_image(vein_mask, os.path.join(target_dir, "vein", f"{key}.png"))


def main():
    # Create directory structur

    # Process each dataset
    datasets = ["RMHAS", "FIVES", "RITE"]
    for dataset in datasets:
        print(f"Processing {dataset} dataset...")
        process_dataset(dataset)
        print(f"Finished processing {dataset} dataset")


if __name__ == "__main__":
    main()
