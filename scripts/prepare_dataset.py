import os
import glob
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm


DRIVE_DATASET_PATH = "/content/drive/MyDrive/datasets/coco"  # we used the COCO dataset
SUBSET_DATASET_PATH = "/content/drive/MyDrive/datasets/coco_subset_16000"
NUM_IMAGES_TO_EXTRACT = 16000


# Get all image paths (with common extensions)
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

def collect_image_paths(folder):
    """Get all image paths from folder and subfolders, duplicates removed"""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        # Match files in the immediate directory
        paths.extend(glob.glob(os.path.join(folder, ext)))
        # Match files in all subdirectories (recursive depth search)
        paths.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
    # Remove duplicates
    return list(set(paths))


def is_grayscale_image(img_path):
    """Check if image is grayscale by comparing RGB channels"""
    try:
        img = Image.open(img_path).convert("RGB")  # convert image to RGB mode
        img_array = np.array(img)

        # Check if all channels are identical
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]  # separate R, G, B channels
        return np.array_equal(r, g) and np.array_equal(g, b)  # an image is grayscale if all channels are identical
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        raise e


def remove_grayscale_images(dataset_path):
    """Grayscale photos removal from dataset"""
    all_image_paths = collect_image_paths(dataset_path)

    if len(all_image_paths) == 0:
        print(f"No images found in {dataset_path}. Please check the path!")
    else:
        print(f"Found {len(all_image_paths)} images in dataset!")
        print("Deleting grayscale images...\n")

        grayscale_count = 0
        color_count = 0
        error_count = 0

        # Process each image
        for img_path in tqdm(all_image_paths, desc="Processing images"):  # tqdm - shows progress meter
            try:
                if is_grayscale_image(img_path):
                    # Delete grayscale image from Drive for dataset correction
                    os.remove(img_path)
                    grayscale_count += 1
                else:
                    color_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nError processing {os.path.basename(img_path)}: {e}")

        print("-" * 40)
        print(f"Processing complete! Cleanup summary:")
        print(f"   Processed:      {len(all_image_paths)}")
        print(f"   Deleted (gray): {grayscale_count}")
        print(f"   Kept (color):   {color_count}")
        print(f"   Errors:         {error_count}")
        print("-" * 40)


def create_subset(src_path, dst_path, n):
    """Create a subset dataset folder"""
    all_image_paths = collect_image_paths(src_path)

    if len(all_image_paths) == 0:
        print(f"No images found in {src_path}. Please check the path!")
    else:
        print(f"Found {len(all_image_paths)} images in dataset!")

        # Shuffle images with random-sampling with seed 42
        np.random.seed(42)
        np.random.shuffle(all_image_paths)

        # Select the first N images after shuffling
        selected_paths = all_image_paths[:n]

        # Create the destination directory if it doesn't exist
        os.makedirs(dst_path, exist_ok=True)

        print(f"Copying {len(selected_paths)} images to: {dst_path}...")

        copy_count = 0
        error_count = 0

        for img_path in tqdm(selected_paths, desc="Copying files"):
            try:
                filename = os.path.basename(img_path)  # extract file name
                dest_path = os.path.join(dst_path, filename)  # create file with the above file name in the destination folder
                # copy the image to the destination path
                shutil.copy2(img_path, dest_path)  # shutil.copy2 preserves original file metadata
                copy_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nError copying {filename}: {e}")

        print("-" * 40)
        print("Execution Summary:")
        print(f"  Images to copy: {len(all_image_paths)}")
        print(f"  Copied:         {copy_count}")
        print(f"  Errors:         {error_count}")
        print("-" * 40)


def count_images(folder_path):
    """Count the images left in dataset subfolder after manually filtering them"""
    all_image_paths = collect_image_paths(folder_path)

    if len(all_image_paths) == 0:
        print(f"No images found in {folder_path}. Please check the path!")
    else:
        print(f"Found {len(all_image_paths)} images in dataset!")

        # Count by extension
        extensions_count = {}
        for img_path in all_image_paths:
            ext = os.path.splitext(img_path)[1].lower()  # split text into root and extension, and get the ext
            extensions_count[ext] = extensions_count.get(ext, 0) + 1  # dictionary of extension frequency

        print(f"\nDataset distribution:")
        for ext, count in sorted(extensions_count.items()):
            print(f"   {ext}: {count} images")


if __name__ == "__main__":
    # Run each step in order. Comment out steps you have already completed.
    remove_grayscale_images(DRIVE_DATASET_PATH)
    create_subset(DRIVE_DATASET_PATH, SUBSET_DATASET_PATH, NUM_IMAGES_TO_EXTRACT)
    count_images(SUBSET_DATASET_PATH)
