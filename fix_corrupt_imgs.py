import os
from PIL import Image

def validate_images(directory):
    corrupted_files = []
    
    # Walk through directory and sub-directories
    for dirpath, _, filenames in os.walk(directory):
        print(f"Scanning directory: {dirpath}")
        
        for image_file in filenames:
            # Check for common image extensions
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(dirpath, image_file)
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted_files.append(image_path)
                    print(f"Error with {image_path}: {e}")
    
    return corrupted_files


# Example usage:
directory = "./data/train/healthy" 
corrupted_images = validate_images(directory)
if corrupted_images:
    print(f"Found {len(corrupted_images)} corrupted images.")
else:
    print("All images are valid!")