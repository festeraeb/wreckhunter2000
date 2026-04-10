import numpy as np
from skimage.measure import regionprops, label
from skimage.io import imread
import os

def extract_features(image_path):
    """
    Extract aspect ratio, symmetry, and boundary sharpness from an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Extracted features.
    """
    image = imread(image_path, as_gray=True)
    labeled_image = label(image > 0.5)  # Thresholding to binary
    regions = regionprops(labeled_image)

    features = []
    for region in regions:
        aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
        symmetry = np.abs(region.centroid[0] - region.centroid[1])
        boundary_sharpness = np.std(region.perimeter)

        features.append({
            "aspect_ratio": aspect_ratio,
            "symmetry": symmetry,
            "boundary_sharpness": boundary_sharpness
        })

    return features

if __name__ == "__main__":
    # Example usage
    image_dir = "../sample_data/sar_optical_patches"
    output_file = "../sample_data/features.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_features = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            features = extract_features(image_path)
            all_features.extend(features)

    print(f"Extracted features: {all_features}")