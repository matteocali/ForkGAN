from operator import le
from weakref import ref
import cv2
import numpy as np
import argparse
import typing as tp
from skimage.exposure import match_histograms
from pathlib import Path
from tqdm import tqdm


# Define a function to convert a string to a Path object
def str2path(path):
    return Path(path)


# Define the arguments parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--reference_path', dest='reference', default='datasets/night_acdc', type=str2path, help='path of the dataset to use as reference')
parser.add_argument('--input_path', dest='input', required=False, type=str2path, help='path of the dataset to convert')
parser.add_argument('--output_path', dest='output', required=False, type=str2path, help='path of the output folder')
parser.add_argument('--hist_path', dest='hist_path', default=Path('./hist.npy'), type=str2path, help='path of the file to save the histogram of the reference dataset')
parser.add_argument('--half', dest='half', action='store_true', help='use only the bottom half of the images to extract the histogram')
parser.add_argument('--quartile', dest='quartile', action='store_true', help='remove the 25th and 75th percentile from the histograms')


# Define the auxiliary functions
def extract_hist(images: list, b_half: bool = False, quartile: bool = False, diagonal: bool = False) -> np.ndarray:
    # Extract all the night iamges Y histograms and perform a mean
    ref_ycrcb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in tqdm(images, desc='Converting to YCrCb color space', leave=False, total=len(images))]
    ref_y_hist = [cv2.calcHist(img, [0], None, [256], [0, 256]) for img in tqdm(ref_ycrcb_images, desc='Extracting Y histograms', leave=False, total=len(images))]
    if b_half:
        ref_y_hist = [cv2.calcHist(img[img.shape[0]//2:, :], [0], None, [128], [0, 256]) for img in tqdm(ref_ycrcb_images, desc='Extracting Y histograms', leave=False, total=len(images))]
    elif quartile:
        ref_y_hist = [np.where((hist < np.percentile(hist, 25, keepdims=True, axis=1)) & (hist > np.percentile(hist, 75, keepdims=True, axis=1)), 0, hist) for hist in ref_y_hist]
    else:
        pass
    return sum(ref_y_hist) / len(ref_y_hist) # Avarage the histograms

def hist_spec(images: list, hist: np.ndarray) -> tp.Tuple[tp.List, tp.List, tp.List]:
    # Pre-define the output lists
    equalized_images = []
    saturated_images = []
    hist_specialized_images = []

    # Equalize the histogram of each image
    for img in tqdm(images, desc='Equalizing histograms', leave=False, total=len(images)):
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
        ycrcb_img[..., 0] = cv2.equalizeHist(ycrcb_img[..., 0])  # Equalize the histogram of the Y channel
        equalized_images.append(cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR))  # Convert back to BGR color space
    
    # Increase the saturation of each image
    for img in tqdm(images, desc='Increasing saturation', leave=False, total=len(images)):
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
        ycrcb_img[..., 0] = cv2.subtract(ycrcb_img[..., 0], 15)  # Decrease the brightness of the Y channel
        saturated_images.append(cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR))  # Convert back to BGR color space

    # Specialize the histogram of each image
    for img in tqdm(images, desc='Specializing histograms', leave=False, total=len(images)):
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
        ycrcb_img[..., 0] = match_histograms(ycrcb_img[..., 0], hist)  # Specialize the histogram of the Y channel
        hist_specialized_images.append(cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR))  # Convert back to BGR color space

    return equalized_images, saturated_images, hist_specialized_images


if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    # Defien input and output folders
    ref_path = args.reference
    hist_path = args.hist_path

    if args.input is not None:
        in_folder = args.input
        out_folder = args.output
        if out_folder is None:
            out_folder = in_folder.parent / f'{in_folder.name}_specified_hist'
        out_folder.mkdir(exist_ok=True, parents=True)

    # Extract the histograms of the reference dataset
    ref_images = [cv2.imread(str(img)) for img in tqdm(ref_path.rglob('*.[pj][np][gg]*'), desc='Loading images', leave=False)]
    ref_y_hist = extract_hist(ref_images, b_half=args.half, quartile=args.quartile, diagonal=args.diagonal)

    if hist_path is not None:
        np.save(str(hist_path), ref_y_hist)

    if args.input is None:
        exit(0)
    
    # Specialize the histogram of the input images
    images = [cv2.imread(str(img)) for img in tqdm(in_folder.rglob('*.[pj][np][gg]*'), desc='Loading images', leave=False)]
    equalized_images, saturated_images, hist_specialized_images = hist_spec(images, ref_y_hist)

    # Load all the images in the given folder
    images = [cv2.imread(str(img)) for img in in_folder.glob('*.png')]
    night_images = [cv2.imread(str(img)) for img in ref_path.glob('*.jpg')]

    # Add title to each image
    images = [cv2.putText(img, 'Original image', (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3) for img in images]
    equalized_images = [cv2.putText(img, 'Equalized histogram image', (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3) for img in equalized_images]
    hist_specialized_images = [cv2.putText(img, 'Specialized histogram image', (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3) for img in hist_specialized_images]
    saturated_images = [cv2.putText(img, 'Reduced brightness image', (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3) for img in saturated_images]

    # Concatenate the images
    concat_images = [cv2.hconcat([images[i], equalized_images[i], hist_specialized_images[i], saturated_images[i]]) for i in range(len(images))]

    # Save the images
    for i in range(len(images)):
        cv2.imwrite(str(out_folder / f'concat_{i}.png'), concat_images[i])

    # Visualize the results side by side with the original images
    for i in range(len(images)):
        cv2.imshow('img', concat_images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()