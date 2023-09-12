import cv2
import numpy as np
import argparse
from skimage.exposure import match_histograms
from pathlib import Path


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='alderley', help='path of the dataset used for training')


# Defien input and output folders
in_folder = Path('check/multi_src_40e/testB2A_uavid_512_single_std')
night_folder = Path('datasets/multi_src/testA')
out_folder = Path('hist_optim_results')
out_folder.mkdir(exist_ok=True)

# Load all the images in the given folder
images = [cv2.imread(str(img)) for img in in_folder.glob('*.png')]
night_images = [cv2.imread(str(img)) for img in night_folder.glob('*.jpg')]

# Pre-define the output lists
equalized_images = []
saturated_images = []
hist_specialized_images = []

# Equalize the histogram of each image
for img in images:
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
    ycrcb_img[..., 0] = cv2.equalizeHist(ycrcb_img[..., 0])  # Equalize the histogram of the Y channel
    equalized_images.append(cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR))  # Convert back to BGR color space

# Increase the saturation of each image
for img in images:
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
    ycrcb_img[..., 0] = cv2.subtract(ycrcb_img[..., 0], 15)  # Decrease the brightness of the Y channel
    saturated_images.append(cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR))  # Convert back to BGR color space

# Extract all the night iamges Y histograms and perform a mean
night_ycrcb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in night_images]
night_y_hist = [cv2.calcHist(img, [0], None, [256], [0, 256]) for img in night_ycrcb_images]
night_y_hist_avg = sum(night_y_hist) / len(night_y_hist) # Avarage the histograms

# Save the night images Y histograms
np.save(str(out_folder / 'night_y_hist.npy'), night_y_hist_avg)

# Specialize the histogram of each image
for img in images:
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
    ycrcb_img[..., 0] = match_histograms(ycrcb_img[..., 0], night_y_hist_avg)  # Specialize the histogram of the Y channel
    hist_specialized_images.append(cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR))  # Convert back to BGR color space

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