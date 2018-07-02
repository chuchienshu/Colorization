import numpy as np
from skimage.io import imread
from skimage import color
import random
from skimage.transform import resize
from glob import glob
from os.path import *

# Read into file paths and randomly shuffle them
root = '/home/chuchienshu/Documents/propagation_refine/data/sintel_test_clean/'
filename_lists = sorted(glob(join(root, '*/*.png')))

random.shuffle(filename_lists)
# Load 313 bins on the gamut
# points (313, 2)
points = np.load('/home/chuchienshu/Documents/propagation_classification/models/custom_layers/pts_in_hull.npy')
points = points.astype(np.float64)
# points (1, 313, 2)
points = points[None, :, :]

# probs (313,)
probs = np.zeros((points.shape[1]), dtype=np.float64)
num = 0

def get_index( in_data ):

	expand_in_data = np.expand_dims(in_data, axis=1)
	distance = np.sum(np.square(expand_in_data - points), axis=2)

	return np.argmin(distance, axis=1)

for num, img_f in enumerate(filename_lists):
    img = imread(img_f)
    img = resize(img, (256, 256), preserve_range=True)

    # Make sure the image is rgb format
    if len(img.shape) != 3 or img.shape[2] != 3:
        continue
    img_lab = color.rgb2lab(img)
    img_lab = img_lab.reshape((-1, 3))

    # img_ab (256^2, 2)
    img_ab = img_lab[:, 1:].astype(np.float64)

    nd_index = get_index(img_ab)
    for i in nd_index:
        i = int(i)
        probs[i] += 1
    print(num)


# Calculate probability of each bin
probs = probs / np.sum(probs)
#print(probs)
# Save the result
print(np.sum(probs))
np.save('sintel_ctest_prior_probs', probs)
