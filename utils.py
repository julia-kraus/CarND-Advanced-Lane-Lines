import cv2
import numpy as np

def to_grayscale(img):
    """returns the image converted to grayscale. Pay attention to how an image was loaded. If it was loaded using
    matplotlib.image, the image is in RGB, if it was loaded with cv2.imread, it is in BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def load_image(path, to_rgb=True):
    """
    Load image from the given path. By default the returned image is in RGB.
    When to_rgb is set to False the image return is in BGR. Returns by default a RGB image.
    """
    img = cv2.imread(path)
    return img if not to_rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    