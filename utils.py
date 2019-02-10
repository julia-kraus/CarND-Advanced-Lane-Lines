import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(img):
    """returns the image converted to grayscale. Pay attention to how an image was loaded. If it was loaded using
    matplotlib.image, the image is in RGB, if it was loaded with cv2.imread, it is in BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def hls(img):
    """Returns the image converted to HSL colorspace."""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

def load_image(path, to_rgb=True):
    """
    Load image from the given path. By default the returned image is in RGB.
    When to_rgb is set to False the image return is in BGR. Returns by default a RGB image.
    """
    img = cv2.imread(path)
    
    if not to_rgb:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_img_lists(image_lists, image_names=None, title=None, figsize=(20, 20), ticks=False):
    """Helper function that shows several lists of images alongside each other"""
    rows = len(image_lists[0])
    cols = len(image_lists)
    cmap = None
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for j in range(cols):
        for i in range(rows):
            # if there is more than one image in the list --> more than one row
            if rows >=2:
                ax = axes[i, j]
                image = image_lists[j][i]
                image_name = image_names[j][i]
            
            else: 
                ax = axes[j]
                image = image_lists[j]
                image_name=image_names[j]
                
            
            # if image has less than three color channels
            if image.shape[-1] < 3 or len(image.shape)<3:
                cmap="gray"
                image = np.reshape(image, (image.shape[0], image.shape[1]))
                
            if not ticks:
                ax.axis("off")
                
            ax.imshow(image, cmap=cmap)
            ax.set_title(image_name)
            
    fig.suptitle(title, fontsize=10, y=1)
    fig.tight_layout()
    plt.show()
    
    return
            