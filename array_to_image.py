from PIL import Image
import numpy as np
import json


def array_to_image(arrs: np.array):
    """
    convert multi-2-d array to images
    :param arrs: n * h * w numpy array, n: number of images, h: height of image, w: weight of image
    :return: void
    """
    for i, arr in enumerate(arrs):
        img = Image.fromarray(arr).convert('RGB')
        with open('imgs/{}.png'.format(i), 'wb') as f:
            img.save(f)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    images = np.load(config['image_file'])
    array_to_image(images)