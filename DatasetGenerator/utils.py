import cv2
import numpy as np


def scale_image(image, max_size_length):
    if len(image.shape) != 2:
        raise ValueError("Image must have only one channel. Shape has to be (h, w).")
    height, width = image.shape
    max_i = np.argmax([height, width])
    if max_i == 0:
        new_height, new_width = max_size_length, int(max_size_length/height*width)
    else:
        new_height, new_width = int(max_size_length / width * height), max_size_length
    return cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)


if __name__ == "__main__":
    img = cv2.imread("tests/test_inter.png", cv2.IMREAD_GRAYSCALE)
    res = scale_image(img, 250)
    cv2.imshow("test", res)
    cv2.waitKey(0)