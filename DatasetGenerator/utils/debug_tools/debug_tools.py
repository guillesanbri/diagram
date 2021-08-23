import numpy as np
import random
import cv2


def visualize_clusters(image, coordinates, labels):
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    palette = [random.choices(range(256), k=3) for _ in range(max(labels)+1)]
    for i, coord in enumerate(coordinates):
        if labels[i] != -1:
            canvas[coord[0], coord[1]] = palette[labels[i]]
    return canvas


def visualize_element(image, coordinates):
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for coord in coordinates:
        canvas[coord[0], coord[1]] = (255, 0, 0)
    return canvas


def draw_bbox(image, bbox):
    return cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 220, 255), 2)
