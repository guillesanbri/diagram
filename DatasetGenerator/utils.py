import cv2
import glob
import json
import numpy as np


# TODO: Write tests for these functions
# TODO: Test excluding and including certain shapes.
def get_paths(path, element_ids):
    """
    Finds all diagram element files with and id in the element_ids list
    and returns a list with their paths.

    :param path: Path to the directory where the images of the
     elements are located.
    :param element_ids: List of element ids to match.
    :return: List with a path for every file found with an id on element_ids.
    """
    paths = []
    for _id in element_ids:
        filename = f'?????-???-???-{_id}-???-???.png'
        paths += glob.glob(path + filename)
    return paths


def get_shapes_paths(elements_path,
                     include_suffixes=None, exclude_suffixes=None):
    """
    Gets a list containing every path corresponding to a shape image.
    Paths can be filtered by their suffixes, either inclusively or exclusively.
    Suffixes are NOT ids, suffixes are, ie: rectangular-parallelogram,
    triangle, ellipse, etc.

    :param elements_path: Path to the directory where the images of the
     elements are located.
    :param include_suffixes: A list of suffixes to be included in the list.
     If both include and exclude suffixes are empty, all shapes are included
     in the returned list.
    :param exclude_suffixes: A list of suffixes to be excluded from the list.
     If both include and exclude suffixes are empty, all shapes are included
     in the returned list.
    :return: List with a path for each valid shape image.
    :raises ValueError: If both include_suffixes and exclude_suffixes are
     populated.
    """
    with open("./picture_ids/element_id.json") as f:
        suffix_dict = json.load(f)["shapes"]
    if include_suffixes and exclude_suffixes:
        raise ValueError("Only one of the two prefixes lists can be defined"
                         "at the same time.")
    elif include_suffixes:
        ids = [suffix_dict[key] for key in include_suffixes]
    elif exclude_suffixes:
        all_ids = suffix_dict.values()
        exclude_ids = [suffix_dict[key] for key in exclude_suffixes]
        ids = set(all_ids) - set(exclude_ids)
    else:  # empty_include and empty_exclude
        ids = suffix_dict.values()
    paths = get_paths(elements_path, ids)
    return paths


def scale_image(image, max_size_length):
    """
    Scales a given image (np.array) with only one channel given a maximum size.
    This maximum size will dictate the size of the largest size of the image,
    while its shorter side will be scaled accordingly to preserve the aspect
    ratio.
    :param image: Image to be scaled.
    :param max_size_length: Size in pixels of the largest side of the resulting
     image.
    :return: Resized image.
    :raises ValueError: If the image has more than two dimensions, ie: a channel
     dimension.
    """
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
    print(get_shapes_paths("elements/"))
    print(get_shapes_paths("elements/", include_suffixes=["ellipse"]))
    print(get_shapes_paths("elements/", exclude_suffixes=["ellipse"]))
    img = cv2.imread("tests/test_inter.png", cv2.IMREAD_GRAYSCALE)
    res = scale_image(img, 250)
    cv2.imshow("test", res)
    cv2.waitKey(0)
