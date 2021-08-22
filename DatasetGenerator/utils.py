from PIL import Image
import numpy as np
import math
import glob
import json
import os
import cv2


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


def get_element_paths(element_type, elements_path="elements/",
                      include_suffixes=None, exclude_suffixes=None):
    """
    Gets a list with every image path of a specified element.
    Paths can be filtered by their suffixes, either inclusively or exclusively.
    Suffixes are NOT ids, suffixes are, ie: rectangular-parallelogram,
    triangle, ellipse, etc.

    :param element_type: Type of the element to load paths of, this can take
     values that are first order keys in the element_id.json file, ie: texts,
     connections, shapes, etc.
    :param elements_path: Path to the directory where the images of the
     elements are located.
    :param include_suffixes: A list of suffixes to be included in the list.
     If both include and exclude suffixes are empty, all shapes are included
     in the returned list.
    :param exclude_suffixes: A list of suffixes to be excluded from the list.
     If both include and exclude suffixes are empty, all shapes are included
     in the returned list.
    :return: Tuple containing: (Dictionary of id as keys and the suffix as
     value, List with a path for each valid shape image).
    :raises ValueError: If both include_suffixes and exclude_suffixes are
     populated.
    """
    with open("./picture_ids/element_id.json") as f:
        try:
            suffix_dict = json.load(f)[element_type]
        except KeyError:
            raise KeyError(f"{element_type} is not a valid element, "
                           f"check the element_id.json file.")
    if include_suffixes and exclude_suffixes:
        raise ValueError("Only one of the two prefixes lists can be defined"
                         "at the same time.")
    elif include_suffixes:
        suffixes = include_suffixes
        ids = [suffix_dict[key] for key in include_suffixes]
    elif exclude_suffixes:
        all_ids = suffix_dict.values()
        exclude_ids = [suffix_dict[key] for key in exclude_suffixes]
        suffixes = set(suffix_dict.keys()) - set(exclude_suffixes)
        ids = set(all_ids) - set(exclude_ids)
    else:  # empty_include and empty_exclude
        suffixes = suffix_dict.keys()
        ids = suffix_dict.values()
    paths = get_paths(elements_path, ids)
    return dict(zip(ids, suffixes)), paths


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


def check_file_path(file):
    """
    Checks if a file exists. If it does not exist, the same file path is
    returned. If it does exist, the user is asked to confirm overwriting
    or to add a suffix to the file name.

    :param file: Original path+filename to be checked.
    :return: Checked or modified filename.
    """
    new_file = file
    if os.path.isfile(file):
        k = input(f"{file} already exists, do you want"
                  + " to overwrite it? (y/n): ")
        filename = file.split(".")[-2]
        extension = file.split(".")[-1]
        suffix = ""
        while k.lower() != 'y' and suffix == "":
            suffix = input("Write a new suffix to add to the file: ")
            new_file = f'{filename}-{suffix}.{extension}'
            if os.path.isfile(new_file):
                print("Suffix is already in use.")
                suffix = ""
    return new_file


# TODO: Change Tests from TestSyntheticDiagram to TestUtils
def overlaps(b1, b2):
    """
    Checks if two boxes given by their upper left and lower right corners
    are overlapping.

    :param b1: Box denoted by a dictionary with at least the keys ulx, uly
     (upper left) and lrx, lry (lower right).
    :param b2: Box denoted by a dictionary with at least the keys ulx, uly
     (upper left) and lrx, lry (lower right).
    :return: True if the two boxes overlap, False if not.
    """
    b1_over_b2 = b1["lry"] < b2["uly"]
    b2_over_b1 = b2["lry"] < b1["uly"]
    b1_right_b2 = b1["ulx"] > b2["lrx"]
    b2_right_b1 = b2["ulx"] > b1["lrx"]
    return not (b1_over_b2 or b2_over_b1 or b1_right_b2 or b2_right_b1)


def int_mean(*args):
    """
    Calculates the mean of the numbers passed as arguments, casting the result
    as an integer.

    :param args: Numbers to be averaged.
    :return: Integer casting of the mean of the numbers.
    """
    return np.mean(args, dtype=np.int32)


def get_box_center(box):
    """
    Gets the center (x, y) of a box.

    :param box: Box denoted by a dictionary with at least the keys ulx, uly
     (upper left) and lrx, lry (lower right).
    :return: A tuple of integers (x_center, y_center).
    """
    return (int_mean(box["ulx"], box["lrx"]),
            int_mean(box["uly"], box["lry"]))


def get_farthest_point_from(origin, points):
    """
    Selects the farthest point from the origin given a list of points.
    Take into account that both the origin and the points in the points list
    must share the coordinates order (x,y or y,x).

    :param origin: Point to calculate distances from.
    :param points: List of points where the farthest to origin will be selected
     from.
    :return: The point from the points list as declared in said list with the
     largest distance to the origin point.
    """
    max_distance_point = origin
    max_distance = 0
    for point in points:
        distance = np.linalg.norm(origin-point)
        if distance > max_distance:
            max_distance = distance
            max_distance_point = point
    return max_distance_point


def get_pixels_coords(image):
    """
    Given a binary image, returns a list of the coordinates of every non-zero
    pixel in said image. Positions are given in (x, y) coordinates.

    :param image: Binary images to extract coordinates from.
    :return: List of positions of non-zero pixels in (x, y) format.
    """
    white_points = np.argwhere(image)
    return [np.flip(c[:2]) for c in white_points]


def get_angle_two_points(p1, p2, out_format='decimal'):
    """
    Calculates the angle between a line traced from p1 to p2 and the x axis.
    This function considers the x-axis to increment horizontally and the y-axis
    to increment downwards.

    :param p1: Origin point (x, y).
    :param p2: End point (x, y).
    :param out_format: Format of the result, it can be 'decimal' or 'radian'.
     Default is 'decimal'.
    :return: Angle in the specified format.
    """
    valid_formats = {'decimal', 'radian'}
    if out_format not in valid_formats:
        raise ValueError(f"Format must be one of {valid_formats}")
    delta_x = p2[0] - p1[0]
    delta_y = p1[1] - p2[1]
    angle_rad = math.atan2(delta_y, delta_x)
    if out_format == 'decimal':
        return angle_rad * 180 / math.pi
    elif out_format == 'radian':
        return angle_rad


def get_element_box_dict(element_img, x, y, element_id, corner=None):
    """
    Generates a box dictionary from the specified information. The point
    defined by the (x, y) parameters will match the specified corner.

    :param element_img: Image of the element to be described by the dictionary.
    :param x: Horizontal coordinate of the desired position.
    :param y: Vertical coordinate of the desired position.
    :param element_id: Id of the element to be described by the dictionary.
    :param corner: Corner that the image will take as actual origin for the
     image. This corner must follow the format:
         (0, 0)----------(1, 0)
           |                |
           |                |
           |                |
           |                |
           |                |
         (0, 1)----------(1, 1)
     (NOTE: NO CHECK IS PERFORMED BY THIS FUNCTION TO VERIFY THAT THE ELEMENT
      IMAGE WILL FIT IN THE DIAGRAM IMAGE WHEN ITS COORDINATES ARE MODIFIED).
    :return: Box dictionary with the keys "ulx", "uly", "lrx", lry" and "id"
     corresponding to the final upper left and lower right coordinate where
     the image should be placed.
    """
    if corner is None:
        corner = [0, 0]
    return {"ulx": x - element_img.shape[1] * corner[0],
            "uly": y - element_img.shape[0] * corner[1],
            "lrx": x + element_img.shape[1] * (1 - corner[0]),
            "lry": y + element_img.shape[0] * (1 - corner[1]),
            "id": element_id}


def match_connection_img_to_points(image, connection_id, p1, p2):
    """
    Transforms a connection image where the connection is placed horizontally
    to fit between two given points connecting them.

    :param image: Image with the connection (Horizontally placed).
    :param connection_id: Id of the connection being modified.
    :param p1: Origin point where the connection origin will be placed.
    :param p2: End point where the connection end will be placed.
    :return: Tuple containing (the box dictionary defining the connection and
     its position, the modified connection image).
    """
    # Get distance between points
    distance = np.linalg.norm(p2 - p1)
    # Get angle between points
    angle = get_angle_two_points(p1, p2)
    # Get corner of the connection
    corner = get_connection_image_corner(angle)

    # Scale connection to match distance in the x-axis
    new_shape = (int(distance), image.shape[0])
    connection_img = cv2.resize(image, new_shape,
                                interpolation=cv2.INTER_NEAREST_EXACT)
    # Rotate image to match the angle between points
    # TODO: Test the speed taking into account the conversion from np to PIL
    connection_img_PIL = Image.fromarray(connection_img)
    rotated_PIL = connection_img_PIL.rotate(angle, expand=True)
    connection_img = np.array(rotated_PIL)
    # Generate the element box taking into account the selected corner
    connection_box = get_element_box_dict(connection_img,
                                          *p1,
                                          connection_id,
                                          corner)
    return connection_box, connection_img


def get_connection_image_corner(decimal_angle):
    """
    Gets the corner that should be superposed with the origin of a
    connection image to ensure that the rotation fits between the points.
    Corners follow the convention:
        (0, 0)----------(1, 0)
          |                |
          |                |
          |                |
          |                |
          |                |
        (0, 1)----------(1, 1)
    This corner election ensures that, for example, a horizontal connection
    that has been rotated 45 degrees, should have its origin in the lower
    left corner instead of the upper left to ensure that the connection line
    is not misplaced.

    :param decimal_angle: Decimal angle in the range (-180, 180]
    :return: Corner as specified above corresponding to the origin corner of
     the rotated connection.
    """
    if 0 < decimal_angle <= 90:
        return [0, 1]
    elif 90 < decimal_angle <= 180:
        return [1, 1]
    elif -90 < decimal_angle <= 0:
        return [0, 0]
    elif -180 < decimal_angle <= -90:
        return [1, 0]


if __name__ == "__main__":
    print(get_element_paths("shapes"))
    print(get_element_paths("shapes", include_suffixes=["ellipse"]))
    print(get_element_paths("shapes", exclude_suffixes=["ellipse"]))
    img = cv2.imread("tests/test_inter.png", cv2.IMREAD_GRAYSCALE)
    res = scale_image(img, 250)
    cv2.imshow("test", res)
    cv2.waitKey(0)
