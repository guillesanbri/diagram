import collision_detector as cd
import numpy as np
import utils
import hashlib
import cv2


# TODO: Randomize rotation of shapes before placing them (Care with annotations!).
# TODO: Add text and connections to annotations.
# TODO: Test this class.
class SyntheticDiagram:
    """
    Object representing one single synthetic diagram and its parameters.
    This class is intended to be used from a DiagramGenerator instance.
    """
    def __init__(self, shapes_paths, connections_paths, texts_paths,
                 output_shape, min_shape, n_shapes, max_placement_iter=5,
                 shape_size_rng_range=1.5, seed=None, debug=False):
        """
        Initializes an instance of a SyntheticDiagram

        :param shapes_paths: List of paths pointing to every shape available
         to build an artificial diagram.
        :param connections_paths: List of paths pointing to every connection
         available to build an artificial diagram.
        :param texts_paths: List of paths pointing to every text available
         to build an artificial diagram.
        :param output_shape: Shape (h, w) of the output image (diagram).
        :param min_shape: Minimum shape (h, w) of the actual diagram to be
         constructed inside the output image.
        :param n_shapes: Number of shapes to be randomly chosen and placed
         into the generated diagram.
        :param max_placement_iter: Maximum number of iterations to try to
         place each shape without overlapping with any previously placed shape.
        :param shape_size_rng_range: Factor which defines the upper limit of
         the size randomization that takes place in the __randomize_shape_size
         method.
        :param seed: Seed to the random number generator used to generate the
         diagram.
        :param debug: Mode of execution, setting this parameter to True
         will draw the actual diagram area inside the output image.
        """
        # Random number generator
        self.rng = np.random.default_rng(seed)
        # Output image and annotation
        self.output_shape = output_shape
        self.output_img = np.zeros(self.output_shape, dtype=np.uint8)
        # Randomized diagram parametrization (rparam)
        self.rheight = self.rng.integers(min_shape[0], self.output_shape[0])
        self.rwidth = self.rng.integers(min_shape[1], self.output_shape[1])
        self.x0 = (self.output_shape[1] - self.rwidth) // 2
        self.y0 = (self.output_shape[0] - self.rheight) // 2
        # Paths
        self.shapes_paths = shapes_paths
        self.connections_paths = connections_paths
        self.texts_paths = texts_paths
        # Shapes
        self.n_shapes = n_shapes
        self.max_placement_iter = max_placement_iter
        self.placed_shapes = []  # [{"ulx", "uly", "lrx", "lry", "id"}, ...]
        self.max_shape_size = self.calculate_max_shape_size()
        self.shape_size_rng_range = shape_size_rng_range
        # Mode
        self.debug = debug

    def calculate_max_shape_size(self):
        """
        Calculates the maximum size of a shape as a function of
        the minimum size of the diagram. This function may be modified.

        :return: Maximum size of the shapes.
        """
        min_side = np.min([self.rwidth, self.rheight])
        return int(min_side / self.n_shapes * 2)

    def get_name(self):
        """
        Generates a unique and deterministic filename for a generated diagram.
        :return: Filename with schema 'synth{md5}.png'
        """
        diagram_md5 = hashlib.md5(self.output_img.tobytes()).hexdigest()
        return f'synth{diagram_md5}.png'

    def randomize_shape_size(self, size):
        """
        Modifies a given size to be randomized between said value and an upper
        limit given by the given size multiplied by self.shape_size_rng_range.

        :param size: Initial size to be modified.
        :return: New size randomized in the range
         [size, self.shape_size_rng_range * size)
        """
        lower_limit = size  # This could be modified
        upper_limit = int((1 + self.shape_size_rng_range) * size)
        return self.rng.integers(lower_limit, upper_limit)

    # TODO: Write this method.
    # TODO: Test behaviour of the model with and without rotations.
    def randomize_shape_rotation(self):
        pass

    def randomize_shape_location(self, shape_shape):
        """
        Generates a randomized point (x,y) taking into account the actual
        diagram area and the size of the element (shape) being placed. The
        generated coordinates ensure that the placed shape will be within
        the randomized limits of the diagram.

        :param shape_shape: Shape (h, w) of the shape being placed into
         the diagram.
        :return: Tuple (xs, ys) containing the randomized coordinates.
        """
        xs0 = self.rng.integers(self.x0, self.x0 + self.rwidth - shape_shape[1])
        ys0 = self.rng.integers(self.y0, self.y0 + self.rheight - shape_shape[0])
        return xs0, ys0

    def place_element(self, x, y, image, box_dict):
        """
        Places the element defined by an image and a box_dict into the output
        image and appends its position to self.placed_shapes.

        :param x: Coordinate of the point of the output image where the upper
         left corner of the element will be placed.
        :param y: Coordinate of the point of the output image where the upper
         left corner of the element will be placed.
        :param image: OpenCV image (np.array) containing the element that
         is going to be placed.
        :param box_dict: Box denoted by a dictionary with at least the keys
         ulx, uly (upper left); lrx, lry (lower right) and id (element_id).
        """
        try:
            required_keys = ["ulx", "uly", "lrx", "lry", "id"]
            for key in required_keys:
                _ = box_dict[key]
        except KeyError:
            raise KeyError("box_dict does not have the required keys "
                           "to be stored")
        # Place the shape
        self.output_img[y:y + image.shape[0], x:x + image.shape[1]] |= image
        # Store the placed element
        self.placed_shapes.append(box_dict)

    def try_to_place_shape(self, shape_img, element_id):
        """
        Tries to place a given shape image into the output image taking into
        account the already placed shapes. If successful, it places the shape
        calling the place_element() method.

        :param shape_img: OpenCV image (np.array) containing the shape that
         is going to be placed.
        :param element_id: Id of the shape contained in shape_img.
        """
        # Try to fit the shape n times, break loop if placed.
        for i in range(self.max_placement_iter):
            # Randomize placement of the shape.
            xs0, ys0 = self.randomize_shape_location(shape_img.shape)
            new_shape_position = {"ulx": xs0,
                                  "uly": ys0,
                                  "lrx": xs0 + shape_img.shape[1],
                                  "lry": ys0 + shape_img.shape[0],
                                  "id": element_id}
            # Check for overlapping.
            overlapping = False
            for shape_position in self.placed_shapes:
                if utils.overlaps(shape_position, new_shape_position):
                    overlapping = True
                    break
            if not overlapping:
                self.place_element(xs0, ys0, shape_img, new_shape_position)
                break

    def __draw_randomized_limits(self):
        """
        Draws a rectangle in the output image signaling the area occupied
        by the real diagram (randomized at initialization).
        """
        cv2.rectangle(self.output_img, (self.x0, self.y0),
                      (self.x0 + self.rwidth, self.y0 + self.rheight),
                      color=255, thickness=3)

    def get_annotation(self):
        """
        Gets a string where the boxes placed into a diagram are annotated.
        These boxes follow the schema:
        x_min,y_min,x_max,y_max,element_id box2 ...

        :return: Annotation with as many boxes as objects have been placed.
        """
        annotation = ""
        for shape in self.placed_shapes:
            x_min = shape["ulx"]
            y_min = shape["uly"]
            x_max = shape["lrx"]
            y_max = shape["lry"]
            element_id = shape["id"]
            annotation += f"{x_min},{y_min},{x_max},{y_max},{element_id} "
        return annotation[:-1]

    def load_random_from(self, paths):
        """
        Chooses a random path from a list of image paths and reads the
        corresponding image. Path must be in the format
        ?????-???-???-{_id}-???-???.png as the element id is retrieved
        from the path.

        :param paths: List of paths of images.
        :return: Tuple (element id, OpenCV image) from the chosen image. Id is the
         corresponding element id extracted from the filename.
        :raises ValueError: If the path argument is empty.
        """
        if paths is None:
            raise ValueError("paths argument can not be an empty list.")
        chosen_path = self.rng.choice(paths)
        chosen_img = cv2.imread(chosen_path, cv2.IMREAD_GRAYSCALE)
        element_id = chosen_path.split("/")[-1].split(".")[0].split("-")[3]
        return element_id, chosen_img

    def add_shapes(self):
        """
        Parametrizes as many shapes as indicated in self.n_shapes and tries
        to place them into the output image.
        """
        for shape_index in range(self.n_shapes):
            # Choose a random shape and read it.
            element_id, shape_img = self.load_random_from(self.shapes_paths)
            # Randomize its maximum shape and scale it.
            randomized_size = self.randomize_shape_size(self.max_shape_size)
            shape_img = utils.scale_image(shape_img, randomized_size)
            # Randomize its rotation and rotate it
            pass
            # Try to place the shape
            self.try_to_place_shape(shape_img, element_id)

    # TODO: Refactor this method
    def add_connections(self):
        """
        Generates random connections between the shapes that have been placed
        into the output image.
        """
        # Generate lines from center to center
        # Get intersections - Convex Hull of the points of each shape?
        # Draw lines from intersection point to intersection point
        blank0 = np.zeros(self.output_shape, dtype=np.uint8)
        lines = blank0.copy()
        for i, s1 in enumerate(self.placed_shapes):
            for s2 in self.placed_shapes[i+1:]:
                if s1 is not s2:
                    # Get real origins
                    blank1 = np.zeros(self.output_shape, dtype=np.uint8)
                    blank2 = np.zeros(self.output_shape, dtype=np.uint8)
                    roi1 = self.output_img[s1["uly"]:s1["lry"], s1["ulx"]:s1["lrx"]].copy()
                    roi2 = self.output_img[s2["uly"]:s2["lry"], s2["ulx"]:s2["lrx"]].copy()
                    blank1[s1["uly"]:s1["lry"], s1["ulx"]:s1["lrx"]] = roi1
                    blank2[s2["uly"]:s2["lry"], s2["ulx"]:s2["lrx"]] = roi2
                    c1 = utils.get_box_center(s1)
                    c2 = utils.get_box_center(s2)
                    line = cv2.line(blank0.copy(), c1, c2, 255, 5)

                    intersection_img1 = blank1 & line
                    intersection_points1 = [np.flip(c[:2]) for c in np.argwhere(intersection_img1)]
                    origin1 = cd.get_farthest_point_from(c1, intersection_points1)

                    intersection_img2 = blank2 & line
                    intersection_points2 = [np.flip(c[:2]) for c in np.argwhere(intersection_img2)]
                    origin2 = cd.get_farthest_point_from(c2, intersection_points2)

                    # Check for overlap between
                    additional_intersections = (self.output_img & line) - intersection_img1 - intersection_img2
                    if np.all((additional_intersections == 0)):  # No intersections with other shapes
                        lines = cv2.line(lines, origin1, origin2, 255, 2)
        self.output_img |= lines
        cv2.imshow("output", self.output_img)
        cv2.waitKey(0)

    def generate(self):
        """
        Generates the output image containing an artificial diagram composed
        by a set of the available elements.

        :return: A tuple containing (Annotation for the generated image in
         format box1 box2 box3..., Output image with the generated diagram).
         Box annotation follow the schema min_x,min_y,max_x,max_y,element_id
         where the element_id represents the id in the element_id.json, NOT
         a detection class id.
        """
        if self.debug:
            self.__draw_randomized_limits()

        self.add_shapes()
        self.add_connections()
        # self.add_texts()

        return self.get_annotation(), self.output_img
