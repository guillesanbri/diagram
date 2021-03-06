from DatasetGenerator.ConnectionsManager import ConnectionsManager
import DatasetGenerator.utils.utils as utils
import numpy as np
import hashlib
import cv2


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
        self.instance_masks = {}
        self.instance_masks_counter = {}
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

    def randomize_flip(self, image):
        """
        Randomly flips an image. The outcomes can be no flip, horizontal flip,
        vertical flip, or both at the same time.

        :param image: Image that may be flipped.
        :return: Modified or unmodified copy of the specified image.
        """
        options = [None, -1, 0, 1]
        random_decision = self.rng.choice(options)
        if random_decision is not None:
            return cv2.flip(image.copy(), random_decision)
        else:
            return image.copy()

    def place_element(self, image, box):
        """
        Places the element defined by an image and a box into the output
        image and appends its position to self.placed_shapes. This function
        generates an instance mask every time an element is placed.

        :param image: OpenCV image (np.array) containing the element that
         is going to be placed.
        :param box: Box denoted by a dictionary with at least the keys
         ulx, uly (upper left); lrx, lry (lower right) and id (element_id).
        """
        try:
            required_keys = ["ulx", "uly", "lrx", "lry", "id"]
            for key in required_keys:
                _ = box[key]
        except KeyError:
            raise KeyError("box_dict does not have the required keys "
                           "to be stored")
        # Place the shape
        self.output_img[box["uly"]:box["lry"], box["ulx"]:box["lrx"]] |= image
        # Store the placed element
        self.placed_shapes.append(box)
        # Generate the corresponding mask
        self.generate_mask(image, box)

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
            new_shape_position = utils.get_element_box_dict(shape_img, xs0, ys0, element_id)
            # Check for overlapping.
            overlapping = False
            for shape_position in self.placed_shapes:
                if utils.overlaps(shape_position, new_shape_position):
                    overlapping = True
                    break
            if not overlapping:
                self.place_element(shape_img, new_shape_position)
                break

    def generate_mask(self, element_image, box):
        """
        Generates an image with the mask of the element defined by the passed
        parameters or updates an existing one to give the pixels corresponding
        to another instance of the same class. This image is allocated in the
        self.instance_masks dictionary with key = element_id. Pixel values in
        the masks range from 0 (background) to num_instances in each class.

        :param element_image: Image of the element that is going to be labelled
         in the mask.
        :param box: Box defining the position and element_id of the
         element_image.
        """
        if box["id"] in self.instance_masks:
            # If the element already has a mask where instances are being drawn
            class_mask = self.instance_masks[box["id"]].copy()
            pixel_value = self.instance_masks_counter[box["id"]] + 1
            if pixel_value >= 255:
                raise ValueError("Too many instances of the same class.")
        else:
            # If the element does not yet have a mask
            class_mask = np.zeros(self.output_shape, dtype=np.uint8)
            pixel_value = 1
        y0, y1, x0, x1 = box["uly"], box["lry"], box["ulx"], box["lrx"]
        # Assign a different value to each instance and deal with overlapping
        class_mask[y0:y1, x0:x1] |= (element_image // 255 * pixel_value)
        class_mask[class_mask >= pixel_value] = pixel_value
        # Update the new mask and the new class instance counter
        self.instance_masks[box["id"]] = class_mask
        self.instance_masks_counter[box["id"]] = pixel_value

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
            # Randomly flip the image
            shape_img = self.randomize_flip(shape_img)
            # Randomize its maximum shape and scale it.
            randomized_size = self.randomize_shape_size(self.max_shape_size)
            shape_img = utils.scale_image(shape_img, randomized_size)
            # Randomize its rotation and rotate it
            pass
            # Try to place the shape
            self.try_to_place_shape(shape_img, element_id)

    def add_connections(self, connection_dropout=0.2):
        """
        Generates random connections between the shapes that have been placed
        into the output image.

        :param connection_dropout: Percentage of the connections that will be
         discarded and therefore won't appear in the resulting diagram. Default
         value is 0.2 (20% of connections are dropped).
        """
        cm = ConnectionsManager(synthetic_diagram=self)
        # Get pairs of valid points between shapes.
        points = cm.get_valid_points()
        connection_dropout_prob = 1 - connection_dropout
        for point_pair in points:
            if self.rng.random() <= connection_dropout_prob:
                p1, p2 = point_pair
                # Load random connection
                connection_id, connection_img = self.load_random_from(self.connections_paths)
                # Randomly flip the image
                connection_img = self.randomize_flip(connection_img)
                # Transform the connection image to match a line between the
                # two points and get its box.
                connection_box, connection_img = utils.match_connection_img_to_points(connection_img,
                                                                                      connection_id,
                                                                                      p1, p2)
                # Place the connection into the diagram image
                self.place_element(connection_img, connection_box)

    def generate(self):
        """
        Generates the output image containing an artificial diagram composed
        by a set of the available elements.

        :return: Output image with the generated diagram.
        """
        if self.debug:
            self.__draw_randomized_limits()

        self.add_shapes()
        self.add_connections()
        # self.add_texts()

        return self.output_img

    def get_boxes_annotations(self):
        """
        Gets the boxes annotations from a fully generated synthetic diagram.
        Each box annotation follows the schema:
        min_x,min_y,max_x,max_y,element_id

        :return: Annotation for the generated image in format
         box1 box2 box3...
        """
        return self.get_annotation()

    def get_instances_annotations(self):
        """
        Gets a dictionary of arrays where each key correspond to an element_id
        where each value is an image whose pixels denote different instances
        of said element. Each instance is labeled with a unique pixel value
        from 1 to number_instances (zero is background).

        :return: Dictionary of element_ids and instance label images.
        """
        return self.instance_masks
