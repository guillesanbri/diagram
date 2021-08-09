import numpy as np
import utils
import hashlib
import cv2


# TODO: Randomize rotation of shapes before placing them (Care with annotations!).
class SyntheticDiagram:
    def __init__(self, shapes_paths, connections_paths, texts_paths,
                 output_shape, min_shape, n_shapes, max_placement_iter=5,
                 shape_size_rng_range=1.5, seed=None, debug=False):
        # Random number generator
        self.rng = np.random.default_rng(seed)
        # Output image and annotation
        self.output_shape = output_shape
        self.output_img = np.zeros(self.output_shape, dtype=np.uint8)
        # TODO: Make annotations
        self.annotation = None
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
        # [{"ulx", "uly", "lrx", "lry"}, {"ulx", "uly", "lrx", "lry"}, ...]
        self.placed_shapes = []
        self.shape_size_rng_range = shape_size_rng_range
        # Mode
        self.debug = debug

    def __randomize_shape_size(self, size):
        lower_limit = size  # This could be modified
        upper_limit = int((1 + self.shape_size_rng_range) * size)
        return self.rng.integers(lower_limit, upper_limit)

    def __randomize_shape_location(self, shape_shape):
        xs0 = self.rng.integers(self.x0, self.x0 + self.rwidth - shape_shape[1])
        ys0 = self.rng.integers(self.y0, self.y0 + self.rheight - shape_shape[0])
        return xs0, ys0

    @staticmethod
    def overlaps(b1, b2):
        b1_over_b2 = b1["lry"] < b2["uly"]
        b2_over_b1 = b2["lry"] < b1["uly"]
        b1_right_b2 = b1["ulx"] > b2["lrx"]
        b2_right_b1 = b2["ulx"] > b1["lrx"]
        return not (b1_over_b2 or b2_over_b1 or b1_right_b2 or b2_right_b1)

    def __place_element_into_output_img(self, shape_img):
        # Try to fit the shape n times, break loop if placed.
        for i in range(self.max_placement_iter):
            # Randomize placement of the shape.
            xs0, ys0 = self.__randomize_shape_location(shape_img.shape)
            new_shape_position = {"ulx": xs0,
                                  "uly": ys0,
                                  "lrx": xs0 + shape_img.shape[1],
                                  "lry": ys0 + shape_img.shape[0]}
            # Check for overlapping.
            overlapping = False
            for shape_position in self.placed_shapes:
                if self.overlaps(shape_position, new_shape_position):
                    overlapping = True
                    break
            if not overlapping:
                self.output_img[ys0:ys0 + shape_img.shape[0],
                                xs0:xs0 + shape_img.shape[1]] |= shape_img
                self.placed_shapes.append(new_shape_position)
                break

    def __draw_randomized_limits(self):
        cv2.rectangle(self.output_img, (self.x0, self.y0),
                      (self.x0 + self.rwidth, self.y0 + self.rheight),
                      color=255, thickness=3)

    def generate(self):
        if self.debug:
            self.__draw_randomized_limits()

        # Define the maximum size of a shape as a function of the minimum size of the diagram.
        # This could be modified.
        min_side = np.min([self.rwidth, self.rheight])
        max_shape_size = int(min_side / self.n_shapes * 2)

        for shape_index in range(self.n_shapes):
            # Choose a random shape and read it.
            shape_path = self.rng.choice(self.shapes_paths)
            shape_img = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
            # Randomize its maximum shape and scale it.
            randomized_size = self.__randomize_shape_size(max_shape_size)
            shape_img = utils.scale_image(shape_img, randomized_size)
            # Try to place the shape
            self.__place_element_into_output_img(shape_img)
        return self.annotation, self.output_img

    def get_name(self):
        return f'synth{hashlib.md5(self.output_img.tobytes()).hexdigest()}.png'
