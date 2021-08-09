import numpy as np
import glob
import hashlib
import cv2


# TODO: Documentation of this class
# TODO: Documentation of the utils functions
# TODO: Tests of this class
import utils


class DiagramGenerator:
    def __init__(self, elements_path, output_path, quantity,
                 output_shape=(1024, 1024), min_height=340, min_width=340,
                 max_placement_iter=5, seed=None):
        # Paths
        self.elements_path = elements_path
        self.output_path = output_path
        # Elements paths
        self.texts = None
        # TODO: Move this function to a utils file where it loads the paths after reading the id files.
        self.shape_paths = glob.glob(self.elements_path + f'?????-???-???-???-???-???.png')
        self.connections = None

        # Diagram parametrization
        # TODO: Add warning if min dimensions are not between a percentage of output shape
        self.quantity = quantity
        self.output_shape = output_shape  # (h, w)
        self.min_height = min_height
        self.min_width = min_width
        self.max_placement_iter = max_placement_iter
        # TODO: Tune this.
        self.shape_size_rng_range = 1.5
        self.rng = np.random.default_rng(seed)

    def __randomize_shape_size(self, size):
        # TODO: Tune this
        lower_limit = int((1 - self.shape_size_rng_range) * size)
        lower_limit = size
        upper_limit = int((1 + self.shape_size_rng_range) * size)
        return self.rng.integers(lower_limit, upper_limit)

    def __randomize_shape_location(self, x0, y0, diagram_shape, shape_shape):
        xs0 = self.rng.integers(x0, x0 + diagram_shape[1] - shape_shape[1])
        ys0 = self.rng.integers(y0, y0 + diagram_shape[0] - shape_shape[0])
        return xs0, ys0

    @staticmethod
    def overlaps(b1, b2):
        b1_over_b2 = b1["lry"] < b2["uly"]
        b2_over_b1 = b2["lry"] < b1["uly"]
        b1_right_b2 = b1["ulx"] > b2["lrx"]
        b2_right_b1 = b2["ulx"] > b1["lrx"]
        return not (b1_over_b2 or b2_over_b1 or b1_right_b2 or b2_right_b1)

    def __place_element(self, diagram, element, x, y):
        temp = diagram.copy()
        temp[y:y + element.shape[0], x:x + element.shape[1]] |= element
        return temp

    def __generate_one(self, n_shapes):
        annotation = None
        # Parametrize real size of the diagram.
        diagram = np.zeros(self.output_shape, dtype=np.uint8)
        diagram_height = self.rng.integers(self.min_height, self.output_shape[0])
        diagram_width = self.rng.integers(self.min_width, self.output_shape[1])
        diagram_shape = (diagram_height, diagram_width)
        output_area = self.output_shape[0]*self.output_shape[1]
        real_area = diagram_height*diagram_width
        area_percentage = real_area / output_area

        # DEBUG
        x0 = (self.output_shape[1] - diagram_width)//2
        y0 = (self.output_shape[0] - diagram_height)//2
        cv2.rectangle(diagram, (x0, y0),
                      (x0 + diagram_width, y0 + diagram_height),
                      color=255, thickness=3)
        cv2.putText(diagram, f"Area percentage: {area_percentage*100:.2f}",
                    (x0+10, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=255)
        # DEBUG

        # Define the maximum size of a shape as a function of the minimum size of the diagram.
        # TODO: Maybe tune this.
        max_shape_size = int(np.min([diagram_width, diagram_height]) / n_shapes * 2)
        # Create empty list to store placed shapes upper left and lower right corners.
        # ie: [{"ulx", "uly", "lrx", "lry"}, {"ulx", "uly", "lrx", "lry"}, ...]
        shapes_positions = []
        for shape_index in range(n_shapes):
            # TODO: Randomize size of the shapes before placing them
            # TODO: Randomize rotation of shapes before placing them (Care with annotations!).
            # TODO: Add loading of shapes, text, and connections correctly.
            # TODO: Pad to square in the preprocessing function to new inputs.
            # TODO: Rename shape to something else to avoid confusion with .shape .
            # Choose a random shape and read it.
            shape_path = self.rng.choice(self.shape_paths)
            shape_img = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
            # Randomize its maximum shape and scale it.
            randomized_size = self.__randomize_shape_size(max_shape_size)
            shape_img = utils.scale_image(shape_img, randomized_size)

            # Try to fit the shape n times, breaks loop if placed.
            for i in range(self.max_placement_iter):
                # Randomize placement of the shape.
                xs0, ys0 = self.__randomize_shape_location(x0, y0,
                                                           diagram_shape,
                                                           shape_img.shape)
                new_shape_position = {"ulx": xs0,
                                      "uly": ys0,
                                      "lrx": xs0 + shape_img.shape[1],
                                      "lry": ys0 + shape_img.shape[0]}
                # Check for overlapping.
                overlapping = False
                for shape_position in shapes_positions:
                    if self.overlaps(shape_position, new_shape_position):
                        overlapping = True
                        break
                if not overlapping:
                    diagram = self.__place_element(diagram, shape_img, xs0, ys0)
                    shapes_positions.append(new_shape_position)
                    break
        return annotation, diagram

    def run(self):
        # TODO: Make annotations
        annotations = None
        for i in range(self.quantity):
            annotation, diagram = self.__generate_one(n_shapes=10)
            cv2.imwrite(self.output_path + f'gen{hashlib.md5(diagram.tobytes()).hexdigest()}.png', diagram)
        return annotations


if __name__ == "__main__":
    dg = DiagramGenerator("elements/", "diagrams/", 10, seed=42)
    # TODO: Include annotations saving into run method?
    annotations = dg.run()
    # dg.save_annotation_as(annotations, "VOC")  # Convert to typical bbox annotation formats.
