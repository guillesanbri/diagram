from SyntheticDiagram import SyntheticDiagram
import numpy as np
import utils
import cv2


# TODO: Documentation of this class
# TODO: Tests of this class
# TODO: Generate report of the configuration of each generated dataset
# TODO: Add loading of shapes, text, and connections correctly.
# TODO: Pad to square in the preprocessing function to new inputs.
class DiagramGenerator:
    """
    Object to create a set of synthetic parametrized diagrams.
    These diagrams are annotated with bounding boxes based on
    the placed elements.
    """
    def __init__(self, output_path, quantity,
                 output_shape=(1024, 1024), min_shape=(340, 340),
                 max_placement_iter=5, seed=None, debug=False):
        """
        Initializes a instance of a DiagramGenerator.

        :param output_path: Path of the directory where the generated diagrams
         will be saved.
        :param quantity: Number of diagrams to generate.
        :param output_shape: Shape (h, w) of the generated images.
        :param min_shape: Minimum shape (h, w) of the actual diagram
         inside the output image.
        :param max_placement_iter: Maximum number of attempts to place a
         shape in a diagram without overlapping.
        :param seed: Seed for the random number generator.
        :param debug: Mode of execution, setting this parameter to True
         will make the seeds of each diagram non random.
        """
        # Paths
        self.output_path = output_path
        # Diagram parametrization
        # TODO: Add warning if min dimensions are not between a percentage of output shape
        self.quantity = quantity
        self.output_shape = output_shape  # (h, w)
        self.min_shape = min_shape  # (h, w)
        self.max_placement_iter = max_placement_iter
        self.debug = debug
        self.rng = np.random.default_rng(seed)

    def run(self, shapes_paths, connections_paths, texts_paths):
        """
        Initiates the generation of the diagrams. This method parameters allow
        to configure the set of shapes, connections and texts passed to the
        SyntheticDiagram object.

        :param shapes_paths: Array of the paths to each valid shape image to use
         during diagram generation.
        :param connections_paths: Array of the paths to each valid connection
         image to use during diagram generation.
        :param texts_paths: Array of the paths to each valid text image to use
         during diagram generation.
        :return: annotations?
        """
        annotations = None
        for i in range(self.quantity):
            seed = None
            if self.debug:
                seed = self.rng.integers(self.quantity*100)
            synth_diagram = SyntheticDiagram(shapes_paths,
                                             None,
                                             None,
                                             self.output_shape,
                                             self.min_shape,
                                             10, seed=seed)
            annotation, diagram = synth_diagram.generate()
            cv2.imwrite(self.output_path + synth_diagram.get_name(), diagram)
        return annotations


if __name__ == "__main__":
    shapes_paths = utils.get_shapes_paths("elements/")
    dg = DiagramGenerator("diagrams/", 10, seed=42, debug=True)
    # TODO: Include annotations saving into run method?
    annotations = dg.run(shapes_paths, None, None)
    # dg.save_annotation_as(annotations, "VOC")  # Convert to typical bbox annotation formats.
