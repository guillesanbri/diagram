from SyntheticDiagram import SyntheticDiagram
import numpy as np
import cv2
import glob


# TODO: Documentation of this class
# TODO: Documentation of the utils functions
# TODO: Tests of this class
# TODO: Generate report of the configuration of each generated dataset
# TODO: Add loading of shapes, text, and connections correctly.
# TODO: Pad to square in the preprocessing function to new inputs.
class DiagramGenerator:
    def __init__(self, elements_path, output_path, quantity,
                 output_shape=(1024, 1024), min_shape=(340, 340),
                 max_placement_iter=5, seed=None, debug=False):
        # Paths
        self.output_path = output_path
        # Elements paths
        self.texts = None
        self.connections = None
        self.elements_path = elements_path
        # TODO: Move this function to a utils file where it loads the paths after reading the id files.
        self.shape_paths = glob.glob(self.elements_path + f'?????-???-???-???-???-???.png')
        # Diagram parametrization
        # TODO: Add warning if min dimensions are not between a percentage of output shape
        self.quantity = quantity
        self.output_shape = output_shape  # (h, w)
        self.min_shape = min_shape
        self.max_placement_iter = max_placement_iter
        self.debug = debug
        self.rng = np.random.default_rng(seed)

    def run(self, shapes_paths, connection_paths, text_paths):
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
    dg = DiagramGenerator("elements/", "diagrams/", 10, seed=42, debug=True)
    # TODO: Include annotations saving into run method?
    annotations = dg.run(dg.shape_paths, None, None)
    # dg.save_annotation_as(annotations, "VOC")  # Convert to typical bbox annotation formats.
