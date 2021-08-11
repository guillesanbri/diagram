from SyntheticDiagram import SyntheticDiagram
from tqdm import tqdm
import numpy as np
import warnings
import utils
import json
import cv2


# TODO: Tests of this class
# TODO: Randomize number of shapes in each diagram
# TODO: Generate report of the configuration of each generated dataset
# TODO: Add loading of shapes, text, and connections correctly.
# TODO: Pad to square in the preprocessing function to new inputs.
# TODO: Add estimation of the size of the dataset.
class DiagramGenerator:
    """
    Object to create a set of synthetic parametrized diagrams.
    These diagrams are annotated with bounding boxes based on
    the placed elements.
    """
    def __init__(self, output_path, quantity, annotation_file="annotation.txt",
                 output_shape=(1024, 1024), min_shape=(340, 340),
                 max_placement_iter=5, seed=None, debug=False):
        """
        Initializes a instance of a DiagramGenerator.

        :param output_path: Path of the directory where the generated diagrams
         will be saved.
        :param quantity: Number of diagrams to generate.
        :param annotation_file: File where annotations will be saved.
         If the file it already exists the program will ask if
         overwriting is desired.
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
        self.annotation_file = annotation_file
        # Diagram parametrization
        # TODO: Add warning if min dimensions are not between a percentage of output shape
        self.quantity = quantity
        self.output_shape = output_shape  # (h, w)
        self.min_shape = min_shape  # (h, w)
        self.max_placement_iter = max_placement_iter
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        # Annotations
        self.annotations = []

    # TODO: Test if the dictionary can be modified to group all shapes ->
    # it can, but proper testing has to be done.
    def __translate_annotations(self, ids_suffixes):
        """
        Updates the self.annotations array to substitute each box id by
        a class id starting from 0. New boxes annotations will follow the next
        schema: min_x,min_y,max_x,max_y,class
        :param ids_suffixes: Dictionary of id as keys and the suffix as value.
        :return: A dictionary mapping each new id class to its corresponding
         element suffix. ie: new_class_id:element_suffix
        """
        translated_annotations = []
        # Create a dictionary of suffix:new_class_id
        distinct_suffixes = set(ids_suffixes.values())
        class_ids = [i for i in range(len(distinct_suffixes))]
        class_dict = dict(zip(distinct_suffixes, class_ids))
        for annotation in self.annotations:
            boxes = annotation.split(" ")[1:]
            boxes_elements = [box.split(",") for box in boxes]
            for box in boxes_elements:
                box[-1] = str(class_dict[ids_suffixes[box[-1]]])
            translated_boxes = [",".join(be) for be in boxes_elements]
            translated = " ".join(translated_boxes)
            translated_annotations.append(translated)
        self.annotations = translated_annotations
        return dict(zip(class_ids, ids_suffixes.values()))

    def __save_annotations(self):
        """
        Generates and saves an annotation file following the model:

        path_to_img1 box1 box2 box3 ...
        path_to_img2 box1 box2 box2 ...
        ...

        with each box defined as:

        x_min,y_min,

        """
        if self.annotations is None:
            warnings.warn("No annotations have been previously generated!")
        self.annotation_file = utils.check_file_path(self.annotation_file)
        with open(self.annotation_file, 'w') as f:
            f.write('\n'.join(self.annotations))

    def run(self, shapes_paths, connections_paths, texts_paths, classes):
        """
        Initiates the generation of the diagrams. This method parameters allow
        to configure the set of shapes, connections and texts passed to the
        SyntheticDiagram object. Calling this method generates diagrams in the
        self.output_path directory and an annotation file self.annotation_path.

        :param shapes_paths: Array of the paths to each valid shape image to
         use during diagram generation.
        :param connections_paths: Array of the paths to each valid connection
         image to use during diagram generation.
        :param texts_paths: Array of the paths to each valid text image to use
         during diagram generation.
        :param classes: Dictionary of id:element of every element to be placed
         in the generated diagrams.
        :return: Dictionary of class_id:element_suffix where class_id indicates
         each value annotated in the annotations file and element_suffix
         indicates its correspondent element as passed through the classes
         parameter of this method.
         If the classes param is {'xxx':'a', 'yyy':'b'}, the returned dict
         will be {0:'a', 1:'b'}. If the classes param is {'xxx':'c', 'yyy':'c'}
         the returned dict will be {0:'c'}.
        """
        for _ in tqdm(range(self.quantity)):
            seed = None
            if self.debug:
                seed = self.rng.integers(self.quantity*100)
            synth_diagram = SyntheticDiagram(shapes_paths,
                                             connections_paths,
                                             texts_paths,
                                             self.output_shape,
                                             self.min_shape,
                                             10, seed=seed)
            annotation, diagram = synth_diagram.generate()
            output_path = self.output_path + synth_diagram.get_name()
            annotation = f"{output_path} " + annotation
            self.annotations.append(annotation)
            cv2.imwrite(output_path, diagram)
        annotated_classes = self.__translate_annotations(classes)
        self.__save_annotations()
        return annotated_classes


if __name__ == "__main__":
    # Read shapes, connections and texts
    shapes_dict, shapes_paths = utils.get_shapes_paths("elements/")
    # Read connections
    # Read texts

    # Dictionary of id:element_suffix
    # TODO: Facilitate overwrite
    # Element tagging can be overwritten as follows
    # shapes_dict = {'400': 'shape', '600': 'shape', '800': 'shape'}
    ids_suffixes = {**shapes_dict}

    dg = DiagramGenerator("diagrams/", 10, seed=42, debug=True)
    annotation_classes = dg.run(shapes_paths, None, None, ids_suffixes)
    classes_json_path = utils.check_file_path('annotated_classes.json')
    with open(classes_json_path, 'w') as f:
        json.dump(annotation_classes, f)
    # dg.save_annotation_as(annotations, "VOC")  # Convert to typical bbox annotation formats.
