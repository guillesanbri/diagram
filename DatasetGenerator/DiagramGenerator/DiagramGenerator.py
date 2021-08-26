from DatasetGenerator.SyntheticDiagram import SyntheticDiagram
import DatasetGenerator.utils.utils as utils
from tqdm import tqdm
import numpy as np
import warnings
import cv2
import os


# TODO: Pad to square in the preprocessing function to new inputs.
class DiagramGenerator:
    """
    Object to create a set of synthetic parametrized diagrams.
    These diagrams are annotated with bounding boxes based on
    the placed elements.
    """
    def __init__(self, output_path, quantity, shape_n_range,
                 annotation_file="annotation.txt", output_shape=(1024, 1024),
                 min_shape=(340, 340), max_placement_iter=5, seed=None,
                 debug=False):
        """
        Initializes an instance of a DiagramGenerator.

        :param output_path: Path of the directory where the generated diagrams
         will be saved.
        :param quantity: Number of diagrams to generate.
        :param shape_n_range: Range [min, max) from where to sample the
         randomized number of shapes in each diagram.
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
        self.quantity = quantity
        self.shape_n_range = shape_n_range
        self.output_shape = output_shape  # (h, w)
        self.min_shape = min_shape  # (h, w)
        self.max_placement_iter = max_placement_iter
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        # Annotations
        self.detection_annotations = []

        # Create the output dir if it does not exist
        os.makedirs(self.output_path, exist_ok=True)
        # Create the masks dir if it does not exist
        os.makedirs(self.output_path + "masks", exist_ok=True)

        # Check the minimum number of shapes to put in the diagrams
        minimum = min(self.shape_n_range)
        if minimum < 5:
            raise ValueError("Minimum value of self.shape_n_range can not be"
                             "less than 5.")

    def translate_detection_annotations(self, ids_suffixes):
        """
        Updates the self.annotations array to substitute each box id by
        its element (class) suffix. New boxes annotations will follow the
        next schema: min_x,min_y,max_x,max_y,element_suffix

        :param ids_suffixes: Dictionary of id as keys and the suffix as value.
        """
        translated_annotations = []
        for annotation in self.detection_annotations:
            path = annotation.split(" ")[0]
            boxes = annotation.split(" ")[1:]
            boxes_elements = [box.split(",") for box in boxes]
            for box in boxes_elements:
                box[-1] = str(ids_suffixes[box[-1]])
            translated_boxes = [",".join(be) for be in boxes_elements]
            translated = " ".join(translated_boxes)
            translated = f"{path} " + translated
            translated_annotations.append(translated)
        self.detection_annotations = translated_annotations

    def save_detection_annotations(self):
        """
        Generates and saves an annotation file following the model:

        path_to_img1 box1 box2 box3 ...
        path_to_img2 box1 box2 box2 ...
        ...

        with each box defined as:

        x_min,y_min,x_max,y_max,class_suffix

        """
        if self.detection_annotations is None:
            warnings.warn("No annotations have been previously generated!")
        self.annotation_file = utils.check_file_path(self.annotation_file)
        with open(self.annotation_file, 'w') as fw:
            fw.write('\n'.join(self.detection_annotations))

    def get_random_n_shapes(self):
        """
        Gets a random number of shapes between the specified values in
        self.shape_n_range.
        **Warning**: This method is fixed on 5 as the current default values
        to generate the size of a shape (max) is
        min_size_diagram*2*(1+1.5)/n_shapes and therefore if the n_shapes
        is less than 5 the shape can be bigger than the diagram, raising an
        error in the random integer method. However, if this default values
        change this method will break.

        :return: Randomized number of shapes.
        :raises ValueError: If the minimum value of self.shape_n_range
         makes the randomized shape size bigger than the diagram making
         it unable to be placed.
        """
        minimum = min(self.shape_n_range)
        maximum = max(self.shape_n_range)
        return self.rng.integers(minimum, maximum)

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
        """
        for _ in tqdm(range(self.quantity)):
            seed = None
            if self.debug:
                seed = self.rng.integers(self.quantity*100)

            n_shapes = self.get_random_n_shapes()
            synth_diagram = SyntheticDiagram(shapes_paths,
                                             connections_paths,
                                             texts_paths,
                                             self.output_shape,
                                             self.min_shape,
                                             n_shapes, seed=seed)
            diagram = synth_diagram.generate()
            boxes_annotation = synth_diagram.get_boxes_annotations()
            instance_annotation = synth_diagram.get_instances_annotations()
            diagram_filename = synth_diagram.get_name()
            output_path = self.output_path + diagram_filename
            boxes_annotation = f"{output_path} " + boxes_annotation
            self.detection_annotations.append(boxes_annotation)
            cv2.imwrite(output_path, diagram)
            diagram_hash = diagram_filename.split(".")[0]
            diagram_mask_dir = f"{self.output_path}masks/{diagram_hash}/"
            os.makedirs(diagram_mask_dir, exist_ok=True)
            for element_class, element_mask in instance_annotation.items():
                filepath = f"{diagram_mask_dir}{classes[element_class]}.png"
                cv2.imwrite(filepath, element_mask)
        self.translate_detection_annotations(classes)
        self.save_detection_annotations()
