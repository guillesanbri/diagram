from DatasetGenerator.utils.utils import get_element_paths
import DatasetGenerator.utils.utils as utils
from DatasetGenerator.DiagramGenerator import DiagramGenerator
import json
import os

if __name__ == "__main__":
    # Read shapes
    shapes_dict, shapes_paths = get_element_paths("shapes")
    print(shapes_paths)
    # Read connections
    connections_dict, connections_paths = get_element_paths("connections")
    # Read texts

    # Dictionary of id:element_suffix
    # Element tagging can be overwritten as follows
    # shapes_dict = {'400': 'shape', '600': 'shape', '800': 'shape'}
    ids_suffixes = {**shapes_dict, **connections_dict}

    dg = DiagramGenerator("diagrams/", 10, (5, 15), seed=42, debug=True)
    annotation_classes = dg.run(shapes_paths, connections_paths, None, ids_suffixes)
    classes_json_path = utils.check_file_path('annotated_classes.json')
    with open(classes_json_path, 'w') as f:
        json.dump(annotation_classes, f)
    # dg.save_annotation_as(annotations, "VOC")  # Convert to typical bbox annotation formats.
