from DatasetGenerator.utils.utils import get_element_paths
from DatasetGenerator.DiagramGenerator import DiagramGenerator

if __name__ == "__main__":
    # Read shapes
    shapes_dict, shapes_paths = get_element_paths("shapes")
    # Read connections
    connections_dict, connections_paths = get_element_paths("connections")
    # Read texts

    # Dictionary of id:element_suffix
    ids_suffixes = {**shapes_dict, **connections_dict}

    dg = DiagramGenerator("diagrams/", 10, (5, 15), seed=42, debug=True)
    dg.run(shapes_paths, connections_paths, None, ids_suffixes)
