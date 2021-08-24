from DatasetGenerator.utils.debug_tools.debug_tools import visualize_clusters, draw_bbox
from sklearn.cluster import DBSCAN
import numpy as np
import cv2


class ElementExtractor:
    """
    Object to extract elements from a picture where a series of base
    elements to construct the diagram dataset have been drawn. This
    class includes an static method to generate a bounding box from
    a set of coordinates.

    Basic usage:
    ee = ElementExtractor("imagepath/sheet-aaa-bbb-ccc-ddd.jpeg")
    ee.extract()
    ee.save_elements("output/")
    """

    def __init__(self, image_path, image):
        """
        Initializes an instance of ElementExtractor.

        :param image_path: File path of the input image to process.
        :raises ValueError: If the name if the image is not formatted
         accordingly to the nomenclature sheet-www-xxx-yyy-zzz.jpeg
        :raises FileNotFoundError: If the image can not be loaded.
        """
        self.image_path = image_path
        info = self.image_path.split("/")[-1].split(".")[0].split("-")
        if len(info) != 5:
            raise ValueError("Image name is not formatted correctly.")
        self.sheet_id = info[0]
        self.author_id = info[1]
        self.device_id = info[2]
        self.element_id = info[3]
        self.tool_id = info[4]
        self.image = image  # cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError("Image could not be loaded.")
        self.image_grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.binarized = cv2.adaptiveThreshold(self.image_grayscale, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 6)
        self.cluster_vis = None
        self.bounding_boxes = None
        self.image_area = self.image.shape[0] * self.image.shape[1]
        self.elements = None

    @staticmethod
    def coordinates_to_bbox(coordinates):
        """
        Characterizes a bounding box that encloses an array of coordinates.

        :param coordinates: An array of coordinates [[y0, x0], [y1, x1], ...]
        :return: An array [y0, x0, y1, x1] where p0 is the upper left corner
         and p1 the lower right one.
        """
        y0 = np.min(coordinates.T[0])
        x0 = np.min(coordinates.T[1])
        y1 = np.max(coordinates.T[0])
        x1 = np.max(coordinates.T[1])
        return [y0, x0, y1, x1]

    @staticmethod
    def get_bbox_area(bbox):
        """
        Calculates the area of a given bounding box.

        :param bbox: An array [y0, x0, y1, x1] where p0 is the upper left
         corner and p1 the lower right one.
        :return: Area of the specified bounding box.
        """
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def extract(self, eps=20, min_samples=20, debug_clustering=False,
                debug_segmentation=False):
        """
        Populates the self.elements attribute with an array of the coordinates
        of every element in the image.

        :param eps: Eps parameter of the clustering algorithm.
        :param min_samples: Min_samples parameter of the clustering algorithm.
        :param debug_clustering: If True, shows an opencv GUI with an image with
         each cluster colorized of a single color.
        :param debug_segmentation: If True, shows an opencv GUI with an image where
         a bounding box has been drawn over every detected element.
        """
        # Get coordinates of the pixels with value = 1
        # Coords are retrieved as follows: [[y0, x0], [y1, x1], ...] (using opencv image coords)
        coordinates = np.argwhere(self.binarized)
        # Clustering is done via DBSCAN, OPTICS may be a (memory) cheaper choice
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        self.cluster_vis = visualize_clusters(self.image.copy(), coordinates, clustering.labels_)
        if debug_clustering:
            cv2.imshow("clusters", self.cluster_vis)
            cv2.waitKey(0)
        # DBSCAN tags outliers as -1, so those values are not considered.
        elements = []
        for label in range(np.max(clustering.labels_) + 1):
            label_coords_indices = np.where(clustering.labels_ == label)
            labels_coords = coordinates[label_coords_indices]
            # Store bounding box for each label defined by y0, x0, y1, x1
            # being p0 the upper left corner and p1 the lower right one.
            bbox = ElementExtractor.coordinates_to_bbox(labels_coords)
            if self.get_bbox_area(bbox) / self.image_area * 1000 >= 1:
                elements.append(bbox)
        self.elements = elements
        debug_bboxes = self.image.copy()
        for bbox in self.elements:
            debug_bboxes = draw_bbox(debug_bboxes, bbox)
        self.bounding_boxes = debug_bboxes
        if debug_segmentation:
            cv2.imshow("Bounding boxes", debug_bboxes)
            cv2.waitKey(0)

    def save_elements(self, directory_path):
        """
        Saves each detected element as an independent image in the specified directory.
        Saved images follow the same naming convention as their corresponding
        picture, with an additional 3 digits numerical Id.

        :param directory_path: Path of the directory where images will be saved.
        :raises RuntimeError: If no elements have been previously extracted.
        """
        if self.elements is None:
            raise RuntimeError("Elements have not been extracted from this image,"
                               " you may want to call extract() on this"
                               " ElementExtractor instance.")
        for i, bbox in enumerate(self.elements):
            bbox_image = self.binarized.copy()[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1]
            # bbox_image = self.image.copy()[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
            cv2.imwrite(directory_path +
                        f'{self.image_path.split("/")[-1].split(".")[0]}-{str(i).zfill(3)}.png',
                        bbox_image)


if __name__ == "__main__":
    ee = ElementExtractor("pictures/00000-000-000-a00-002.jpg")
    ee.extract(debug_clustering=True, debug_segmentation=True)
    ee.save_elements("elements/")
