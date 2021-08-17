import numpy as np
import utils
import cv2


class ConnectionsManager:
    def __init__(self, synthetic_diagram):
        self.synth_diagram = synthetic_diagram
        self.diagram = self.synth_diagram.output_img.copy()
        self.output_shape = self.synth_diagram.output_shape
        self.blank = np.zeros(self.output_shape, dtype=np.uint8)
        self.valid_points = []

    def get_diagram_roi(self, box):
        """
        Gets a empty image where only the box specified by the box argument
        has been copied from the output image of the associated
        SyntheticDiagram.

        :param box: Box denoted by a dictionary with at least the keys ulx, uly
         (upper left) and lrx, lry (lower right).
        :return: OpenCV black image with only the roi specified by box from the
         output diagram.
        """
        blank = self.blank.copy()
        roi = np.ogrid[box["uly"]:box["lry"], box["ulx"]:box["lrx"]]
        roi_img = self.diagram[tuple(roi)]
        blank[tuple(roi)] = roi_img
        return blank

    def get_valid_points(self):
        for i, s1 in enumerate(self.synth_diagram.placed_shapes):
            for s2 in self.synth_diagram.placed_shapes[i + 1:]:
                # For every two shapes in the diagram
                if s1 is not s2:
                    # Get rois and their centers
                    diagram_roi1 = self.get_diagram_roi(s1)
                    diagram_roi2 = self.get_diagram_roi(s2)
                    c1 = utils.get_box_center(s1)
                    c2 = utils.get_box_center(s2)
                    # Trace a line between the centers
                    line = cv2.line(self.blank.copy(), c1, c2, 255, 5)
                    # Get intersection points of the traced line and each shape
                    intxn1 = diagram_roi1 & line  # Intersection image
                    intxn_points1 = utils.get_pixels_coords(intxn1)
                    intxn2 = diagram_roi2 & line  # Intersection image
                    intxn_points2 = utils.get_pixels_coords(intxn2)
                    # Take the farthest point for each shape
                    origin1 = utils.get_farthest_point_from(c1, intxn_points1)
                    origin2 = utils.get_farthest_point_from(c2, intxn_points2)
                    # Check overlap on shapes between the two shapes
                    intxns = intxn1 | intxn2  # Image with both intersections
                    additional_intersections = (self.diagram & line) - intxns
                    # If no intersection with other shapes, save both points
                    if np.all((additional_intersections == 0)):
                        self.valid_points.append((origin1, origin2))
        return self.valid_points
