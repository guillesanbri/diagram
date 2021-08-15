import numpy as np
import random
import cv2


def random_point(mat):
    return ((random.randint(0, mat.shape[1])),
            (random.randint(0, mat.shape[0])))


def get_farthest_point_from(origin, points):
    max_distance_point = origin
    max_distance = 0
    for point in points:
        distance = np.linalg.norm(origin-point)
        if distance > max_distance:
            max_distance = distance
            max_distance_point = point
    return max_distance_point


if __name__ == "__main__":
    img = cv2.imread("tests/test_convex_hull.png")
    img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
    k = 0
    while k != ord('q'):
        c1 = np.array([370, 600])
        p2 = random_point(img)
        black = np.zeros(img.shape, dtype=np.uint8)
        line_img = cv2.line(black, c1, p2, (255, 0, 0), 10)  # Tune line size
        intersection_img = img & line_img
        # Get y, x and reverse them to be x, y
        intersection_points = [np.flip(c[:2]) for c in np.argwhere(intersection_img)]
        intersection = get_farthest_point_from(c1, intersection_points)
        full = img | line_img
        full = cv2.circle(full, intersection, 5, (0, 255, 0), 2)
        cv2.imshow("Press q to exit", full)
        cv2.imshow("Intersections", intersection_img)
        k = cv2.waitKey(0)
