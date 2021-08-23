from DatasetGenerator.bounding_box import bounding_box as bb
import random
import json
import cv2
import os


def get_annotations(annotations_file="./annotation.txt"):
    with open(annotations_file) as f:
        annotations = f.readlines()
    clean_annotations = [a.rstrip() for a in annotations]
    paths = [a.split(" ")[0] for a in clean_annotations]
    boxes = [a.split(" ")[1:] for a in clean_annotations]
    return dict(zip(paths, boxes))


def get_classes(classes_json="./annotated_classes.json"):
    with open(classes_json) as f:
        classes = json.load(f)
    return classes


if __name__ == "__main__":
    path = "diagrams"
    fxy = 0.75  # scale factor
    diagrams = os.listdir(path)
    annotations = get_annotations("./annotation.txt")
    classes = get_classes("./annotated_classes.json")
    k = None
    while k != ord('q'):
        choice = f"{path}/{random.choice(diagrams)}"
        img = cv2.imread(choice)
        img = cv2.resize(img, (0, 0), fx=fxy, fy=fxy)
        boxes = annotations[choice]
        for box in boxes:
            box_ = box.split(",")
            bb.add(img, int(box_[0]) * fxy, int(box_[1]) * fxy, int(box_[2]) * fxy, int(box_[3]) * fxy,
                   classes[box_[4]])
        cv2.imshow("Annotation viewer - Press q to quit", img)
        k = cv2.waitKey(0)
