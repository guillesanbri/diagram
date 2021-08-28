import os

from DatasetGenerator.ElementExtractor import ElementExtractor
import cv2

# TODO: Change this to instructions on terminal
window = "Press space to confirm - Press d to discard - Press q to abort"
image = None
element_extractor = None
eps = 20
min_samples = 20


def on_trackbar_eps(val):
    global eps
    eps = val
    show_element_extractor(eps, min_samples)


def on_trackbar_samples(val):
    global min_samples
    min_samples = val
    show_element_extractor(eps, min_samples)


# TODO: If you click on a element you remove it
def discard_element(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        element_extractor.add_discard_point((x, y))
        show_element_extractor(eps, min_samples)


def show_element_extractor(eps, min_samples):
    element_extractor.extract(eps, min_samples)
    clustering_image = element_extractor.bounding_boxes
    cv2.imshow(window, clustering_image)


if __name__ == "__main__":
    # Get list of the paths of the pictures with elements to extract
    pictures_dir = "pictures/"
    already_processed_dir = "pictures/already_extracted/"
    extensions = ["jpg", "jpeg", "png"]
    pictures_paths = [p for p in os.listdir(pictures_dir) if p.split(".")[-1] in extensions]

    # Create window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 720, 720)
    trackbar_eps = "Clustering algorithm eps"
    trackbar_min_samples = "Clustering algorithm min samples"
    cv2.createTrackbar(trackbar_eps, window, eps, 100, on_trackbar_eps)
    cv2.createTrackbar(trackbar_min_samples, window, min_samples,
                       250, on_trackbar_samples)
    cv2.setMouseCallback(window, discard_element)
    # Iterate over each picture
    k = None
    for picture_name in pictures_paths:
        full_path = pictures_dir + picture_name
        image = cv2.imread(full_path)
        display_image = image
        element_extractor = ElementExtractor(full_path, image)
        show_element_extractor(eps, min_samples)
        while True:
            k = cv2.waitKey(0)
            if k == ord("q"):
                print("Aborted")
                break
            elif k == ord("d"):
                print(f"Discarded {picture_name}")
                break
            elif k == ord(" "):
                print(f"Saving elements from {picture_name}")
                element_extractor.save_elements("elements/")
                print(f"Moving picture from {pictures_dir} to {already_processed_dir}")
                os.rename(full_path, already_processed_dir + picture_name)
                break
        if k == ord("q"):
            break
