from DatasetGenerator.ElementExtractor import ElementExtractor
import cv2

window = "Press space to confirm - Press d to discard - Press q to abort"
image = None
element_extractor = None
eps = 20
min_samples = 20


def on_trackbar_eps(val):
    global eps
    eps = val
    show_element_extractor(eps, min_samples)
    print(eps, min_samples)


def on_trackbar_samples(val):
    global min_samples
    min_samples = val
    show_element_extractor(eps, min_samples)
    print(eps, min_samples)


def show_element_extractor(eps, min_samples):
    element_extractor.extract(eps, min_samples)
    clustering_image = element_extractor.bounding_boxes
    cv2.imshow(window, clustering_image)


if __name__ == "__main__":
    # Get list of the paths of the pictures with elements to extract
    pictures = ["DatasetGenerator/pictures/00000-000-000-a00-002.jpg",
                "DatasetGenerator/pictures/easy1-aaa-bbb-ccc-ddd.jpeg"]
    # Create window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 720, 720)
    trackbar_eps = "Clustering algorithm eps"
    trackbar_min_samples = "Clustering algorithm min samples"
    cv2.createTrackbar(trackbar_eps, window, eps, 100, on_trackbar_eps)
    cv2.createTrackbar(trackbar_min_samples, window, min_samples,
                       250, on_trackbar_samples)
    # Iterate over each picture
    for picture_path in pictures:
        image = cv2.imread(picture_path)
        display_image = image
        element_extractor = ElementExtractor(picture_path, image)
        show_element_extractor(eps, min_samples)
        k = cv2.waitKey(0)
        if k == ord("q"):
            print("Aborted")
            break
        elif k == ord("d"):
            print(f"Discarded {picture_path}")
            pass
        else:
            print(f"Saving elements from {picture_path}")
            element_extractor.save_elements("elements/")
