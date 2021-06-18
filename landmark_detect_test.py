import os
import sys
import cv2
work_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append()
from pfld_landmark_detection import LandmarkDetector, draw_landmarks, pts2landmarks


if __name__ == '__main__':
    detector = LandmarkDetector()
    image_path = os.path.join(os.getcwd(), "../../images/04370.jpg")
    image = cv2.imread(image_path, 1)
    landmark = detector.detect(image)
    print(landmark)
    draw_landmarks(image, landmark)
    cv2.imshow("image", image)
    cv2.waitKey()

