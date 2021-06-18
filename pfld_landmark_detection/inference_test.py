from inference import LandmarkDetector, draw_landmarks
import cv2
import os


def image_detect(image_path):
    image = cv2.imread(image_path)
    landmark_detector = LandmarkDetector()
    landmark = landmark_detector.detect(image)
    draw_landmarks(image, landmark)
    cv2.imshow("image", image)
    cv2.waitKey()


def video_detect(video_path=0):
    """
    直接对视频进行检测关键点不是很准，
    需要先进行人脸框的检测后再进行关键点的检测，
    这样关键点检测的精度能够得到提升
    """
    landmark_detector = LandmarkDetector()
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print(video_path, " is not valid!")
        exit(-1)
    while True:
        _, frame = video.read()
        landmark = landmark_detector.detect(frame)
        draw_landmarks(frame, landmark)
        cv2.imshow("image", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决OpenMP报错问题
    image_path = r'/Users/heroin/dataset/girls/train_1/images/00349_13.jpg'
    image_detect(image_path)
    # video_detect()

