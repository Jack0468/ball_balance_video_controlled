import cv2

class MockReceiver:
    def __init__(self, image_path):
        self.frame = cv2.imread(image_path)
    def get_latest_frame(self):
        return self.frame
    def stop(self):
        pass
