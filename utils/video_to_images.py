import cv2
import os


class VideoFrames:
    """Retrieves successive frames from a video."""

    def __init__(self, path):
        self.path = path
        self.frames = self._get_frames()

    def _get_frames(self):
        """Reads frames from a video."""

        frames = []
        vidcap = cv2.VideoCapture(self.path)

        success = True
        while success:
            success, image = vidcap.read()
            if success:
                frames.append(image)

        return frames

    def dump_images(self, path):
        """Dumps images to a directory."""

        os.makedirs(path, exist_ok=True)

        for i, frame in enumerate(self.frames):
            cv2.imwrite(f"{path}/{i}.jpg", frame)
