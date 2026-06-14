# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""

import pyvirtualcam
import numpy as np
import cv2
from PIL import ImageGrab
from matplotlib import pyplot as plt


class VirtualCamera:
    def __init__(self, fps, width, height):
        # Set up standard properties like frame rate and resolution sizes for our virtual camera output.
        self.fps = fps
        self.width = width
        self.height = height

    def capture_screen(self, plt_inside=False, alt_width=0, alt_height=0):
        width = alt_width if alt_width > 0 else self.width
        height = alt_height if alt_height > 0 else self.height
        while True:
            # Take a screenshot using the pillow library box coordinates
            img = ImageGrab.grab(bbox=(0, 0, width, height))
            img_np = np.array(img) # Convert it into a numpy matrix array
            if plt_inside:
                plt.imshow(img_np)
                plt.axis('off')
                plt.show()
            yield img_np # Return the screenshot frame to our processing loop

    def capture_cv_video(self, camera_id, bgr_to_rgb=False):
        cv_vid = cv2.VideoCapture(camera_id)
        # Crash if the script can't connect to the physical camera hardware
        if not cv_vid.isOpened():
            raise RuntimeError('Video-Output cannot be opened.')
        # Configure video properties to match our width, height, and FPS settings
        cv_vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cv_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv_vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cv_vid.set(cv2.CAP_PROP_FPS, self.fps)

        width = int(cv_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cv_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cv_vid.get(cv2.CAP_PROP_FPS)
        print(f'Camera properties: ({width}x{height} @ {fps_in}fps)')

        # create a small preview window — press Q in this window to quit
        cv2.namedWindow("Preview (press Q to quit)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview (press Q to quit)", 640, 360)
        # Infinite loop to constantly pull live images from the lens
        while True:
            ret, frame = cv_vid.read()
            if not ret:
                raise RuntimeError('Camera image cannot be loaded.')
            if bgr_to_rgb:
                frame = frame[...,::-1]

            yield frame # Send the frame right out to our filter processing pipeline

    def virtual_cam_interaction(self, img_generator, print_fps=True):
        print('Quit camera stream with "q" in the preview window')
        # Open up the pyvirtualcam context connection using our preset size values
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, print_fps=print_fps) as cam:
            for img in img_generator:
                cam.send(img)
                cam.sleep_until_next_frame()