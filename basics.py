# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit
import numpy as np
import cv2
import mediapipe as mp
import os
from collections import Counter


# PART 1: HISTOGRAM & BASIC STATS


@njit
def histogram_figure_numba(np_img):
    """
    Calculates the RGB histogram values. We use @njit here because 
    looping through millions of pixels in normal Python makes the video super laggy.
    """
    # Create empty arrays to count the colors (0 to 255)
    r_count = np.zeros(shape=(256,), dtype=np.float64)
    g_count = np.zeros(shape=(256,), dtype=np.float64)
    b_count = np.zeros(shape=(256,), dtype=np.float64)

    height = np_img.shape[0]
    width = np_img.shape[1]
    total_pixels = height * width

    # Go through every single pixel and count the RGB values
    for y in range(height):
        for x in range(width):
            r_count[np_img[y, x, 0]] += 1
            g_count[np_img[y, x, 1]] += 1
            b_count[np_img[y, x, 2]] += 1

    # Scale the values down so the graph fits nicely in our window
    if total_pixels > 0:
        r_bars = r_count / total_pixels * 3.0
        g_bars = g_count / total_pixels * 3.0
        b_bars = b_count / total_pixels * 3.0
    else:
        r_bars = r_count
        g_bars = g_count
        b_bars = b_count

    return r_bars, g_bars, b_bars


def compute_image_stats(np_img):
    """Calculates Mean, Std Dev, Min, Max, and Mode for each channel."""
    stats_lines = []
    channel_names = ['R', 'G', 'B']

    for i, name in enumerate(channel_names):
        # Flatten the 2D image channel into 1D so numpy can do math on it easily
        channel = np_img[:, :, i].flatten()
        mean = np.mean(channel)
        std = np.std(channel)
        cmax = np.max(channel)
        cmin = np.min(channel)
        
        # Count frequencies to find the most common pixel value (the mode)
        counts = np.bincount(channel)
        mode = np.argmax(counts)
        
        stats_lines.append(f'{name} - mean: {mean:.2f}, std: {std:.2f}, min: {cmin}, max: {cmax}, mode: {mode}')

    return stats_lines


def compute_entropy(np_img):
    """Calculates the image entropy (tells us how detailed/busy the image is)."""
    flat = np_img.flatten()
    total_pixels = flat.size
    
    # Get the probability for each pixel value
    counts = np.bincount(flat, minlength=256)
    probabilities = counts / total_pixels
    
    # Epsilon is a tiny number to prevent a crash if a probability is exactly 0
    epsilon = 1e-10
    entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
    return entropy


def equalize_histogram(np_img):
    """Stretches out the image contrast using cumulative distribution function (CDF)."""
    equalized = np.zeros_like(np_img)
    for i in range(3):
        channel = np_img[:, :, i]
        total_pixels = channel.size
        
        # Calculate individual channel histogram and cumulative sum
        hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 255))
        cdf = hist.cumsum()
        
        # Normalize the values to map them back between 0 and 255
        cdf_min = cdf[cdf > 0].min()
        cdf_normalized = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(np.uint8)
        
        # Apply the new balanced values to the channel
        equalized[:, :, i] = cdf_normalized[channel]
    return equalized



# PART 2: BASIC IMAGE FILTERS


def apply_sobel_filter(np_img):
    """Finds borders/edges in the image using Sobel kernels."""
    # Convert to grayscale first since edge detection just looks at brightness changes
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    # Look for edges in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the horizontal and vertical edges together
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    # Convert back to RGB so the virtual camera can display it normally
    sobel_rgb = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
    return sobel_rgb


def apply_gaussian_blur(np_img, kernel_size=5):
    """Blurs the image using a Gaussian blur matrix."""
    # Safety check: kernel size must be an odd number (5, 7, 9...)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)
    return blurred


def apply_sharpen_filter(np_img):
    """Sharpens the details in the image using a standard custom matrix."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(np_img, -1, kernel)
    return sharpened


def apply_gabor_filter(np_img):
    """Applies a Gabor filter (helps detect textures at specific angles)."""
    # Create a kernel pointing at 45 degrees (pi / 4)
    gabor_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    result = np.zeros_like(np_img)
    
    # Apply it to R, G, and B separately
    for i in range(3):
        filtered = cv2.filter2D(np_img[:, :, i], cv2.CV_64F, gabor_kernel)
        result[:, :, i] = np.clip(np.abs(filtered), 0, 255).astype(np.uint8)
    return result


def linear_transform(np_img, alpha=1.5, beta=30):
    """Changes brightness and contrast. Alpha = contrast, Beta = brightness."""
    transformed = np_img.astype(np.float32) * alpha + beta
    # Clip values so they stay between 0 and 255 (no wrapping bugs)
    return np.clip(transformed, 0, 255).astype(np.uint8)


# Dictionary linking key indices to our functions
FILTERS = {
    0: ("No Filter", lambda img: img.copy()),
    1: ("Sobel Filter", apply_sobel_filter),
    2: ("Gaussian Blur", apply_gaussian_blur),
    3: ("Sharpen Filter", apply_sharpen_filter),
    4: ("Gabor Filter", apply_gabor_filter)
}


def build_stats_overlay_text(np_img):
    """Groups all our text stats together into a list of strings."""
    lines = [" Image Statistics:"]
    lines += compute_image_stats(np_img)
    entropy = compute_entropy(np_img)
    lines.append(f'Entropy: {entropy:.3f} bit')
    return lines



# PART 3: MEDIAPIPE GESTURE DETECTION (SOMETHING SPECIAL)


import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Download the pre-trained neural network weights if we don't have them yet
_MODEL_PATH = "hand_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(_MODEL_PATH):
    print("Downloading MediaPipe hand landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Model downloaded.")

# Landmark index locations from MediaPipe's documentation
FINGERTIP_IDS  = [4, 8, 12, 16, 20]   # Tips of thumb, index, middle, ring, pinky
FINGER_MCP_IDS = [3, 5,  9, 13, 17]   # Base knuckles of fingers

# Pairs of tracking points to draw the white skeleton lines
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Set up MediaPipe Hand Landmarker configurations
_base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
_hand_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
hands_detector = mp_vision.HandLandmarker.create_from_options(_hand_options)

# Link gesture names to our image assets
GESTURE_TO_MEME = {
    "thumbs_up":   "thumbs_up.png",
    "peace":       "peace.png",
    "open_palm":   "open_palm.png",
    "fist":        "fist.png",
    "hang_loose":  "hang_loose.png",
    "pointing_up": "pointing_up.png",
}

GESTURE_DISPLAY_NAMES = {
    "thumbs_up":   "Thumbs Up!",
    "peace":       "Peace",
    "open_palm":   "Open Palm",
    "fist":        "Fist",
    "hang_loose":  "Hang Loose",
    "pointing_up": "Pointing Up",
    "none":        "",
}


def get_extended_fingers(landmarks):
    """
    Checks if fingers are open or closed.
    We check if the fingertip is above the middle knuckle (PIP joint). 
    This is way more stable than checking the base knuckles when tilting your hand.
    """
    MARGIN = 0.02
    finger_tips = [8,  12, 16, 20]
    finger_pips = [6,  10, 14, 18]
    fingers = []
    
    # In OpenCV coordinates, a smaller Y value means higher up on the screen!
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y - MARGIN)

    # Thumb logic: uses horizontal distances to see if it is sticking out sideways
    thumb_tip   = landmarks[4]
    index_mcp   = landmarks[5]
    middle_mcp  = landmarks[9]
    dist_extended = abs(thumb_tip.x - index_mcp.x) + abs(thumb_tip.y - index_mcp.y)
    dist_tucked   = abs(thumb_tip.x - middle_mcp.x) + abs(thumb_tip.y - middle_mcp.y)
    thumb = dist_extended > 0.1

    return [thumb] + fingers


def _draw_landmarks(frame, landmarks):
    """Draws white lines and green tracking circles over our hand skeleton."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 255, 0), -1)


DEBUG_GESTURES = False

def classify_gesture(landmarks):
    """Matches the combination of open/closed fingers to a named gesture state."""
    ext = get_extended_fingers(landmarks)
    thumb, index, middle, ring, pinky = ext
    extended_count = sum(ext)
    finger_count = sum(ext[1:])

    # Fast check to see if all 4 main fingers are completely curled down
    pip_ids = [6, 10, 14, 18]
    fingers_curled = all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip([8,12,16,20], pip_ids))

    if DEBUG_GESTURES:
        print(f"T={thumb} I={index} M={middle} R={ring} P={pinky} curled={fingers_curled} count={finger_count}")

    # Fist: all fingers curled, thumb tucked in
    if fingers_curled and not thumb:
        return "fist"

    # Thumbs up: thumb is out, but everything else is closed
    if thumb and fingers_curled:
        return "thumbs_up"

    # Hang loose: thumb and pinky out, middle fingers curled down
    if not thumb and index and middle and ring and pinky:
        return "hang_loose"

    # Open palm: everything extended flat
    if thumb and finger_count == 4:
        return "open_palm"

    # Peace: only index and middle fingers are up
    if index and middle and not ring and not pinky:
        return "peace"

    # Pointing up: only index finger is up
    if index and not middle and not ring and not pinky and finger_count == 1:
        return "pointing_up"

    return "none"


def load_meme_images(memes_folder="memes"):
    """Loads static PNG images from the memes directory and checks the alpha layers."""
    meme_images = {}
    for gesture, filename in GESTURE_TO_MEME.items():
        filepath = os.path.join(memes_folder, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # If image has alpha layer, flip channels from BGR to RGB
                if img.shape[2] == 4:
                    img[:, :, :3] = img[:, :, 2::-1]
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    alpha = np.ones((*img.shape[:2], 1), dtype=np.uint8) * 255
                    img = np.concatenate([img, alpha], axis=2)
                meme_images[gesture] = img
            else:
                meme_images[gesture] = None
        else:
            meme_images[gesture] = None
    return meme_images


def make_placeholder_meme(gesture_name, width=300, height=200):
    """Fallback function: makes a basic color box if a image file is missing."""
    colours = {
        "thumbs_up":   (255, 200, 0),
        "peace":       (0, 200, 255),
        "open_palm":   (0, 255, 100),
        "fist":        (255, 80, 80),
        "hang_loose":  (255, 140, 0),
        "pointing_up": (0, 180, 255),
    }
    colour = colours.get(gesture_name, (200, 200, 200))
    placeholder = np.full((height, width, 3), colour, dtype=np.uint8)
    label = GESTURE_DISPLAY_NAMES.get(gesture_name, gesture_name)
    cv2.putText(placeholder, label, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
    return np.concatenate([placeholder, alpha], axis=2)


def overlay_meme(frame, meme_rgba, position="top_right", scale=0.30):
    """Uses alpha blending to paste transparency graphics cleanly over our background stream."""
    frame_h, frame_w = frame.shape[:2]
    meme_w = int(frame_w * scale)
    meme_h = int(meme_rgba.shape[0] * meme_w / meme_rgba.shape[1])
    meme_resized = cv2.resize(meme_rgba, (meme_w, meme_h))

    margin = 20
    # Coordinate calculations based on where we want to render the overlay box
    if position == "top_right":
        x1, y1 = frame_w - meme_w - margin, margin
    elif position == "top_left":
        x1, y1 = margin, margin
    elif position == "bottom_right":
        x1, y1 = frame_w - meme_w - margin, frame_h - meme_h - margin
    elif position == "bottom_left":
        x1, y1 = margin, frame_h - meme_h - margin
    else:
        x1, y1 = (frame_w - meme_w) // 2, (frame_h - meme_h) // 2

    x2, y2 = x1 + meme_w, y1 + meme_h
    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return frame

    # Split RGB channels from Alpha data masks
    meme_rgb = meme_resized[:, :, :3].astype(np.float32)
    meme_alpha = meme_resized[:, :, 3].astype(np.float32) / 255.0
    alpha_3ch = np.stack([meme_alpha] * 3, axis=2)
    
    # Mathematical blend equation to handle smooth transparency borders
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    blended = alpha_3ch * meme_rgb + (1.0 - alpha_3ch) * roi
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame


class VideoPlayer:
    """Manages playing mp4 video files back in a loop if used instead of flat images."""
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.loaded = os.path.exists(video_path)
        if self.loaded:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.loaded = False

    def next_frame_rgba(self):
        """Grabs the next video frame and auto-loops back to start frame at EOF."""
        if not self.loaded:
            return None
        ret, frame = self.cap.read()
        if not ret:
            # End of video reached — jump back to frame index 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None
        # Convert BGR to RGB matrix and add a full transparency alpha block layer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        alpha = np.ones((*frame_rgb.shape[:2], 1), dtype=np.uint8) * 255
        return np.concatenate([frame_rgb, alpha], axis=2)

    def reset(self):
        if self.loaded:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap:
            self.cap.release()


class GestureProcessor:
    """The manager class that takes incoming video frames, runs AI checks, and drops overlays."""
    VIDEO_GESTURES = ["open_palm", "peace", "fist", "hang_loose", "thumbs_up", "pointing_up"]

    def __init__(self, memes_folder="memes", show_landmarks=False):
        self.meme_images = load_meme_images(memes_folder)
        self.show_landmarks = show_landmarks
        for gesture in GESTURE_TO_MEME:
            if self.meme_images.get(gesture) is None:
                self.meme_images[gesture] = make_placeholder_meme(gesture)
        self.gesture_history = []
        self.history_length = 5  # Keeps track of last 5 frames to smooth out jumping/flickering
        self.enabled = True
        self.prev_gesture = "none"

        # Load video media elements if available in folder
        self.videos = {}
        for gesture in self.VIDEO_GESTURES:
            path = os.path.join(memes_folder, f"{gesture}.mp4")
            player = VideoPlayer(path)
            self.videos[gesture] = player
            if player.loaded:
                print(f"Loaded video: {path}")
            else:
                print(f"No video found at {path} — using image fallback")

    def _smooth_gesture(self, gesture):
        """Uses a majority vote over recent frames to avoid flickering detections."""
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        counts = Counter(self.gesture_history)
        return counts.most_common(1)[0][0]

    def process(self, frame):
        """Main calculation pipeline step run on every frame step."""
        if not self.enabled:
            return frame

        # Convert the frame format into MediaPipe's custom Image wrapper
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = hands_detector.detect(mp_image)

        detected_gesture = "none"

        # If a hand skeleton is detected, classify what shape it is making
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            if self.show_landmarks:
                _draw_landmarks(frame, hand_landmarks)
            detected_gesture = classify_gesture(hand_landmarks)

        stable_gesture = self._smooth_gesture(detected_gesture)

        # If we switch to a new gesture, restart that gesture's video from frame 0
        if stable_gesture != self.prev_gesture and stable_gesture in self.videos:
            self.videos[stable_gesture].reset()

        # Render either looped video frames or fallback static asset boxes
        if stable_gesture != "none":
            player = self.videos.get(stable_gesture)
            if player and player.loaded:
                video_frame = player.next_frame_rgba()
                if video_frame is not None:
                    frame = overlay_meme(frame, video_frame, position="top_right", scale=0.30)
            else:
                meme = self.meme_images.get(stable_gesture)
                if meme is not None:
                    frame = overlay_meme(frame, meme, position="top_right", scale=0.30)

        self.prev_gesture = stable_gesture

        # Put status string tracking labels onto the upper left canvas border corner
        display_name = GESTURE_DISPLAY_NAMES.get(stable_gesture, "")
        if display_name:
            frame = cv2.putText(
                frame, f"Gesture: {display_name}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3, cv2.LINE_AA
            )

        return frame