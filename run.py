# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
"""
Keyboard controls (press in the Preview window):
  [F] — cycle through filters
  [L] — toggle linear transformation
  [E] — toggle histogram equalization
  [S] — toggle stats overlay
  [M] — toggle meme/gesture overlay
  [K] — toggle landmark skeleton
  [Q] — quit
"""

import cv2
from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba, build_stats_overlay_text, equalize_histogram
from basics import linear_transform, FILTERS
from basics import GestureProcessor


def custom_processing(img_source_generator):
    """
    Main sequence step generator. This loops through frames, listens to keyboard triggers,
    runs image matrix filters, updates live plots, and sends everything to the virtual camera.
    """
    # Initialize our fast live Matplotlib histogram plot elements
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    # Setup default toggle states for our keyboard options
    show_equalized = False
    show_stats = False
    current_filter = 0
    use_linear = False
    
    # Custom values for Linear Transformation contrast (alpha) and brightness (beta)
    lin_alpha = 1.5
    lin_beta = 30.0

    # Initialize the deep learning MediaPipe Hand Gesture tracker instance
    gesture_proc = GestureProcessor(memes_folder="memes", show_landmarks=False)

    # Set up our local window preview frame size
    cv2.namedWindow("Preview (press Q to quit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview (press Q to quit)", 640, 360)

    # Main stream loop processing frames one by one
    for sequence in img_source_generator:
        # Check if user pressed any keys on keyboard (-1 means no key pressed)
        key = cv2.waitKey(1) & 0xFF

        # Quit script if user presses 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        # Toggle histogram equalization contrast mapping
        if key == ord('e'):
            show_equalized = not show_equalized
            print(f"Equalization: {'ON' if show_equalized else 'OFF'}")

        # Toggle image text statistics numbers display
        if key == ord('s'):
            show_stats = not show_stats
            print(f"Stats: {'ON' if show_stats else 'OFF'}")

        # Cycle up through the filters index list mapping dict items
        if key == ord('f'):
            current_filter = (current_filter + 1) % len(FILTERS)
            print(f"Filter: {FILTERS[current_filter][0]}")

        # Toggle linear contrast matrix multiplication blocks
        if key == ord('l'):
            use_linear = not use_linear
            print(f"Linear Transform: {'ON' if use_linear else 'OFF'}")

        # Toggle MediaPipe overlay rendering step
        if key == ord('m'):
            gesture_proc.enabled = not gesture_proc.enabled
            print(f"Meme overlay: {'ON' if gesture_proc.enabled else 'OFF'}")

        # Toggle showing green skeleton joint tracking dots
        if key == ord('k'):
            gesture_proc.show_landmarks = not gesture_proc.show_landmarks
            print(f"Landmarks: {'ON' if gesture_proc.show_landmarks else 'OFF'}")

        # --- RUNTIME DATA OPERATIONS PROCESSING PIPELINE ---

        #  Run Equalization if toggled ON, otherwise keep raw data copy
        if show_equalized:
            display_frame = equalize_histogram(sequence)
        else:
            display_frame = sequence.copy()

        #  Apply linear transform equation for exposure shifts
        if use_linear:
            display_frame = linear_transform(display_frame, alpha=lin_alpha, beta=lin_beta)

        #  Run our selected custom filters (Sobel edge, blur, sharpen, etc.)
        filter_name, filter_fn = FILTERS[current_filter]
        display_frame = filter_fn(display_frame)

        #  Run the frames through MediaPipe to evaluate tracking states & draw memes
        display_frame = gesture_proc.process(display_frame)

        #  Extract raw RGB arrays and draw updated values inside plot buffers
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        #  Overlay the live Matplotlib histogram plot straight onto our stream array
        display_frame = plot_overlay_to_image(display_frame, fig)

        # Draw statistics menu data if user turned stats display ON
        if show_stats:
            stats_text = build_stats_overlay_text(sequence)
            display_frame = plot_strings_to_image(display_frame, stats_text, text_color=(255, 255, 0), right_space=1240, top_space=500)

        # Define lines for the instructions HUD layout menu.
        # Fixed top_space at 530 so 35px line gaps don't drop past our 720p resolution bottom border canvas.
        hud_lines = [
            f"[F]Filter: {filter_name}",
            f"[L]Linear: {'ON' if use_linear else 'OFF'}",
            f"[E]Equalise: {'ON' if show_equalized else 'OFF'}",
            f"[M]Memes: {'ON' if gesture_proc.enabled else 'OFF'}",
            "[S] Stats [K] Skeleton [Q] Quit"
        ]
        display_frame = plot_strings_to_image(display_frame, hud_lines, text_color=(0, 220, 255), right_space=520, top_space=530)

        # Revert colors from RGB back to standard OpenCV BGR space for the preview window
        preview = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Preview (press Q to quit)", preview)

        # Yield frame arrays directly into the pyvirtualcam runtime device loop
        yield display_frame


def main():
    """Defines stream shape configurations and boots up camera interfaces."""
    width = 1280
    height = 720
    fps = 30

    vc = VirtualCamera(fps, width, height)

    # Launch camera stream pipeline.
    # Note: On newer MacBooks (M4 hardware layer), built-in cameras map 
    # to device index 1 instead of index 0.
    vc.virtual_cam_interaction(
        custom_processing(
            vc.capture_cv_video(1, bgr_to_rgb=True)
        )
    )

if __name__ == "__main__":
    main()