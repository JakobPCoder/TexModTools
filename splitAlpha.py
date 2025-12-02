# Copyright Notice
# - TexModTools
# - First published 2022 - Copyright, Jakob Wapenhensch
# - https://creativecommons.org/licenses/by-nc/4.0/
# - https://creativecommons.org/licenses/by-nc/4.0/legalcode

import os
import cv2
import numpy as np
import pathlib

INPUT_FILE_FORMAT = ".png"

def isAlphaConstant(alpha_channel, threshold=1e-6):
    """
    Check if the alpha channel has almost zero variance (constant color).

    Args:
        alpha_channel: 2D numpy array representing the alpha channel
        threshold: Variance threshold below which alpha is considered constant

    Returns:
        bool: True if alpha variance is below threshold (constant), False otherwise
    """
    variance = np.var(alpha_channel.astype(np.float64))
    return variance < threshold

def processImage(path):
    # Read the image
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Failed to read {path}")
        return

    # Check if image has alpha channel (4 channels)
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Split into RGB and alpha
        rgb = img[:, :, :3]  # RGB channels
        alpha = img[:, :, 3]  # Alpha channel

        # Check if alpha channel is constant (variance almost 0)
        if isAlphaConstant(alpha):
            # Save only RGB version (overwrites original)
            cv2.imwrite(str(path), rgb)
            print(f"Processed {path} -> RGB only (constant alpha detected, alpha discarded)")
        else:
            # Save RGB version (overwrites original)
            cv2.imwrite(str(path), rgb)

            # Save alpha as grayscale PNG with "_a" suffix
            alpha_path = path.parent / (path.stem + "_a" + path.suffix)
            cv2.imwrite(str(alpha_path), alpha)

            print(f"Processed {path} -> RGB saved, alpha saved as {alpha_path}")
    else:
        print(f"Skipped {path} (no alpha channel)")

    return

basePath = os.path.dirname(os.path.realpath(__file__))
with os.scandir(basePath) as dir:
    for entry in dir:
        if entry.is_file():
            path = pathlib.Path(entry)
            if path.suffix.lower() == INPUT_FILE_FORMAT:
                processImage(path)
