import cv2
import numpy as np


def single_scale_retinex(image, sigma=30):
    """
    Single Scale Retinex (SSR)

    image:
        Input BGR image from OpenCV

    sigma:
        Gaussian blur scale
    """

    # Convert image to float
    image = image.astype(np.float32) + 1.0

    # Gaussian blur gives an estimate of illumination
    illumination = cv2.GaussianBlur(
        image,
        (0, 0),
        sigma
    )

    # Retinex formula:
    # R(x) = log(I(x)) - log(L(x))
    retinex = np.log(image) - np.log(illumination)

    # Normalize each channel to 0-255
    result = np.zeros_like(retinex)

    for channel in range(3):
        channel_data = retinex[:, :, channel]

        minimum = channel_data.min()
        maximum = channel_data.max()

        result[:, :, channel] = (
            (channel_data - minimum)
            / (maximum - minimum)
            * 255
        )

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


def enhance_low_light(image, sigma=30):
    """
    Enhance a low-light image using Retinex.
    """

    retinex_result = single_scale_retinex(
        image,
        sigma
    )

    # Improve contrast
    lab = cv2.cvtColor(
        retinex_result,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    # CLAHE improves local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced