import streamlit as st
import cv2
import numpy as np
from PIL import Image
from retinex import enhance_low_light


st.set_page_config(
    page_title="Retinex Low-Light Enhancement",
    page_icon="🌙",
    layout="wide"
)


st.title("🌙 Retinex-Based Low-Light Image Enhancement")

st.write(
    "Upload a low-light image and enhance it using "
    "Retinex-based image processing."
)


uploaded_file = st.file_uploader(
    "Upload a low-light image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Convert PIL → NumPy
    rgb_image = np.array(pil_image)

    # Convert RGB → BGR for OpenCV
    bgr_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR
    )

    # Enhance image
    enhanced_bgr = enhance_low_light(
        bgr_image,
        sigma=30
    )

    # Convert BGR → RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced_bgr,
        cv2.COLOR_BGR2RGB
    )

    st.subheader("Image Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Original")
        st.image(
            rgb_image,
            use_container_width=True
        )

    with col2:
        st.write("### Enhanced")
        st.image(
            enhanced_rgb,
            use_container_width=True
        )

    # Convert enhanced image to PNG
    success, encoded_image = cv2.imencode(
        ".png",
        enhanced_bgr
    )

    if success:

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=encoded_image.tobytes(),
            file_name="enhanced_image.png",
            mime="image/png"
        )