#!/usr/bin/env python
# coding: utf-8
import math

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from keras.models import load_model


saved_model = load_model('U_net_model_new_2.h5')


# checking whether a image is bright or not using L channel of an image
def isbright(img, dim=200, thresh=0.8):
    img = cv2.resize(img, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    # Normalizing L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    return np.mean(L) > thresh


def predict(image):

    #if not isbright(image):
    #    return image
    image = image / 255.
    output = saved_model.predict(np.array([image]))
    output = output.reshape(128,128,3)

    return output


def gamma_correction(img):
    #if not isbright(img):
    #     return img
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 1
    mean = np.mean(val)
    gamma = math.log(mid * 255) / math.log(mean)
    print(gamma)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    output = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    # applying bilateral filtering to preserve details and reduce noise
    output = cv2.bilateralFilter(output, 9, 50, 50)

    # applying Median blur to remove any salt and pepper noise if available
    output = cv2.medianBlur(output, 3)

    return output


def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("Low Light Image enhancement")
    st.markdown("Model to enhance the images taken during low light")
    st.subheader("Image")
    image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if image_file is not None:
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        st.write(file_details)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.resize(opencv_image, (128, 128))
        col1, col2, col3 = st.columns(3)

        with col2:
            img = st.empty()
            img.image(opencv_image, channels="BGR")
            st.button("Light up- UNET", key="predict")
            st.button("Light up- Gamma_correction", key="gamma")
            st.button("show Orignal", key="original")
            if st.session_state.predict:
                output = predict(opencv_image)
                img.image(output, clamp=True, channels="BGR")
            if st.session_state.original:
                img.image(opencv_image, channels="BGR")
            if st.session_state.gamma:
                output = gamma_correction(opencv_image)
                img.image(output, channels="RGB")



if __name__ == '__main__':
    main()
