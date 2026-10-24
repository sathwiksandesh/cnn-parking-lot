
import sys

import cv2
import numpy as np
import sklearn
import streamlit as st
import tensorflow as tf

print("=" * 45)
print("CNN PARKING LOT - ENVIRONMENT CHECK")
print("=" * 45)

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)
print("Streamlit:", st.__version__)

print("\nGPU devices:", tf.config.list_physical_devices("GPU"))
print("\nEnvironment check completed.")
