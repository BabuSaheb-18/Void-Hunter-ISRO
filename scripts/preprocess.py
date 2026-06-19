import os
import cv2
import numpy as np
import rasterio

# -----------------------------
# PATHS
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENE_PATH = os.path.join(
    ROOT,
    "dataset",
    "landsat_raw",
    "Scene001",
    "LC08_L2SP_146040_20241221_20241228_02_T1"
)

RGB_OUTPUT = os.path.join(ROOT, "dataset", "processed", "rgb")
IR_OUTPUT = os.path.join(ROOT, "dataset", "processed", "ir")

os.makedirs(RGB_OUTPUT, exist_ok=True)
os.makedirs(IR_OUTPUT, exist_ok=True)

# -----------------------------
# READ BAND
# -----------------------------
def read_band(filename):
    path = os.path.join(SCENE_PATH, filename)

    with rasterio.open(path) as src:
        band = src.read(1)

    return band.astype(np.float32)

# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize(img):

    img = np.nan_to_num(img)

    minimum = np.min(img)
    maximum = np.max(img)

    img = (img - minimum) / (maximum - minimum + 1e-8)

    img = (img * 255).astype(np.uint8)

    return img

# -----------------------------
# CREATE RGB IMAGE
# -----------------------------
def create_rgb():

    red = normalize(read_band("LC08_L2SP_146040_20241221_20241228_02_T1_SR_B4.TIF"))

    green = normalize(read_band("LC08_L2SP_146040_20241221_20241228_02_T1_SR_B3.TIF"))

    blue = normalize(read_band("LC08_L2SP_146040_20241221_20241228_02_T1_SR_B2.TIF"))

    rgb = np.dstack((red, green, blue))

    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    save_path = os.path.join(RGB_OUTPUT, "scene001_rgb.png")

    cv2.imwrite(save_path, rgb)

    print("RGB Saved ->", save_path)

# -----------------------------
# CREATE THERMAL IMAGE
# -----------------------------
def create_ir():

    thermal = read_band("LC08_L2SP_146040_20241221_20241228_02_T1_ST_B10.TIF")

    thermal = normalize(thermal)

    save_path = os.path.join(IR_OUTPUT, "scene001_ir.png")

    cv2.imwrite(save_path, thermal)

    print("IR Saved ->", save_path)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("PROCESSING LANDSAT SCENE")
    print("=" * 50)

    create_rgb()

    create_ir()

    print("=" * 50)
    print("DONE")
    print("=" * 50)