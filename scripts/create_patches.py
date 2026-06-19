import os
import cv2
import numpy as np

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RGB_IMAGE = os.path.join(
    ROOT,
    "dataset",
    "processed",
    "rgb",
    "scene001_rgb.png"
)

IR_IMAGE = os.path.join(
    ROOT,
    "dataset",
    "processed",
    "ir",
    "scene001_ir.png"
)

RGB_OUTPUT = os.path.join(ROOT, "dataset", "patches", "rgb")
IR_OUTPUT = os.path.join(ROOT, "dataset", "patches", "ir")

os.makedirs(RGB_OUTPUT, exist_ok=True)
os.makedirs(IR_OUTPUT, exist_ok=True)

PATCH_SIZE = 256

# Skip patches having less than this much useful data
MIN_VALID_RATIO = 0.60


# ---------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------

rgb = cv2.imread(RGB_IMAGE)

ir = cv2.imread(IR_IMAGE, cv2.IMREAD_GRAYSCALE)

if rgb is None:
    raise FileNotFoundError(RGB_IMAGE)

if ir is None:
    raise FileNotFoundError(IR_IMAGE)

height, width = ir.shape

print(f"Image Size : {width} x {height}")

count = 0
skipped = 0

# ---------------------------------------------------
# PATCH LOOP
# ---------------------------------------------------

for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):

    for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):

        rgb_patch = rgb[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        ir_patch = ir[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        # Ignore patches with mostly black pixels
        valid_pixels = np.sum(ir_patch > 5)

        ratio = valid_pixels / (PATCH_SIZE * PATCH_SIZE)

        if ratio < MIN_VALID_RATIO:
            skipped += 1
            continue

        count += 1

        rgb_name = f"rgb_{count:06d}.png"
        ir_name = f"ir_{count:06d}.png"

        cv2.imwrite(
            os.path.join(RGB_OUTPUT, rgb_name),
            rgb_patch
        )

        cv2.imwrite(
            os.path.join(IR_OUTPUT, ir_name),
            ir_patch
        )

print("=" * 50)
print("Patch Generation Complete")
print("=" * 50)

print("Generated :", count)
print("Skipped   :", skipped)