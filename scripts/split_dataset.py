import os
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCH_RGB = os.path.join(ROOT, "dataset", "patches", "rgb")
PATCH_IR = os.path.join(ROOT, "dataset", "patches", "ir")

TRAIN_RGB = os.path.join(ROOT, "dataset", "train", "rgb")
TRAIN_IR = os.path.join(ROOT, "dataset", "train", "ir")

VAL_RGB = os.path.join(ROOT, "dataset", "val", "rgb")
VAL_IR = os.path.join(ROOT, "dataset", "val", "ir")

TEST_RGB = os.path.join(ROOT, "dataset", "test", "rgb")
TEST_IR = os.path.join(ROOT, "dataset", "test", "ir")

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    TRAIN_RGB, TRAIN_IR,
    VAL_RGB, VAL_IR,
    TEST_RGB, TEST_IR
]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# READ FILES
# ============================================================

rgb_files = sorted(os.listdir(PATCH_RGB))
ir_files = sorted(os.listdir(PATCH_IR))

assert len(rgb_files) == len(ir_files), "RGB and IR counts do not match!"

pairs = list(zip(rgb_files, ir_files))

random.seed(42)
random.shuffle(pairs)

total = len(pairs)

train_size = int(total * 0.80)
val_size = int(total * 0.10)

train_pairs = pairs[:train_size]
val_pairs = pairs[train_size:train_size + val_size]
test_pairs = pairs[train_size + val_size:]

# ============================================================
# COPY FUNCTION
# ============================================================

def copy_pairs(pair_list, rgb_dst, ir_dst):

    for rgb_name, ir_name in pair_list:

        shutil.copy2(
            os.path.join(PATCH_RGB, rgb_name),
            os.path.join(rgb_dst, rgb_name)
        )

        shutil.copy2(
            os.path.join(PATCH_IR, ir_name),
            os.path.join(ir_dst, ir_name)
        )

# ============================================================
# SPLIT DATASET
# ============================================================

copy_pairs(train_pairs, TRAIN_RGB, TRAIN_IR)
copy_pairs(val_pairs, VAL_RGB, VAL_IR)
copy_pairs(test_pairs, TEST_RGB, TEST_IR)

# ============================================================
# REPORT
# ============================================================

print("=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)

print(f"Total Images : {total}")
print(f"Train        : {len(train_pairs)}")
print(f"Validation   : {len(val_pairs)}")
print(f"Test         : {len(test_pairs)}")