import os
import cv2
import torch

from torch.utils.data import Dataset

class LandsatDataset(Dataset):

    def __init__(self, rgb_dir, ir_dir):

        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir

        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.ir_files = sorted(os.listdir(ir_dir))

        assert len(self.rgb_files) == len(self.ir_files)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):

        rgb_path = os.path.join(
            self.rgb_dir,
            self.rgb_files[idx]
        )

        ir_path = os.path.join(
            self.ir_dir,
            self.ir_files[idx]
        )

        rgb = cv2.imread(rgb_path)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
  

        # Resize both images
        rgb = cv2.resize(rgb, (256, 256))
        ir = cv2.resize(ir, (256, 256))

        rgb = rgb.astype("float32") / 255.0

        ir = ir.astype("float32") / 255.0

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        ir = torch.from_numpy(ir).unsqueeze(0)

        return ir, rgb