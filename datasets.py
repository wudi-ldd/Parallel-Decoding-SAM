import os
import cv2
import torch
from torch.utils.data import Dataset


def read_split_files(file_path):
    """Load newline-separated filenames for a dataset split.

    Args:
        file_path (str | os.PathLike): Path to a text file containing one filename per line.

    Returns:
        list[str]: Ordered list of filenames from the split file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        file_names = f.read().strip().split('\n')
    return file_names


class SegmentationDataset(Dataset):
    """Dataset wrapper that aligns pore masks with SAM preprocessing.

    Args:
        image_dir (str | os.PathLike): Directory containing RGB image files.
        mask_dir (str | os.PathLike): Directory containing binary mask PNG files.
        sam_model: SAM model instance providing `preprocess` for images.
        file_list (Sequence[str]): Filenames (without extension changes) to include.
        mask_size (tuple[int, int], optional): Target spatial size ``(H, W)`` for masks. Defaults to ``(1024, 1024)``.
        device (str | torch.device, optional): Device on which tensors are materialised. Defaults to ``'cpu'``.
    """

    def __init__(self, image_dir, mask_dir, sam_model, file_list, mask_size=(1024, 1024), device='cpu'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sam_model = sam_model
        self.mask_size = mask_size
        self.device = device
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith('.png') and f.replace('.png', '') in file_list
        ]

    def __len__(self):
        """Return the number of image/mask pairs available in the split."""
        return len(self.image_files)

    @staticmethod
    def _choose_interp_for_image(src_hw, dst_hw):
        """Select an OpenCV interpolation mode based on whether we downscale or upscale.

        Args:
            src_hw (tuple[int, int]): Source height and width.
            dst_hw (tuple[int, int]): Target height and width.

        Returns:
            int: OpenCV interpolation flag (`cv2.INTER_AREA` or `cv2.INTER_LINEAR`).
        """
        sh, sw = src_hw
        dh, dw = dst_hw
        if sh * sw > dh * dw:
            return cv2.INTER_AREA
        return cv2.INTER_LINEAR

    def __getitem__(self, idx):
        """Load an image/mask pair and produce SAM-ready tensors.

        Args:
            idx (int): Index inside the filtered :attr:`image_files` list.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Preprocessed RGB tensor of shape ``(3, 1024, 1024)`` and
            the corresponding binary mask tensor of shape ``(1024, 1024)``.
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        img_interp = self._choose_interp_for_image((h, w), (1024, 1024))
        image = cv2.resize(image, (1024, 1024), interpolation=img_interp)

        mask_path = os.path.join(self.mask_dir, image_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)

        input_image_torch = torch.as_tensor(image, dtype=torch.float32).to(self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        input_image = self.sam_model.preprocess(input_image_torch)

        mask_tensor = torch.as_tensor(mask, dtype=torch.float32).to(self.device)

        return input_image, mask_tensor
