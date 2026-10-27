"""
Pretrained GAN Model Loader and Inference Engine
Supports Progressive GAN (PGAN) models via PyTorch Hub and self-contained
Deep Convolutional GAN (DCGAN) generators with reproducible latent sampling.
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

logger = logging.getLogger(__name__)

# Monkey-patch Adam __init__ to support integer betas from legacy PyTorch 1.x models
# Facebook's pytorch_GAN_zoo used betas=[0, 0.99] which raises ValueError in PyTorch 2.x
_orig_adam_init = optim.Adam.__init__


def _patched_adam_init(self, params, lr=1e-3, betas=(0.9, 0.999), *args, **kwargs):
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas = (float(betas[0]), float(betas[1]))
    return _orig_adam_init(self, params, lr=lr, betas=betas, *args, **kwargs)


optim.Adam.__init__ = _patched_adam_init


class DCGANGenerator(nn.Module):
    """
    Standard Deep Convolutional GAN (DCGAN) Generator.
    Maps a latent vector z ~ N(0, 1) of shape (N, nz, 1, 1) to RGB images (N, 3, 64, 64).
    Uses transposed convolutions, batch normalization, and Tanh activation.
    """
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # Input: (nz) x 1 x 1 -> (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf * 8) x 4 x 4 -> (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf * 4) x 8 x 8 -> (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # DCGAN weight initialization as proposed by Radford et al.
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


SUPPORTED_MODELS = {
    "PGAN-DTD": {
        "name": "Progressive GAN - DTD Textures (128x128)",
        "type": "pgan",
        "hub_name": "DTD",
        "description": "Pretrained on Describable Textures Dataset. Generates intricate surface patterns and organic textures.",
        "resolution": "128x128",
        "latent_dim": 512,
        "is_pretrained": True
    },
    "PGAN-CelebAHQ": {
        "name": "Progressive GAN - CelebA-HQ Faces (512x512)",
        "type": "pgan",
        "hub_name": "celebAHQ-512",
        "description": "Pretrained on high-resolution celebrity portraits. Generates realistic synthetic human faces.",
        "resolution": "512x512",
        "latent_dim": 512,
        "is_pretrained": True
    },
    "DCGAN-Abstract": {
        "name": "Deep Convolutional GAN - Latent Manifold (64x64)",
        "type": "dcgan",
        "hub_name": None,
        "description": "Lightweight DCGAN architecture executing live transposed-convolutional neural synthesis without remote weight dependencies.",
        "resolution": "64x64",
        "latent_dim": 100,
        "is_pretrained": False
    }
}


def get_available_device() -> torch.device:
    """Detect CUDA GPU or fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_gan_model(
    model_key: str,
    device: Optional[torch.device] = None
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """
    Load the selected GAN model.
    Returns:
        model: Loaded PyTorch model instance or None
        meta: Model metadata dictionary or None
        error_msg: String explaining the failure if loading failed, else None
    """
    if model_key not in SUPPORTED_MODELS:
        return None, None, f"Unknown model key '{model_key}'. Available: {list(SUPPORTED_MODELS.keys())}"

    meta = SUPPORTED_MODELS[model_key]
    if device is None:
        device = get_available_device()

    use_gpu = (device.type == "cuda")

    try:
        if meta["type"] == "pgan":
            # Load Progressive GAN from PyTorch Hub with skip_validation to prevent GitHub rate-limiting
            model = torch.hub.load(
                'facebookresearch/pytorch_GAN_zoo:hub',
                'PGAN',
                model_name=meta["hub_name"],
                pretrained=meta["is_pretrained"],
                useGPU=use_gpu,
                skip_validation=True
            )
            return model, meta, None

        elif meta["type"] == "dcgan":
            model = DCGANGenerator(nz=meta["latent_dim"], ngf=64, nc=3)
            model.to(device)
            model.eval()
            return model, meta, None

        else:
            return None, None, f"Unsupported model type '{meta['type']}'"

    except Exception as exc:
        err = f"Failed to load '{meta['name']}': {type(exc).__name__}: {str(exc)}"
        logger.error(err, exc_info=True)
        return None, meta, err


def generate_gan_images(
    model: Any,
    model_key: str,
    num_images: int = 8,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> List[Image.Image]:
    """
    Generate synthetic images using actual PyTorch neural network inference.
    
    Args:
        model: Loaded GAN model instance.
        model_key: Key in SUPPORTED_MODELS.
        num_images: Number of images to generate (1 to 64).
        seed: Random seed for reproducible generation.
        device: PyTorch device (CPU or CUDA).
        
    Returns:
        List of PIL Image objects in RGB format.
    """
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")

    meta = SUPPORTED_MODELS[model_key]
    if device is None:
        device = get_available_device()

    # Set random seeds for reproducible generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pil_images = []

    with torch.no_grad():
        if meta["type"] == "pgan":
            # PGAN buildNoiseData creates latent Gaussian noise tensor
            noise, _ = model.buildNoiseData(num_images)
            raw_output = model.test(noise)  # Tensor shape: (N, 3, H, W)
            # PGAN outputs are roughly in [-1, 1], normalize to [0, 1]
            tensors = raw_output.detach().cpu()
            tensors = (tensors.clamp(-1.0, 1.0) + 1.0) / 2.0

        elif meta["type"] == "dcgan":
            nz = meta["latent_dim"]
            noise = torch.randn(num_images, nz, 1, 1, device=device)
            raw_output = model(noise)
            tensors = raw_output.detach().cpu()
            tensors = (tensors.clamp(-1.0, 1.0) + 1.0) / 2.0

        else:
            raise ValueError(f"Unsupported model type: {meta['type']}")

        # Convert tensors to uint8 (N, H, W, 3) numpy arrays and then PIL Images
        uint8_imgs = (tensors * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        for idx in range(num_images):
            arr = uint8_imgs[idx]
            pil_img = Image.fromarray(arr, mode="RGB")
            pil_images.append(pil_img)

    return pil_images
