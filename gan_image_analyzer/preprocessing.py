"""
Preprocessing and Image Validation Module
Provides image validation, format normalization, ZIP file ingestion,
and cryptographic/perceptual hashing (SHA-256 and dHash).
"""

import io
import os
import zipfile
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}


def compute_sha256_bytes(raw_bytes: bytes) -> str:
    """Calculate SHA-256 hexadecimal hash from raw image bytes."""
    hasher = hashlib.sha256()
    hasher.update(raw_bytes)
    return hasher.hexdigest()


def compute_sha256_image(image: Image.Image) -> str:
    """
    Compute a consistent SHA-256 hash for a PIL Image by saving it to an in-memory PNG buffer.
    Used for generated images or images where raw bytes need a canonical representation.
    """
    buf = io.BytesIO()
    # Normalize image mode to RGB for consistent hashing
    rgb_img = image.convert("RGB")
    rgb_img.save(buf, format="PNG")
    return compute_sha256_bytes(buf.getvalue())


def compute_dhash(image: Image.Image, hash_size: int = 8) -> Tuple[str, np.ndarray]:
    """
    Calculate Difference Hash (dHash) for an image.
    
    Algorithm:
    1. Convert to grayscale.
    2. Resize to (hash_size + 1, hash_size), typically (9, 8).
    3. Compare adjacent horizontal pixels (pixel[col+1] > pixel[col]).
    4. Compute an (hash_size * hash_size)-bit boolean array.
    5. Convert bit array to a hexadecimal string.
    
    Returns:
        hex_hash (str): Hexadecimal string representation (e.g., 16 characters for 64 bits).
        bit_array (np.ndarray): 1D boolean numpy array of shape (hash_size * hash_size,).
    """
    # Resize to (width, height) = (hash_size + 1, hash_size) using bilinear interpolation
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int32)
    
    # Compute difference: True if right pixel is brighter than left pixel
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_array = diff.flatten()
    
    # Convert boolean array to hex string
    hex_str = ""
    for i in range(0, len(bit_array), 4):
        chunk = bit_array[i:i+4]
        val = sum([chunk[j] << (3 - j) for j in range(len(chunk))])
        hex_str += f"{val:x}"
        
    return hex_str, bit_array


def compute_ahash(image: Image.Image, hash_size: int = 8) -> Tuple[str, np.ndarray]:
    """
    Calculate Average Hash (aHash) for an image.
    1. Grayscale.
    2. Resize to (hash_size, hash_size).
    3. Calculate mean pixel value.
    4. Bit is 1 if pixel >= mean, else 0.
    """
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    avg = np.mean(pixels)
    bit_array = (pixels >= avg).flatten()
    
    hex_str = ""
    for i in range(0, len(bit_array), 4):
        chunk = bit_array[i:i+4]
        val = sum([chunk[j] << (3 - j) for j in range(len(chunk))])
        hex_str += f"{val:x}"
        
    return hex_str, bit_array


def validate_and_load_image(
    file_bytes: bytes,
    filename: str = "image.png"
) -> Optional[Dict[str, Any]]:
    """
    Validate and extract metadata from image bytes.
    Returns dictionary with image metadata or None if invalid.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        return None

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.load()  # Verify image integrity
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    # Convert to RGB if needed (e.g., RGBA, Grayscale, CMYK)
    if pil_img.mode != "RGB":
        rgb_img = pil_img.convert("RGB")
    else:
        rgb_img = pil_img.copy()

    width, height = rgb_img.size
    sha256 = compute_sha256_bytes(file_bytes)
    dhash_str, dhash_bits = compute_dhash(rgb_img)
    ahash_str, ahash_bits = compute_ahash(rgb_img)

    return {
        "filename": filename,
        "format": pil_img.format or ext.replace(".", "").upper() or "PNG",
        "original_mode": pil_img.mode,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3) if height > 0 else 1.0,
        "size_bytes": len(file_bytes),
        "sha256": sha256,
        "dhash": dhash_str,
        "dhash_bits": dhash_bits,
        "ahash": ahash_str,
        "ahash_bits": ahash_bits,
        "image": rgb_img
    }


def extract_images_from_zip(zip_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract and validate all supported images from a ZIP file archive.
    Ignores system files like __MACOSX and hidden files.
    """
    loaded_images = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                basename = os.path.basename(info.filename)
                if basename.startswith(".") or "__MACOSX" in info.filename:
                    continue
                ext = os.path.splitext(basename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    try:
                        file_bytes = zf.read(info.filename)
                        parsed = validate_and_load_image(file_bytes, filename=basename)
                        if parsed:
                            loaded_images.append(parsed)
                    except Exception:
                        continue
    except zipfile.BadZipFile:
        pass

    return loaded_images


def package_generated_images(
    images: List[Image.Image],
    prefix: str = "gan_gen"
) -> List[Dict[str, Any]]:
    """
    Format a list of generated PIL images into the standard metadata dictionary.
    """
    packaged = []
    for idx, img in enumerate(images):
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size
        
        # Save to buffer to get byte size and canonical SHA-256
        buf = io.BytesIO()
        rgb_img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        
        sha256 = compute_sha256_bytes(raw_bytes)
        dhash_str, dhash_bits = compute_dhash(rgb_img)
        ahash_str, ahash_bits = compute_ahash(rgb_img)

        packaged.append({
            "filename": f"{prefix}_{idx + 1:03d}.png",
            "format": "PNG",
            "original_mode": "RGB",
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3) if height > 0 else 1.0,
            "size_bytes": len(raw_bytes),
            "sha256": sha256,
            "dhash": dhash_str,
            "dhash_bits": dhash_bits,
            "ahash": ahash_str,
            "ahash_bits": ahash_bits,
            "image": rgb_img
        })
    return packaged
