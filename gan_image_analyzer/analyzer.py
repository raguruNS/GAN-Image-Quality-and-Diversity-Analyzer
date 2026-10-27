"""
Image Quality and Diversity Analyzer Module
Computes pixel-level quality indicators (brightness, contrast, sharpness),
pairwise similarity matrices, duplicate clusters, and diversity classifications.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image


def compute_laplacian_sharpness(gray_array: np.ndarray) -> float:
    """
    Compute sharpness using the variance of the Discrete Laplacian operator.
    Approximates cv2.Laplacian(gray, cv2.CV_64F).var() purely in NumPy.
    
    Higher variance indicates more high-frequency spatial variation (sharper edges).
    Note: Sensor noise, checkerboard artifacts, and fine high-frequency noise can
    also inflate this metric.
    """
    img = gray_array.astype(np.float64)
    # Discrete Laplacian using 4-neighborhood
    # L[y, x] = img[y-1, x] + img[y+1, x] + img[y, x-1] + img[y, x+1] - 4 * img[y, x]
    laplacian = (
        img[:-2, 1:-1] +
        img[2:, 1:-1] +
        img[1:-1, :-2] +
        img[1:-1, 2:] -
        4.0 * img[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def compute_gradient_energy(gray_array: np.ndarray) -> float:
    """
    Compute average gradient magnitude (Tenengrad-like energy) using central differences.
    """
    img = gray_array.astype(np.float64)
    # Central difference gradients
    gx = (img[1:-1, 2:] - img[1:-1, :-2]) * 0.5
    gy = (img[2:, 1:-1] - img[:-2, 1:-1]) * 0.5
    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))


def analyze_single_image_quality(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive quality indicators for a single image item.
    """
    img: Image.Image = item["image"]
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    
    # 1. Brightness: ITU-R 601-2 standard luma or grayscale mean
    # Y = 0.299 R + 0.587 G + 0.114 B
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    
    # Under-exposed (< 25 / 255 = ~10%) and over-exposed (> 230 / 255 = ~90%)
    under_exposed_pct = float(np.mean(gray < 25.0) * 100.0)
    over_exposed_pct = float(np.mean(gray > 230.0) * 100.0)
    
    # 2. Contrast: RMS contrast (Root Mean Square standard deviation of pixel intensities)
    rms_contrast = float(np.std(rgb))
    dynamic_range = float(np.max(gray) - np.min(gray))
    
    # 3. Sharpness
    laplacian_var = compute_laplacian_sharpness(gray)
    grad_energy = compute_gradient_energy(gray)
    
    # 4. Colorfulness / Saturation
    # HSV saturation mean
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    
    return {
        "index": item.get("index", 0),
        "filename": item["filename"],
        "dimensions": f"{item['width']}x{item['height']}",
        "width": item["width"],
        "height": item["height"],
        "aspect_ratio": item["aspect_ratio"],
        "size_kb": round(item["size_bytes"] / 1024, 2),
        "format": item["format"],
        "brightness_mean": round(brightness_mean, 2),
        "brightness_std": round(brightness_std, 2),
        "under_exposed_pct": round(under_exposed_pct, 2),
        "over_exposed_pct": round(over_exposed_pct, 2),
        "rms_contrast": round(rms_contrast, 2),
        "dynamic_range": round(dynamic_range, 1),
        "sharpness_laplacian": round(laplacian_var, 2),
        "gradient_energy": round(grad_energy, 2),
        "mean_saturation": round(mean_saturation, 2),
        "sha256": item["sha256"],
        "dhash": item["dhash"]
    }


def analyze_dataset_quality(images: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate a DataFrame with per-image quality diagnostics.
    """
    records = []
    for idx, item in enumerate(images):
        item["index"] = idx + 1
        record = analyze_single_image_quality(item)
        records.append(record)
    return pd.DataFrame(records)


def compute_pairwise_dhash_similarity(images: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate pairwise Hamming distance and dHash similarity matrix for all image pairs.
    
    Similarity Formula:
        similarity = 1.0 - (hamming_distance / total_bits)
        where total_bits = 64 for 8x8 dHash.
    
    Returns:
        similarity_matrix (np.ndarray): Shape (N, N), diagonal = 1.0, values in [0.0, 1.0].
        distance_matrix (np.ndarray): Shape (N, N), diagonal = 0, integer Hamming distances in [0, 64].
    """
    n = len(images)
    bit_arrays = np.array([img["dhash_bits"] for img in images], dtype=bool)  # (N, 64)
    total_bits = bit_arrays.shape[1]
    
    # Efficient pairwise Hamming distance:
    # (A != B).sum(axis=1) across all pairs
    # Using XOR broadcasting: bit_arrays[:, None, :] ^ bit_arrays[None, :, :]
    xor_mat = bit_arrays[:, None, :] ^ bit_arrays[None, :, :]  # (N, N, 64)
    distance_matrix = np.sum(xor_mat, axis=-1)  # (N, N)
    
    similarity_matrix = 1.0 - (distance_matrix / float(total_bits))
    return similarity_matrix, distance_matrix


def find_similar_and_duplicate_pairs(
    images: List[Dict[str, Any]],
    similarity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    similarity_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Identify exact duplicates, near-duplicate pairs above the threshold,
    and rank the most similar pairs.
    """
    n = len(images)
    exact_duplicates_sha256 = []
    exact_duplicates_dhash = []
    near_duplicate_pairs = []
    all_pairs = []

    # Map SHA-256 to track byte-level duplicates
    sha_map: Dict[str, List[int]] = {}
    for idx, item in enumerate(images):
        sha = item["sha256"]
        sha_map.setdefault(sha, []).append(idx)
        
    for sha, indices in sha_map.items():
        if len(indices) > 1:
            exact_duplicates_sha256.append({
                "sha256": sha,
                "indices": [i + 1 for i in indices],
                "filenames": [images[i]["filename"] for i in indices]
            })

    # Evaluate pairwise similarities for all unique pairs (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            dist = int(distance_matrix[i, j])
            is_exact_sha = images[i]["sha256"] == images[j]["sha256"]
            is_exact_dhash = (dist == 0)

            pair_info = {
                "idx_a": i + 1,
                "idx_b": j + 1,
                "file_a": images[i]["filename"],
                "file_b": images[j]["filename"],
                "similarity": round(sim, 4),
                "hamming_distance": dist,
                "is_exact_sha256": is_exact_sha,
                "is_exact_dhash": is_exact_dhash,
                "img_a": images[i]["image"],
                "img_b": images[j]["image"]
            }
            all_pairs.append(pair_info)

            if is_exact_dhash:
                exact_duplicates_dhash.append(pair_info)

            if sim >= similarity_threshold:
                near_duplicate_pairs.append(pair_info)

    # Sort pairs by similarity descending
    all_pairs.sort(key=lambda p: p["similarity"], reverse=True)
    near_duplicate_pairs.sort(key=lambda p: p["similarity"], reverse=True)

    return {
        "exact_duplicates_sha256": exact_duplicates_sha256,
        "exact_duplicates_dhash": exact_duplicates_dhash,
        "near_duplicate_pairs": near_duplicate_pairs,
        "all_pairs_ranked": all_pairs
    }


def evaluate_diversity_report(
    images: List[Dict[str, Any]],
    similarity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    similarity_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Synthesize image diversity diagnostics, similarity statistics,
    and automatic heuristic classification.
    """
    n = len(images)
    if n <= 1:
        return {
            "total_images": n,
            "unique_sha256_count": n,
            "unique_dhash_count": n,
            "uniqueness_ratio_sha": 1.0,
            "uniqueness_ratio_dhash": 1.0,
            "classification": "Appearing Diverse",
            "classification_color": "green",
            "rationale": "Only one image provided; insufficient sample size for diversity assessment.",
            "mean_similarity": 1.0,
            "median_similarity": 1.0,
            "std_similarity": 0.0,
            "max_similarity": 1.0,
            "min_similarity": 1.0,
            "near_duplicate_count": 0,
            "near_duplicate_pct": 0.0,
            "pairwise_similarities": []
        }

    pair_results = find_similar_and_duplicate_pairs(
        images, similarity_matrix, distance_matrix, similarity_threshold
    )
    
    unique_sha_count = len({item["sha256"] for item in images})
    unique_dhash_count = len({item["dhash"] for item in images})
    
    uniqueness_ratio_sha = unique_sha_count / n
    uniqueness_ratio_dhash = unique_dhash_count / n

    # Extract upper triangle values (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_similarities = similarity_matrix[triu_indices]
    total_pairs = len(pairwise_similarities)

    mean_sim = float(np.mean(pairwise_similarities))
    median_sim = float(np.median(pairwise_similarities))
    std_sim = float(np.std(pairwise_similarities))
    max_sim = float(np.max(pairwise_similarities))
    min_sim = float(np.min(pairwise_similarities))

    near_dup_count = len(pair_results["near_duplicate_pairs"])
    near_dup_pct = (near_dup_count / total_pairs * 100.0) if total_pairs > 0 else 0.0

    # Heuristic Classification Rule
    # Criteria:
    # 1. "Highly Repetitive" if:
    #    - Uniqueness ratio (dHash) < 0.70 OR
    #    - More than 25% of all pairs exceed similarity threshold OR
    #    - Mean similarity > 0.80
    # 2. "Showing Some Repetition" if:
    #    - Uniqueness ratio between 0.70 and 0.92 OR
    #    - 5% to 25% of pairs exceed similarity threshold OR
    #    - Mean similarity between 0.68 and 0.80
    # 3. "Appearing Diverse" if:
    #    - Uniqueness ratio >= 0.92 AND
    #    - Less than 5% pairs exceed similarity threshold AND
    #    - Mean similarity < 0.68
    
    if uniqueness_ratio_dhash < 0.70 or near_dup_pct >= 25.0 or mean_sim >= 0.80:
        classification = "Highly Repetitive"
        classification_color = "error"  # Red
        rationale = (
            f"Significant sample repetition detected: {unique_dhash_count}/{n} unique perceptual hashes "
            f"({uniqueness_ratio_dhash*100:.1f}%), {near_dup_pct:.1f}% of image pairs exceed similarity "
            f"threshold {similarity_threshold:.2f}, with average similarity of {mean_sim:.2f}. "
            f"This suggests strong visual clustering or potential mode collapse."
        )
    elif uniqueness_ratio_dhash < 0.92 or near_dup_pct >= 5.0 or mean_sim >= 0.68:
        classification = "Showing Some Repetition"
        classification_color = "warning"  # Yellow/Orange
        rationale = (
            f"Moderate visual variation: {unique_dhash_count}/{n} unique hashes "
            f"({uniqueness_ratio_dhash*100:.1f}%), {near_dup_count} near-duplicate pair(s) "
            f"({near_dup_pct:.1f}% of pairs), with an average similarity of {mean_sim:.2f}. "
            f"Minor redundancy or localized pattern repetition is present."
        )
    else:
        classification = "Appearing Diverse"
        classification_color = "success"  # Green
        rationale = (
            f"High visual variety: {unique_dhash_count}/{n} unique perceptual hashes "
            f"({uniqueness_ratio_dhash*100:.1f}%), {near_dup_count} near-duplicate pair(s) "
            f"({near_dup_pct:.1f}% of pairs), and average pairwise similarity of {mean_sim:.2f}."
        )

    return {
        "total_images": n,
        "total_pairs": total_pairs,
        "unique_sha256_count": unique_sha_count,
        "unique_dhash_count": unique_dhash_count,
        "uniqueness_ratio_sha": round(uniqueness_ratio_sha, 3),
        "uniqueness_ratio_dhash": round(uniqueness_ratio_dhash, 3),
        "classification": classification,
        "classification_color": classification_color,
        "rationale": rationale,
        "mean_similarity": round(mean_sim, 4),
        "median_similarity": round(median_sim, 4),
        "std_similarity": round(std_sim, 4),
        "max_similarity": round(max_sim, 4),
        "min_similarity": round(min_sim, 4),
        "near_duplicate_count": near_dup_count,
        "near_duplicate_pct": round(near_dup_pct, 2),
        "similarity_threshold": similarity_threshold,
        "exact_duplicates_sha256": pair_results["exact_duplicates_sha256"],
        "exact_duplicates_dhash": pair_results["exact_duplicates_dhash"],
        "near_duplicate_pairs": pair_results["near_duplicate_pairs"],
        "all_pairs_ranked": pair_results["all_pairs_ranked"],
        "pairwise_similarities": pairwise_similarities
    }
