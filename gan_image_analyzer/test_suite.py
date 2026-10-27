"""
Comprehensive Unit Test Suite for GAN Image Quality & Diversity Analyzer
Tests preprocessing, perceptual hashing, quality metrics, diversity heuristics,
and PyTorch GAN model execution.
"""

import unittest
import numpy as np
from PIL import Image

from preprocessing import (
    compute_sha256_bytes,
    compute_sha256_image,
    compute_dhash,
    compute_ahash,
    validate_and_load_image,
    package_generated_images
)
from analyzer import (
    compute_laplacian_sharpness,
    compute_gradient_energy,
    analyze_dataset_quality,
    compute_pairwise_dhash_similarity,
    evaluate_diversity_report
)
from model import (
    load_gan_model,
    generate_gan_images,
    DCGANGenerator,
    SUPPORTED_MODELS
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a test image
        self.img1 = Image.new("RGB", (64, 64), color=(128, 64, 32))
        self.img2 = Image.new("RGB", (64, 64), color=(128, 64, 32))  # Identical
        # Slightly different image
        self.img3 = Image.new("RGB", (64, 64), color=(255, 255, 255))

    def test_sha256_identical(self):
        hash1 = compute_sha256_image(self.img1)
        hash2 = compute_sha256_image(self.img2)
        hash3 = compute_sha256_image(self.img3)
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)

    def test_dhash_computation(self):
        h1, bits1 = compute_dhash(self.img1)
        h2, bits2 = compute_dhash(self.img2)
        self.assertEqual(h1, h2)
        self.assertEqual(len(bits1), 64)
        self.assertTrue(np.array_equal(bits1, bits2))


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create patterned images with different sharpness
        smooth = np.zeros((100, 100), dtype=np.float32)
        noisy = np.random.RandomState(42).randint(0, 255, size=(100, 100)).astype(np.float32)
        self.smooth_sharpness = compute_laplacian_sharpness(smooth)
        self.noisy_sharpness = compute_laplacian_sharpness(noisy)

    def test_sharpness(self):
        self.assertEqual(self.smooth_sharpness, 0.0)
        self.assertGreater(self.noisy_sharpness, 100.0)

    def test_diversity_evaluation(self):
        # Test diverse set
        imgs = [Image.fromarray(np.random.RandomState(i).randint(0, 255, (32, 32, 3), dtype=np.uint8)) for i in range(6)]
        packaged = package_generated_images(imgs)
        sim_mat, dist_mat = compute_pairwise_dhash_similarity(packaged)
        rep = evaluate_diversity_report(packaged, sim_mat, dist_mat, similarity_threshold=0.85)
        self.assertEqual(rep["total_images"], 6)
        self.assertIn(rep["classification"], ["Appearing Diverse", "Showing Some Repetition", "Highly Repetitive"])

    def test_repetitive_detection(self):
        # Test completely duplicated set
        base = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 100)
        imgs = [base.copy() for _ in range(6)]
        packaged = package_generated_images(imgs)
        sim_mat, dist_mat = compute_pairwise_dhash_similarity(packaged)
        rep = evaluate_diversity_report(packaged, sim_mat, dist_mat, similarity_threshold=0.85)
        self.assertEqual(rep["classification"], "Highly Repetitive")
        self.assertEqual(rep["unique_sha256_count"], 1)


class TestModel(unittest.TestCase):
    def test_dcgan_generation(self):
        model, meta, err = load_gan_model("DCGAN-Abstract")
        self.assertIsNone(err)
        self.assertIsNotNone(model)
        imgs = generate_gan_images(model, "DCGAN-Abstract", num_images=2, seed=123)
        self.assertEqual(len(imgs), 2)
        self.assertEqual(imgs[0].size, (64, 64))

    def test_reproducibility_with_seed(self):
        model, _, _ = load_gan_model("DCGAN-Abstract")
        imgs_a = generate_gan_images(model, "DCGAN-Abstract", num_images=2, seed=777)
        imgs_b = generate_gan_images(model, "DCGAN-Abstract", num_images=2, seed=777)
        hash_a = compute_sha256_image(imgs_a[0])
        hash_b = compute_sha256_image(imgs_b[0])
        self.assertEqual(hash_a, hash_b)


if __name__ == "__main__":
    unittest.main()
