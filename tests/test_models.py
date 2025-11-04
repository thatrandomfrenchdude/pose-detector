#!/usr/bin/env python3
"""
Test Model Files
Tests for model file availability and validity
"""

import unittest
from pathlib import Path


class TestModelFiles(unittest.TestCase):
    """Test model files."""
    
    def setUp(self):
        self.model_path = Path("model/model.onnx")
        self.context_path = self.model_path.parent / f"{self.model_path.stem}_ctx.onnx"
    
    def test_model_directory_exists(self):
        """Test if model directory exists."""
        self.assertTrue(self.model_path.parent.exists(), "Model directory not found")
    
    def test_original_model_exists(self):
        """Test if original model file exists."""
        self.assertTrue(self.model_path.exists(), f"Model not found: {self.model_path}")
    
    def test_model_file_size(self):
        """Test if model file has reasonable size."""
        if self.model_path.exists():
            size_mb = self.model_path.stat().st_size / 1024 / 1024
            self.assertGreater(size_mb, 1, "Model file too small")
            self.assertLess(size_mb, 500, "Model file too large")
    
    def test_context_model_optional(self):
        """Test context model (optional)."""
        if self.context_path.exists():
            size_mb = self.context_path.stat().st_size / 1024 / 1024
            self.assertGreater(size_mb, 1, "Context model file too small")


if __name__ == '__main__':
    unittest.main()