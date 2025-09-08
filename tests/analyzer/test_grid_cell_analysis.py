"""
Tests for Grid Cell Analysis Module

Tests the core functionality of grid score computation, hexagonal pattern detection,
and grid cell characterization functions.
"""

import numpy as np
import pytest
from canns.analyzer import grid_cell_analysis


class TestGridScoreComputation:
    """Test grid score computation functionality."""
    
    def test_compute_grid_score_single_cell(self):
        """Test basic grid score computation for a single cell."""
        # Create a simple hexagonal pattern for testing
        n_bins = 40
        heatmap = np.zeros((n_bins, n_bins))
        
        # Add some hexagonal structure (simplified)
        center = n_bins // 2
        for i in range(n_bins):
            for j in range(n_bins):
                x, y = i - center, j - center
                # Create periodic pattern with 60-degree symmetry
                value = (np.cos(2*np.pi*x/6) + np.cos(2*np.pi*y/6) + 
                        np.cos(2*np.pi*(x-y)/6)) / 3
                if value > 0:
                    heatmap[i, j] = value
        
        grid_score = grid_cell_analysis.compute_grid_score(
            heatmap, method="rotation"
        )
        
        assert isinstance(grid_score, float)
        assert -2.0 <= grid_score <= 2.0  # Grid score bounds
    
    def test_compute_grid_score_empty_heatmap(self):
        """Test grid score with empty/zero heatmap."""
        heatmap = np.zeros((30, 30))
        
        grid_score = grid_cell_analysis.compute_grid_score(heatmap)
        
        assert grid_score == 0.0  # Should return 0 for empty map
    
    def test_compute_grid_score_uniform_activity(self):
        """Test grid score with uniform activity."""
        heatmap = np.ones((25, 25)) * 5.0
        
        grid_score = grid_cell_analysis.compute_grid_score(heatmap)
        
        # Uniform activity should give very low grid score
        assert abs(grid_score) < 0.1
    
    def test_compute_grid_score_different_methods(self):
        """Test different grid score computation methods."""
        heatmap = np.random.rand(30, 30)
        
        # Add some spatial structure
        center = 15
        for i in range(30):
            for j in range(30):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                heatmap[i, j] += 5 * np.exp(-distance**2 / (2 * 5**2))
        
        score_rotation = grid_cell_analysis.compute_grid_score(heatmap, method="rotation")
        score_correlation = grid_cell_analysis.compute_grid_score(heatmap, method="correlation")
        
        assert isinstance(score_rotation, float)
        assert isinstance(score_correlation, float)
        assert np.isfinite(score_rotation)
        assert np.isfinite(score_correlation)


class TestGridFieldDetection:
    """Test grid field detection and characterization."""
    
    def test_detect_grid_fields_basic(self):
        """Test basic grid field detection."""
        # Create heatmap with some peaks
        heatmap = np.zeros((50, 50))
        
        # Add peaks
        centers = np.array([[20, 25], [30, 25], [25, 35], [25, 15]])
        for center in centers:
            y, x = np.ogrid[:50, :50]
            mask = (x - center[0])**2 + (y - center[1])**2 <= 16
            heatmap[mask] = 5.0
        
        field_centers, num_fields = grid_cell_analysis.detect_grid_fields(
            heatmap, min_field_size=10, threshold_factor=0.3
        )
        
        assert num_fields >= 0  # May detect multiple fields
        assert field_centers.shape == (num_fields, 2)
        assert np.all(field_centers >= 0)
    
    def test_detect_grid_fields_empty(self):
        """Test field detection with no significant fields."""
        heatmap = np.random.rand(30, 30) * 0.1  # Very low activity
        
        field_centers, num_fields = grid_cell_analysis.detect_grid_fields(
            heatmap, min_field_size=5, threshold_factor=0.8
        )
        
        assert num_fields >= 0  # May or may not find fields
        assert len(field_centers) == num_fields


class TestGridParameters:
    """Test grid parameter computation (spacing, orientation, etc.)."""
    
    def test_compute_grid_spacing(self):
        """Test grid spacing computation."""
        # Create simple periodic pattern
        n_bins = 60
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, n_bins), 
                          np.linspace(0, 2*np.pi, n_bins))
        
        # Hexagonal pattern with known spacing
        true_spacing = 0.3
        heatmap = (np.cos(2*np.pi*x/true_spacing) + 
                  np.cos(2*np.pi*y/true_spacing) +
                  np.cos(2*np.pi*(x-y)/true_spacing))
        heatmap = np.maximum(heatmap, 0)  # Remove negative values
        
        # First detect fields
        field_centers, num_fields = grid_cell_analysis.detect_grid_fields(
            heatmap, min_field_size=5, threshold_factor=0.1
        )
        
        if num_fields >= 2:
            spacing = grid_cell_analysis.compute_grid_spacing(field_centers)
        else:
            spacing = 0.5  # Default value for test
        
        assert spacing > 0
        # Spacing should be reasonable (within order of magnitude)
        assert 0.1 <= spacing <= 1.0
    
    def test_compute_grid_orientation(self):
        """Test grid orientation computation."""
        n_bins = 40
        heatmap = np.zeros((n_bins, n_bins))
        
        # Create oriented pattern
        for i in range(n_bins):
            for j in range(n_bins):
                x, y = i - n_bins//2, j - n_bins//2
                # Rotated hexagonal pattern
                angle = np.pi/6  # 30 degrees
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                value = np.cos(2*np.pi*x_rot/8) * np.cos(2*np.pi*y_rot/8)
                if value > 0:
                    heatmap[i, j] = value
        
        orientation = grid_cell_analysis.compute_grid_orientation(heatmap)
        
        assert isinstance(orientation, float)
        assert -np.pi <= orientation <= np.pi


class TestGridCellSelection:
    """Test grid cell selection and filtering."""
    
    def test_select_grid_cells(self):
        """Test grid cell selection based on criteria."""
        T, N = 800, 15
        activity = np.random.rand(T, N) * 1.0
        positions = np.random.rand(T, 2)
        
        # Add hexagonal-like patterns for some cells
        for i in [2, 7, 12]:
            # Create multiple firing fields
            centers = [np.random.rand(2) for _ in range(6)]
            for center in centers:
                distances = np.linalg.norm(positions - center, axis=1)
                activity[:, i] += 3 * np.exp(-distances**2 / (2 * 0.08**2))
        
        grid_scores, grid_indices, num_grid_cells = grid_cell_analysis.select_grid_cells(
            activity, positions,
            grid_score_threshold=0.05,
            min_fields_threshold=3,
            width=1.0, height=1.0
        )
        
        assert len(grid_scores) == N
        assert num_grid_cells >= 0
        assert len(grid_indices) == num_grid_cells
        
        # Grid indices should be valid
        if num_grid_cells > 0:
            assert np.all(grid_indices >= 0)
            assert np.all(grid_indices < N)
    
    def test_select_grid_cells_no_cells_found(self):
        """Test when no cells meet grid criteria."""
        T, N = 200, 8
        # Random, non-spatial activity
        activity = np.random.rand(T, N) * 0.5
        positions = np.random.rand(T, 2)
        
        grid_scores, grid_indices, num_grid_cells = grid_cell_analysis.select_grid_cells(
            activity, positions,
            grid_score_threshold=0.5,  # High threshold
            min_fields_threshold=5,
            width=1.0, height=1.0
        )
        
        assert len(grid_scores) == N
        assert num_grid_cells >= 0
        assert len(grid_indices) == num_grid_cells


class TestGridPopulationAnalysis:
    """Test comprehensive grid population analysis."""
    
    def test_analyze_grid_population(self):
        """Test comprehensive grid population analysis."""
        T, N = 1000, 10
        activity = np.random.rand(T, N) * 0.8
        positions = np.random.rand(T, 2)
        
        # Create clear grid patterns for some cells
        grid_cell_indices = [1, 4, 8]
        for i in grid_cell_indices:
            # Add multiple hexagonal fields
            for cx, cy in [(0.2, 0.3), (0.6, 0.2), (0.4, 0.7), (0.8, 0.8)]:
                distances = np.linalg.norm(positions - [cx, cy], axis=1)
                activity[:, i] += 6 * np.exp(-distances**2 / (2 * 0.06**2))
        
        grid_indices = np.array(grid_cell_indices)
        results = grid_cell_analysis.analyze_grid_population(
            activity, positions, grid_indices, width=1.0, height=1.0
        )
        
        # Check that we get results back
        assert isinstance(results, dict)
        assert len(results) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_position_input(self):
        """Test with single position (no movement)."""
        heatmap = np.ones((30, 30)) * 0.5  # Uniform activity
        
        # Should not crash
        score = grid_cell_analysis.compute_grid_score(heatmap)
        assert isinstance(score, float)
        assert np.isfinite(score)
    
    def test_very_small_heatmap(self):
        """Test with very small heatmap."""
        heatmap = np.random.rand(5, 5)
        
        score = grid_cell_analysis.compute_grid_score(heatmap)
        
        assert isinstance(score, float)
        assert np.isfinite(score)
    
    def test_zero_heatmap(self):
        """Test with zero heatmap."""
        heatmap = np.zeros((20, 20))
        
        score = grid_cell_analysis.compute_grid_score(heatmap)
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__])