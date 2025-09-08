"""
Tests for Place Cell Analysis Module

Tests the core functionality of place field detection, spatial information
calculation, and place field characterization functions.
"""

import numpy as np
import pytest
from canns.analyzer import place_cell_analysis


class TestComputeFiringField:
    """Test firing field computation functionality."""
    
    def test_basic_firing_field_computation(self):
        """Test basic firing field heatmap generation."""
        T, N = 100, 5
        activity = np.random.rand(T, N) * 10  # Random activity
        positions = np.random.rand(T, 2)  # Random positions in [0,1]
        
        heatmap = place_cell_analysis.compute_firing_field(
            activity, positions, width=1.0, height=1.0, n_bins_x=10, n_bins_y=10
        )
        
        assert heatmap.shape == (10, 10, N)
        assert np.all(heatmap >= 0)  # Non-negative firing rates
        assert np.any(heatmap > 0)   # Should have some activity
    
    def test_empty_activity(self):
        """Test with zero activity."""
        T, N = 50, 3
        activity = np.zeros((T, N))
        positions = np.random.rand(T, 2)
        
        heatmap = place_cell_analysis.compute_firing_field(
            activity, positions, width=1.0, height=1.0
        )
        
        assert np.all(heatmap == 0)
    
    def test_different_environment_sizes(self):
        """Test with different environment dimensions."""
        T, N = 100, 2
        activity = np.random.rand(T, N)
        positions = np.random.rand(T, 2) * [2.0, 1.5]  # Scale to [0,2] x [0,1.5]
        
        heatmap = place_cell_analysis.compute_firing_field(
            activity, positions, width=2.0, height=1.5
        )
        
        assert heatmap.shape[2] == N
        assert np.all(heatmap >= 0)


class TestPlaceScore:
    """Test place score computation."""
    
    def test_compute_place_score_basic(self):
        """Test basic place score computation."""
        T, N = 200, 10
        # Create activity with clear place fields for some neurons
        activity = np.zeros((T, N))
        positions = np.random.rand(T, 2)
        
        # Create a strong place field for neuron 0
        center_pos = np.array([0.5, 0.5])
        distances = np.linalg.norm(positions - center_pos, axis=1)
        activity[:, 0] = 10 * np.exp(-distances**2 / (2 * 0.1**2))
        
        place_scores, sigma_values = place_cell_analysis.compute_place_score(
            activity, positions
        )
        
        assert len(place_scores) == N
        assert len(sigma_values) == N
        assert np.all(place_scores[0] > place_scores[1:])  # Neuron 0 should have highest score
        assert place_scores[0] > 0.1  # Should be a decent score
        assert np.all(sigma_values >= 0)
    
    def test_place_score_with_precomputed_heatmap(self):
        """Test place score with precomputed heatmap."""
        T, N = 100, 5
        activity = np.random.rand(T, N)
        positions = np.random.rand(T, 2)
        
        # Precompute heatmap
        heatmap = place_cell_analysis.compute_firing_field(activity, positions, 1.0, 1.0)
        
        scores1, sigmas1 = place_cell_analysis.compute_place_score(
            activity, positions, heatmap=heatmap
        )
        scores2, sigmas2 = place_cell_analysis.compute_place_score(
            activity, positions
        )
        
        # Should give same results
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)
        np.testing.assert_array_almost_equal(sigmas1, sigmas2, decimal=5)


class TestPlaceCellSelection:
    """Test place cell selection functionality."""
    
    def test_select_place_cells(self):
        """Test place cell selection with threshold."""
        T, N = 300, 20
        activity = np.random.rand(T, N) * 0.5  # Low baseline activity
        positions = np.random.rand(T, 2)
        
        # Add clear place fields for a few neurons
        for i in [2, 7, 15]:
            center = np.random.rand(2)
            distances = np.linalg.norm(positions - center, axis=1)
            activity[:, i] += 8 * np.exp(-distances**2 / (2 * 0.15**2))
        
        place_scores, place_indices, num_place_cells = place_cell_analysis.select_place_cells(
            activity, positions, threshold=0.01
        )
        
        assert len(place_scores) == N
        assert num_place_cells >= 3  # Should find at least the 3 we added
        assert len(place_indices) == num_place_cells
        assert 2 in place_indices  # Should include our strong place cells
        assert 7 in place_indices
        assert 15 in place_indices
    
    def test_no_place_cells_found(self):
        """Test when no place cells meet threshold."""
        T, N = 100, 5
        activity = np.random.rand(T, N) * 0.1  # Very low, random activity
        positions = np.random.rand(T, 2)
        
        place_scores, place_indices, num_place_cells = place_cell_analysis.select_place_cells(
            activity, positions, threshold=0.5  # High threshold
        )
        
        assert len(place_scores) == N
        assert num_place_cells == 0
        assert len(place_indices) == 0


class TestPlaceFieldCenters:
    """Test place field center computation."""
    
    def test_compute_place_field_centers_peak_method(self):
        """Test place field center computation using peak method."""
        T, N = 200, 3
        activity = np.zeros((T, N))
        positions = np.random.rand(T, 2)
        
        # Create place fields at known locations
        centers_true = np.array([[0.3, 0.4], [0.7, 0.2], [0.5, 0.8]])
        
        for i, center in enumerate(centers_true):
            distances = np.linalg.norm(positions - center, axis=1)
            activity[:, i] = 10 * np.exp(-distances**2 / (2 * 0.1**2))
        
        place_indices = np.array([0, 1, 2])
        centers_computed = place_cell_analysis.compute_place_field_centers(
            activity, place_indices, positions, method="peak"
        )
        
        assert centers_computed.shape == (3, 2)
        
        # Centers should be reasonably close to true centers
        for i in range(3):
            distance = np.linalg.norm(centers_computed[i] - centers_true[i])
            assert distance < 0.2  # Within reasonable tolerance
    
    def test_compute_place_field_centers_centroid_method(self):
        """Test place field center computation using centroid method."""
        T, N = 150, 2
        activity = np.zeros((T, N))
        positions = np.random.rand(T, 2)
        
        # Create place field
        center_true = np.array([0.6, 0.3])
        distances = np.linalg.norm(positions - center_true, axis=1)
        activity[:, 0] = 8 * np.exp(-distances**2 / (2 * 0.12**2))
        
        centers = place_cell_analysis.compute_place_field_centers(
            activity, np.array([0]), positions, method="centroid"
        )
        
        assert centers.shape == (1, 2)
        distance = np.linalg.norm(centers[0] - center_true)
        assert distance < 0.15


class TestSpatialInformation:
    """Test spatial information computation."""
    
    def test_compute_spatial_information(self):
        """Test spatial information calculation."""
        T, N = 500, 8
        activity = np.random.rand(T, N) * 2
        positions = np.random.rand(T, 2)
        
        # Add spatially selective activity for some neurons
        for i in [1, 4]:
            center = np.random.rand(2)
            distances = np.linalg.norm(positions - center, axis=1)
            activity[:, i] += 15 * np.exp(-distances**2 / (2 * 0.1**2))
        
        spatial_info = place_cell_analysis.compute_spatial_information(
            activity, positions, n_bins=20
        )
        
        assert len(spatial_info) == N
        assert np.all(spatial_info >= 0)  # Spatial info should be non-negative
        # Neurons with place fields should have higher spatial info
        assert spatial_info[1] > np.mean(spatial_info)
        assert spatial_info[4] > np.mean(spatial_info)
    
    def test_uniform_activity_zero_information(self):
        """Test that uniform activity gives zero spatial information."""
        T, N = 200, 3
        # Uniform activity across space
        activity = np.ones((T, N)) * 5.0
        positions = np.random.rand(T, 2)
        
        spatial_info = place_cell_analysis.compute_spatial_information(
            activity, positions
        )
        
        # Should be very low (approximately zero)
        assert np.all(spatial_info < 0.01)


class TestPlaceFieldStability:
    """Test place field stability analysis."""
    
    def test_compute_place_field_stability(self):
        """Test place field stability between sessions."""
        T, N = 300, 5
        positions1 = np.random.rand(T, 2)
        positions2 = np.random.rand(T, 2)  # Different session
        
        # Create similar place fields in both sessions
        activity1 = np.zeros((T, N))
        activity2 = np.zeros((T, N))
        
        centers = [np.array([0.3, 0.7]), np.array([0.8, 0.2])]
        place_indices = np.array([0, 3])  # Only neurons 0 and 3 have place fields
        
        for i, idx in enumerate(place_indices):
            # Session 1
            distances1 = np.linalg.norm(positions1 - centers[i], axis=1)
            activity1[:, idx] = 10 * np.exp(-distances1**2 / (2 * 0.1**2))
            
            # Session 2 - similar but not identical
            distances2 = np.linalg.norm(positions2 - centers[i], axis=1)
            activity2[:, idx] = 9 * np.exp(-distances2**2 / (2 * 0.12**2))
        
        stability = place_cell_analysis.compute_place_field_stability(
            activity1, activity2, positions1, positions2, place_indices
        )
        
        assert len(stability) == len(place_indices)
        assert np.all(stability >= -1) and np.all(stability <= 1)  # Correlation bounds
        # Should have some positive correlation for place cells
        assert np.mean(stability) > -0.5


class TestAnalyzePlaceFieldProperties:
    """Test comprehensive place field analysis."""
    
    def test_analyze_place_field_properties(self):
        """Test comprehensive place field property analysis."""
        T, N = 400, 12
        activity = np.random.rand(T, N) * 1.0
        positions = np.random.rand(T, 2)
        
        # Add strong place fields for several neurons
        place_cell_indices = [2, 5, 8, 11]
        for i, idx in enumerate(place_cell_indices):
            center = np.array([0.2 + i*0.2, 0.3 + i*0.15])
            distances = np.linalg.norm(positions - center, axis=1)
            activity[:, idx] += 12 * np.exp(-distances**2 / (2 * 0.08**2))
        
        place_indices = np.array(place_cell_indices)
        results = place_cell_analysis.analyze_place_field_properties(
            activity, positions, place_indices
        )
        
        # Check that all expected keys are present
        expected_keys = ['place_indices', 'num_place_cells', 'centers', 
                        'spatial_info', 'heatmaps', 'place_scores', 
                        'field_sizes', 'peak_rates']
        for key in expected_keys:
            assert key in results
        
        # Check dimensions and values
        assert results['num_place_cells'] == len(place_indices)
        assert results['centers'].shape == (len(place_indices), 2)
        assert len(results['spatial_info']) == len(place_indices)
        assert len(results['place_scores']) == len(place_indices)
        assert len(results['field_sizes']) == len(place_indices)
        assert len(results['peak_rates']) == len(place_indices)
        
        # Values should be reasonable
        assert np.all(results['place_scores'] > 0)
        assert np.all(results['field_sizes'] > 0)
        assert np.all(results['peak_rates'] > 0)
        assert np.all(results['spatial_info'] >= 0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_position_input(self):
        """Test with single position (no movement)."""
        T, N = 100, 5
        activity = np.random.rand(T, N)
        positions = np.ones((T, 2)) * 0.5  # All same position
        
        # Should not crash
        heatmap = place_cell_analysis.compute_firing_field(activity, positions, 1.0, 1.0)
        assert heatmap.shape[2] == N
    
    def test_very_small_environment(self):
        """Test with very small environment."""
        T, N = 50, 3
        activity = np.random.rand(T, N)
        positions = np.random.rand(T, 2) * 0.01  # Very small space
        
        place_scores, _ = place_cell_analysis.compute_place_score(
            activity, positions, width=0.01, height=0.01
        )
        
        assert len(place_scores) == N
        assert np.all(np.isfinite(place_scores))
    
    def test_zero_time_steps(self):
        """Test with zero time steps."""
        activity = np.empty((0, 5))
        positions = np.empty((0, 2))
        
        # Should handle empty arrays gracefully
        heatmap = place_cell_analysis.compute_firing_field(activity, positions, 1.0, 1.0)
        assert heatmap.shape == (50, 50, 5)  # Default bins
        assert np.all(heatmap == 0)  # Should be all zeros


if __name__ == "__main__":
    pytest.main([__file__])