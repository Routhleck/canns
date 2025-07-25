"""
Test that hierarchical model refactoring maintains functionality.

This module tests that the split hierarchical model classes still work
correctly after being refactored into separate modules.
"""

import pytest
import brainstate

from canns.models.basic.hierarchical_model import (
    GaussRecUnits, NonRecUnits, BandCell, GridCell,
    HierarchicalPathIntegrationModel, HierarchicalNetwork
)

# Also test direct imports from split modules
from canns.models.basic.units import GaussRecUnits as DirectGRU
from canns.models.basic.band_cell import BandCell as DirectBC
from canns.models.basic.grid_cell import GridCell as DirectGC
from canns.models.basic.hierarchical_integration import (
    HierarchicalPathIntegrationModel as DirectHPIM
)


class TestHierarchicalRefactoring:
    """Test that the hierarchical model refactoring works correctly."""
    
    def test_backward_compatibility_imports(self):
        """Test that original imports still work."""
        # All classes should be importable from original location
        assert GaussRecUnits is not None
        assert NonRecUnits is not None
        assert BandCell is not None
        assert GridCell is not None
        assert HierarchicalPathIntegrationModel is not None
        assert HierarchicalNetwork is not None
    
    def test_direct_imports_work(self):
        """Test that direct imports from split modules work."""
        assert DirectGRU is not None
        assert DirectBC is not None
        assert DirectGC is not None
        assert DirectHPIM is not None
    
    def test_classes_are_identical(self):
        """Test that imported classes are the same objects."""
        # Classes imported from hierarchical_model should be identical
        # to those imported directly from split modules
        assert GaussRecUnits is DirectGRU
        assert BandCell is DirectBC
        assert GridCell is DirectGC
        assert HierarchicalPathIntegrationModel is DirectHPIM
    
    def test_basic_instantiation(self):
        """Test that classes can be instantiated."""
        # Test basic instantiation (without full setup)
        gauss_units = GaussRecUnits(size=10, tau=1.0)
        assert gauss_units.size == 10
        assert gauss_units.tau == 1.0
        
        non_rec_units = NonRecUnits(size=5, tau=2.0)
        assert non_rec_units.size == 5
        assert non_rec_units.tau == 2.0
    
    def test_band_cell_instantiation(self):
        """Test BandCell instantiation."""
        # BandCell requires angle and spacing parameters
        band_cell = BandCell(
            angle=0.0,
            spacing=1.0,
            size=60,
            noise=0.1
        )
        
        assert hasattr(band_cell, 'size')
        assert band_cell.size == 60
    
    def test_grid_cell_instantiation(self):
        """Test GridCell instantiation."""
        grid_cell = GridCell(
            num=100,
            angle=0.0,
            spacing=1.0,
            tau=1.0,
            k=0.1,
            a=0.5
        )
        
        assert grid_cell.num == 100
        assert grid_cell.tau == 1.0
    
    def test_class_methods_exist(self):
        """Test that key methods exist on classes."""
        # GaussRecUnits methods
        gauss_units = GaussRecUnits(size=10)
        assert hasattr(gauss_units, 'update')
        assert hasattr(gauss_units, 'get_center')
        assert hasattr(gauss_units, 'get_stimulus_by_pos')
        assert hasattr(gauss_units, 'make_conn')
        
        # GridCell methods
        grid_cell = GridCell(num=50, angle=0.0, spacing=1.0)
        assert hasattr(grid_cell, 'update')
        assert hasattr(grid_cell, 'get_center')
        assert hasattr(grid_cell, 'make_conn')
        assert hasattr(grid_cell, 'dist')


if __name__ == "__main__":
    pytest.main([__file__])