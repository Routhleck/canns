"""Tests for version management and parsing."""

import pytest
from canns._version import parse_version_info


class TestVersionParsing:
    """Test version string parsing functionality."""
    
    def test_standard_versions(self):
        """Test parsing of standard semantic versions."""
        assert parse_version_info("0.5.1") == (0, 5, 1)
        assert parse_version_info("1.0.0") == (1, 0, 0)
        assert parse_version_info("2.3.4") == (2, 3, 4)
    
    def test_dev_versions(self):
        """Test parsing of development versions."""
        assert parse_version_info("0.5.1+dev") == (0, 5, 1)
        assert parse_version_info("0.5.1.dev6") == (0, 5, 1)
        assert parse_version_info("0.5.1+dev6") == (0, 5, 1)
        assert parse_version_info("2.3.4.dev10") == (2, 3, 4)
    
    def test_git_versions(self):
        """Test parsing of git-based versions."""
        assert parse_version_info("0.1.0+git.abc123") == (0, 1, 0)
        assert parse_version_info("0.5.1.dev6+g12345678") == (0, 5, 1)
        assert parse_version_info("1.2.3+dirty") == (1, 2, 3)
    
    def test_edge_cases(self):
        """Test edge cases and malformed versions."""
        # Should handle missing patch version
        assert parse_version_info("1.0") == (1, 0, 0)
        
        # Should extract first three numeric parts
        assert parse_version_info("1.2.3.4.5") == (1, 2, 3)
        
        # Should handle versions with non-numeric suffixes
        assert parse_version_info("1.2.3alpha") == (1, 2, 3)


class TestVersionImport:
    """Test version import and availability."""
    
    def test_version_import(self):
        """Test that version can be imported from package."""
        import canns
        
        # Should have version attributes
        assert hasattr(canns, '__version__')
        assert hasattr(canns, 'version_info')
        
        # Version should be a string
        assert isinstance(canns.__version__, str)
        
        # Version info should be a tuple of 3 integers
        assert isinstance(canns.version_info, tuple)
        assert len(canns.version_info) == 3
        assert all(isinstance(x, int) for x in canns.version_info)
    
    def test_version_consistency(self):
        """Test that version string and version_info are consistent."""
        import canns
        
        # Parse the version string and compare with version_info
        parsed = parse_version_info(canns.__version__)
        assert parsed == canns.version_info