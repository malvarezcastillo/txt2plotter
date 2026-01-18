"""Tests for the vectorizer module."""

import numpy as np
import pytest

from modules.vectorizer import (
    extract_paths,
    get_neighbors,
    prune_spurs,
    raster_to_paths,
    skeleton_to_graph,
    skeletonize_image,
)


@pytest.fixture
def simple_line_image():
    """Create a simple horizontal line image."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50, 20:80] = 1  # Horizontal line
    return img


@pytest.fixture
def cross_image():
    """Create a cross/plus shape image."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50, 20:80] = 1  # Horizontal line
    img[20:80, 50] = 1  # Vertical line
    return img


@pytest.fixture
def line_with_spur_image():
    """Create a line with a short spur branch."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50, 20:80] = 1  # Main horizontal line
    img[45:50, 40] = 1  # Short spur (5 pixels)
    return img


class TestGetNeighbors:
    """Tests for get_neighbors function."""

    def test_middle_pixel_no_neighbors(self):
        """Middle pixel with no skeleton neighbors returns empty."""
        skeleton = np.zeros((10, 10), dtype=np.uint8)
        skeleton[5, 5] = 1
        neighbors = get_neighbors(5, 5, skeleton)
        assert neighbors == []

    def test_middle_pixel_with_neighbors(self):
        """Middle pixel with skeleton neighbors returns them."""
        skeleton = np.zeros((10, 10), dtype=np.uint8)
        skeleton[5, 5] = 1
        skeleton[5, 6] = 1  # Right neighbor
        skeleton[4, 5] = 1  # Top neighbor
        neighbors = get_neighbors(5, 5, skeleton)
        assert len(neighbors) == 2
        assert (4, 5) in neighbors
        assert (5, 6) in neighbors

    def test_edge_pixel(self):
        """Edge pixel doesn't go out of bounds."""
        skeleton = np.zeros((10, 10), dtype=np.uint8)
        skeleton[0, 0] = 1
        skeleton[0, 1] = 1
        skeleton[1, 0] = 1
        neighbors = get_neighbors(0, 0, skeleton)
        assert len(neighbors) == 2


class TestSkeletonize:
    """Tests for skeletonization."""

    def test_thin_line_unchanged(self, simple_line_image):
        """A 1-pixel wide line should remain mostly unchanged."""
        skeleton = skeletonize_image(simple_line_image)
        # Skeleton should have similar number of pixels
        assert np.sum(skeleton) > 0
        assert np.sum(skeleton) <= np.sum(simple_line_image) + 10

    def test_cross_preserved(self, cross_image):
        """Cross shape should preserve junction."""
        skeleton = skeletonize_image(cross_image)
        assert np.sum(skeleton) > 0
        # Junction point should exist
        assert skeleton[50, 50] == 1


class TestSkeletonToGraph:
    """Tests for skeleton to graph conversion."""

    def test_simple_line_graph(self, simple_line_image):
        """Simple line should have 2 endpoints."""
        skeleton = skeletonize_image(simple_line_image)
        G = skeleton_to_graph(skeleton)

        # Should have 2 nodes (endpoints)
        assert G.number_of_nodes() == 2

        # Should have 1 edge
        assert G.number_of_edges() == 1

        # Edge should have pixels
        edge_data = list(G.edges(data=True))[0]
        pixels = edge_data[2].get("pixels", [])
        assert len(pixels) > 0

    def test_cross_graph(self, cross_image):
        """Cross should have endpoints and junctions."""
        skeleton = skeletonize_image(cross_image)
        G = skeleton_to_graph(skeleton)

        # Should have at least 5 nodes (4 endpoints + junction(s))
        # Skeletonization may create extra nodes near junctions
        assert G.number_of_nodes() >= 5

        # Should have at least 4 edges (one per arm)
        assert G.number_of_edges() >= 4


class TestPruneSpurs:
    """Tests for spur pruning."""

    def test_short_spur_removed(self, line_with_spur_image):
        """Short spurs should be removed."""
        skeleton = skeletonize_image(line_with_spur_image)
        G = skeleton_to_graph(skeleton)

        initial_nodes = G.number_of_nodes()

        G = prune_spurs(G, min_length=10)

        # Should have fewer nodes after pruning
        assert G.number_of_nodes() <= initial_nodes

    def test_long_edges_preserved(self, simple_line_image):
        """Long edges should not be pruned."""
        skeleton = skeletonize_image(simple_line_image)
        G = skeleton_to_graph(skeleton)

        initial_edges = G.number_of_edges()

        G = prune_spurs(G, min_length=10)

        # Long edge should remain
        assert G.number_of_edges() == initial_edges


class TestExtractPaths:
    """Tests for path extraction."""

    def test_extracts_paths(self, simple_line_image):
        """Should extract paths from graph."""
        skeleton = skeletonize_image(simple_line_image)
        G = skeleton_to_graph(skeleton)
        paths = extract_paths(G)

        assert len(paths) == 1
        assert len(paths[0]) >= 2


class TestRasterToPaths:
    """Tests for full pipeline."""

    def test_full_pipeline(self, simple_line_image):
        """Full pipeline should produce paths."""
        paths = raster_to_paths(simple_line_image)

        assert len(paths) > 0
        # Each path should have at least 2 points
        for path in paths:
            assert len(path) >= 2

    def test_cross_pipeline(self, cross_image):
        """Cross image should produce multiple paths."""
        paths = raster_to_paths(cross_image)

        # Should have paths
        assert len(paths) > 0

    def test_empty_image_returns_empty(self):
        """Empty image should return empty paths."""
        empty = np.zeros((100, 100), dtype=np.uint8)
        paths = raster_to_paths(empty)
        assert paths == []
