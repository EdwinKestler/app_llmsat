"""Pytest configuration — prepend stubs to sys.path so the local samgeo stubs
are imported instead of the real package during tests."""

import sys
import os

# Insert the stubs directory at the front of sys.path so that
# ``import samgeo`` resolves to tests/stubs/samgeo during test runs.
stubs_dir = os.path.join(os.path.dirname(__file__), "stubs")
sys.path.insert(0, stubs_dir)
