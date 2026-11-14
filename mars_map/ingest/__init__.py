"""Ingestion (part A): catalog loaders, fetcher with cache, normalizer, global layers, series."""

from .fetch import FetchError, fetch, fetch_product, fetch_window
from .loaders import load_catalog
from .normalize import normalize

__all__ = ["FetchError", "fetch", "fetch_product", "fetch_window", "load_catalog", "normalize"]
