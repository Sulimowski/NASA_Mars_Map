"""Ingestion (part A): catalog loaders, fetcher with cache, normalizer, global layers, series."""

# Verify TLS certificates through the operating system's trust store. The PDS ODE
# server does not send its intermediate certificate; Python's own verifier then fails
# with CERTIFICATE_VERIFY_FAILED on machines that never cached it, while the OS store
# fetches it. Verification stays on.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:                                  # optional: only ODE needs it
    pass

from .fetch import FetchError, fetch, fetch_product, fetch_window
from .loaders import load_catalog
from .normalize import normalize

__all__ = ["FetchError", "fetch", "fetch_product", "fetch_window", "load_catalog", "normalize"]
