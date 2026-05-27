"""LumenX auto-reply agent package.

On Windows, route TLS verification through the OS cert store so corporate
MITM CAs (which are installed in the Windows store but not in certifi's
bundle) work transparently for httpx, the anthropic SDK, requests, etc.
"""
from __future__ import annotations

import platform

if platform.system() == "Windows":
    import truststore
    truststore.inject_into_ssl()
