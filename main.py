"""
Server Entry Point
Run with: python main.py
"""
import os
import sys
import warnings

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ─── Pydantic v1 + Python 3.14 compatibility patch ────────────────────────
# ChromaDB <=1.5.x uses pydantic.v1.BaseSettings which breaks on Python 3.14
# because get_type_hints() behaviour changed. This patch fixes type inference.
if sys.version_info >= (3, 14):
    warnings.filterwarnings("ignore", message=".*Pydantic V1.*Python 3.14.*")
    try:
        import pydantic.v1.fields as _pv1_fields

        _orig_set_default_and_type = _pv1_fields.ModelField._set_default_and_type

        def _patched_set_default_and_type(self):
            try:
                _orig_set_default_and_type(self)
            except _pv1_fields.errors_.ConfigError:
                # Fall back: use the outer_type_ or annotation directly
                if self.type_ is _pv1_fields.Undefined:
                    import typing
                    hint = getattr(self, 'outer_type_', None)
                    if hint is None or hint is _pv1_fields.Undefined:
                        # last resort: set to Any
                        self.type_ = typing.Any
                        self.outer_type_ = typing.Any
                    else:
                        self.type_ = hint
                        self.outer_type_ = hint

        _pv1_fields.ModelField._set_default_and_type = _patched_set_default_and_type
        print("[Compat] Applied pydantic v1 + Python 3.14 patch")
    except Exception as e:
        print(f"[Compat] Patch skipped: {e}")

from app import app  # noqa: E402  (import after load_dotenv)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    print(f"[Server] Starting on http://localhost:{port}")
    print(f"[Server] Client UI at http://localhost:{port}/")
    print(f"[Server] API docs at http://localhost:{port}/docs")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
