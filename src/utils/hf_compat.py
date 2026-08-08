"""Compatibility shim: pyannote.audio 3.x on huggingface_hub>=1.0.

pyannote.audio still calls hf_hub_download(..., use_auth_token=...), a kwarg
huggingface_hub dropped (renamed to `token`) once it hit 1.0. We can't just
pin huggingface_hub back down to a version that still has it, since
transformers (needed by faster-whisper/sentence-transformers) requires
huggingface_hub>=1.5. Instead, patch the hf_hub_download references already
bound inside pyannote.audio's submodules to translate the old kwarg name.

Call patch_pyannote_hf_compat() once, before the first Pipeline/Model
.from_pretrained(..., use_auth_token=...) call.
"""

import functools
import sys

import huggingface_hub

_PATCHED = False


def _accept_use_auth_token(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return func(*args, **kwargs)

    return wrapper


def patch_pyannote_hf_compat() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import pyannote.audio  # noqa: F401  ensure submodules below are loaded

    patched = _accept_use_auth_token(huggingface_hub.hf_hub_download)
    huggingface_hub.hf_hub_download = patched

    for name, module in list(sys.modules.items()):
        if name.startswith("pyannote.audio") and hasattr(module, "hf_hub_download"):
            module.hf_hub_download = patched

    _PATCHED = True
