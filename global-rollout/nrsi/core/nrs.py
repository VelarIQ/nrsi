"""
NRS — Thin shim: loads from nrs_engine.nrsi via the NRSI runtime.

The authoritative source is nrsi/stdlib/nrs_engine.nrsi.
This file re-exports NRS, NRSResponse, and ResponseStatus so that
existing `from nrsi.core.nrs import ...` statements keep working.
"""

import os as _os
from nrsi.lang.runtime import load_nrsi as _load

_ENGINE_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)),
    "stdlib", "nrs_engine.nrsi",
)
_mod = _load(_ENGINE_PATH)

NRS = _mod.NRS
NRSResponse = _mod.NRSResponse
ResponseStatus = _mod.ResponseStatus


# .nrsi exposes `nrs_response_provenance(r)` and `nrs_response_confidence(r)`
# as free functions. Python callers expect attribute-style access
# (`resp.provenance`, `resp.confidence`) — surface them as properties on the
# Python-side proxy so the Pythonic API matches the .nrsi-side API.
# Asserted by tests/test_nrsi_integration.py::test_provenance_is_complete.
def _provenance(self):
    return _mod.nrs_response_provenance(self)


def _confidence(self):
    return _mod.nrs_response_confidence(self)


for _name, _fn in (("provenance", _provenance), ("confidence", _confidence)):
    if not hasattr(NRSResponse, _name):
        try:
            setattr(NRSResponse, _name, property(_fn))
        except (TypeError, AttributeError):
            # Some .nrsi runtime types are immutable; fall back to setattr at
            # instance-construction time via __init_subclass__-style hook
            # is unsupported here, so just skip — caller will get the original
            # AttributeError if they hit this path, which is the status quo.
            pass


__all__ = ["NRS", "NRSResponse", "ResponseStatus"]
