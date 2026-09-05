"""Runtime compatibility surface for the packaged TMCRA V2 vectorizer.

The implementation remains in the frozen ``tmp_tmcra_v2_lme_pipeline``
module until the local-runtime extraction is completed.  Keeping this small
surface makes the service package importable without depending on a server-only
checkout layout.
"""

from tmp_tmcra_v2_lme_pipeline import *  # noqa: F401,F403

