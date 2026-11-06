import os
import sys

# WebODM installs this plugin's OpenCV requirement into a
# per-plugin site-packages (see PluginBase.check_requirements) but only adds it
# to sys.path inside the python_imports() context manager. We add it eagerly
# for OpenCV; release ZIPs also carry the local core under vendor/.
#
# The plugin module imports cleanly WITHOUT signa_core (all signa_core imports
# are lazy, inside the views/worker/forms), so WebODM can instantiate the plugin
# and run check_requirements() to install the deps even on a fresh upload — no
# chicken-and-egg. Paths are inserted unconditionally (a not-yet-existing dir is
# harmless and is picked up once pip populates it).
def _add_site_packages():
    candidates = ["/webodm/app/media/plugins/signa/site-packages"]
    try:
        from django.conf import settings as _dj
        candidates.insert(0, os.path.join(_dj.MEDIA_ROOT, "plugins", "signa", "site-packages"))
    except Exception:
        pass
    for sp in candidates:
        if sp and sp not in sys.path:
            sys.path.insert(0, sp)


_add_site_packages()
_bundled_core = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_bundled_core):
    sys.path.insert(0, _bundled_core)

from .plugin import *  # noqa: E402,F401,F403
