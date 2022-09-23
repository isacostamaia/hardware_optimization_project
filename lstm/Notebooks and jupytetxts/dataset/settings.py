"""
Settings file
Imports the correct settings file:
settings_local.py if existing - is gitignored and can be used to
override settings_default.py in dev environment
settings_default.py if no settings_local.py file
"""
# pylint: disable=import-error, unused-import, wildcard-import, unused-wildcard-import
import os

# Check if settings_local.py exists
settings_local_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'settings_local.py'
)
if os.path.isfile(settings_local_path):
    from .settings_local import *
else:
    from .settings_default import *
