import os

debug_path = os.environ.get("SD_DEBUG_PATH", False)
if not debug_path: debug_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "/tests/out/"
)
