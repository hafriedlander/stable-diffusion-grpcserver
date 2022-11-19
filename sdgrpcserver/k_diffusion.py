import os, sys, types

module_path = os.path.join(os.path.dirname(__file__), "src/k-diffusion/k_diffusion")

import importlib.util

# We load the k_diffusion files directly rather than relying on Python modules
# This allows us to only install the dependancies of the parts we use

for name in ['utils', 'sampling', 'external']:
    module_name = f"{__name__}.{name}"
    file_path = os.path.join(module_path, f"{name}.py")

    # From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
