try:
    from .utils import check_duplicate_nodes, log
    duplicate_dirs = check_duplicate_nodes()
    if duplicate_dirs:
        warning_msg = f"WARNING:  Found {len(duplicate_dirs)} other WanVideoWrapper directories:\n"
        for dir_path in duplicate_dirs:
            warning_msg += f"  - {dir_path}\n"
        log.warning(warning_msg + "Please remove duplicates to avoid possible conflicts.")
except:
    pass

import sys

try:
    from . import wanvideo as _wan_pkg
    sys.modules.setdefault("wanvideo", _wan_pkg)
    from .wanvideo import modules as _wan_modules
    sys.modules.setdefault("wanvideo.modules", _wan_modules)
    from .wanvideo.modules import shot_utils as _wan_shot_utils
    sys.modules.setdefault("wanvideo.modules.shot_utils", _wan_shot_utils)
except Exception:
    pass
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes_model_loading import NODE_CLASS_MAPPINGS as MODEL_LOADING_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MODEL_LOADING_NODE_DISPLAY_NAME_MAPPINGS
from .nodes_sampler import NODE_CLASS_MAPPINGS as SAMPLER_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_NODE_DISPLAY_NAME_MAPPINGS




NODE_CLASS_MAPPINGS.update(MODEL_LOADING_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SAMPLER_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(MODEL_LOADING_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SAMPLER_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
