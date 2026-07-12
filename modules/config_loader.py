"""
SafeGuard AI — Config Loader
=============================
Loads config.yaml with per-camera overrides.
Provides typed access to all detection parameters.
"""

import os
import yaml
import copy
import logging

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


class SafeGuardConfig:
    """
    Loads config.yaml and resolves per-camera overrides.

    Usage:
        cfg = SafeGuardConfig()
        cam_cfg = cfg.for_camera("cam_01")
        threshold = cam_cfg["zone_intrusion"]["dwell_threshold_sec"]
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH):
        self._path = config_path
        self._raw = self._load()

    def _load(self) -> dict:
        try:
            with open(self._path, 'r') as f:
                data = yaml.safe_load(f)
            logger.info(f"Config loaded from {self._path}")
            return data or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self._path}, using empty config")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in config: {e}")
            return {}

    def _global_sections(self) -> dict:
        """Return all top-level keys except 'cameras' and 'global'."""
        return {k: v for k, v in self._raw.items() if k not in ('cameras', 'global')}

    def for_camera(self, camera_id: str) -> dict:
        """
        Return merged config for a specific camera.
        Camera-level overrides are applied on top of global section defaults.
        """
        base = self._global_sections()
        cameras = self._raw.get('cameras', {}) or {}
        cam_overrides = cameras.get(camera_id, {}) or {}
        return _deep_merge(base, cam_overrides)

    @property
    def global_settings(self) -> dict:
        return self._raw.get('global', {}) or {}

    @property
    def camera_ids(self) -> list:
        return list((self._raw.get('cameras', {}) or {}).keys())

    def reload(self):
        """Reload config from disk (useful for hot-reload in long-running process)."""
        self._raw = self._load()


# Module-level singleton — import and use directly
_cfg_instance = None


def get_config(camera_id: str = None, config_path: str = _DEFAULT_CONFIG_PATH) -> dict:
    """
    Convenience function.
    Returns merged config dict for the given camera_id, or global config if None.
    """
    global _cfg_instance
    if _cfg_instance is None or _cfg_instance._path != config_path:
        _cfg_instance = SafeGuardConfig(config_path)
    if camera_id:
        return _cfg_instance.for_camera(camera_id)
    return _cfg_instance._global_sections()
