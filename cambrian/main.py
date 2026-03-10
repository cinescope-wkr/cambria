"""This is the main entrypoint for the ``cambrian`` package. It's used to run the
training and evaluation loops."""

import argparse
import multiprocessing
import os
import platform
import stat
import sys
import warnings
from pathlib import Path

def _prepare_ipc_env() -> None:
    """Prepare IPC-related environment for stable multiprocessing on WSL/Linux."""
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")
    os.environ.setdefault("TMPDIR", "/tmp")
    # In sandboxed/headless environments, writing to ~/.cache may be blocked.
    # Point cache-related env vars to a writable location to avoid noisy warnings.
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mesa_shader_cache")

    try:
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        # Best-effort: caching is optional.
        pass

    shm_path = Path("/dev/shm")
    try:
        if shm_path.exists() and os.geteuid() == 0:
            mode = stat.S_IMODE(shm_path.stat().st_mode)
            if mode != 0o1777:
                shm_path.chmod(0o1777)
    except OSError:
        # Keep running with /tmp fallback even if /dev/shm is not writable.
        os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

    is_wsl2 = "microsoft" in platform.release().lower()
    if is_wsl2:
        multiprocessing.set_start_method("spawn", force=True)


_prepare_ipc_env()

from cambrian import MjCambrianConfig, MjCambrianTrainer, run_hydra


def _maybe_inject_evo_config() -> None:
    """Hydra QoL: if the user overrides `evo.*` without selecting an evo config,
    automatically load a minimal evo config so OmegaConf doesn't crash on `None`.
    """

    argv = sys.argv[1:]
    has_evo_group = any(a.startswith(("evo=", "+evo=")) for a in argv)
    has_evo_field_override = any(
        a.startswith(("evo.", "evo/", "+evo.", "+evo/")) for a in argv
    )

    if has_evo_group or not has_evo_field_override:
        return

    # Insert before the first evo.* override so that explicit overrides still win.
    insert_at = next(
        (
            i
            for i, a in enumerate(sys.argv)
            if a.startswith(("evo.", "evo/", "+evo.", "+evo/"))
        ),
        len(sys.argv),
    )
    sys.argv.insert(insert_at, "evo=basic")
    warnings.warn("Detected `evo.*` overrides; auto-applying `evo=basic`.")


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")

    def _main(
        config: MjCambrianConfig,
        *,
        train: bool,
        eval: bool,
    ) -> float:
        """This method will return a float if training. The float represents the
        "fitness" of the agent that was trained. This can be used by hydra to
        determine the best hyperparameters during sweeps."""
        runner = MjCambrianTrainer(config)

        if train:
            return runner.train()
        elif eval:
            return runner.eval()

    _maybe_inject_evo_config()
    run_hydra(_main, parser=parser, config_name="base")


if __name__ == "__main__":
    main()
