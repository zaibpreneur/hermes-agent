"""A killed tree move that tears the launch repair's own code is still repaired at the next launch.

The repair (``hermes_bootstrap`` -> ``_early_recovery`` under the checkout lock) lives in the tree git
rewrites. ``arm_tree_move`` publishes that closure beside the marker before git writes; the minted
launcher runs it when the checkout's copy cannot even import. Real git, really killed mid-write (a
smudge-filter barrier), the production marker writer in a child that exits, the production launcher;
the application entry is an inert receipt.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import _launchers

SOURCE = Path(__file__).resolve().parents[2]
# What a minted launcher imports before the application entry.
BOOT = ("hermes_bootstrap.py", "hermes_constants.py", "hermes_cli/__init__.py", "hermes_cli/_launchers.py",
        "hermes_cli/_early_recovery.py", "hermes_cli/_parser.py", "hermes_cli/runtime_state.py",
        "hermes_cli/venv_sync.py", "hermes_cli/steward.py", "hermes_cli/stderr_timestamp.py",
        "hermes_cli/update_lock.py", "hermes_cli/update_custody.py", "pm/environments.py",
        "pm/filesystem.py", "pm/paths.py")
_HOLDER = ("import sys, time\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[1])\n"
           "from hermes_cli.update_lock import UpdateLock\n"
           "lock = UpdateLock(path=Path(sys.argv[2]) / 'owner', install_root=Path(sys.argv[2]))\n"
           "assert lock.acquire()\nprint('HELD', flush=True)\ntime.sleep(120)\n")


def _killed_mid_write(tmp_path: Path, rel: str):
    """A checkout whose update git was SIGKILLed after unlinking ``rel``, before writing it."""
    root, home = tmp_path / "checkout", tmp_path / "home"
    home.mkdir()
    for f in BOOT:
        if (SOURCE / f).is_file():
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / f, root / f)
    (root / "hermes_cli/main.py").write_text("def main():\n    print('APP_REACHED')\n    return 0\n", encoding="utf-8")
    env = {"HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "PATH": os.environ.get("PATH", ""),
           "LANG": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    git_exe = shutil.which("git") or "git"

    def git(*args):
        return subprocess.check_output([git_exe, "-C", str(root), *args], env=env, text=True, encoding="utf-8").strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.invalid")
    git("add", "-A")
    git("commit", "-qm", "before")
    pre = git("rev-parse", "HEAD")
    original = (root / rel).read_bytes()
    (root / rel).write_bytes(original + b"\nNEXT_VERSION = True\n")
    git("commit", "-qam", "after")
    target = git("rev-parse", "HEAD")
    git("reset", "-q", "--hard", pre)
    armed = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; "
         "from hermes_cli.update_cmd_commit import arm_tree_move; "
         "arm_tree_move([sys.argv[2]], Path(sys.argv[3]), pre=sys.argv[4], target=sys.argv[5], stash=None)",
         str(SOURCE), git_exe, str(root), pre, target], env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60)
    assert armed.returncode == 0, armed.stderr  # the updater that armed the move is gone

    gate = tmp_path / "filter-ready"
    (tmp_path / "hold.py").write_text(f"from pathlib import Path\nimport time\nPath({str(gate)!r}).touch()\n"
                                      "time.sleep(120)\n", encoding="utf-8")
    (root / ".git/info/attributes").write_text(f"{rel} filter=hold\n", encoding="utf-8")
    git("config", "filter.hold.smudge", shlex.join([sys.executable, str(tmp_path / "hold.py")]))
    git("config", "filter.hold.clean", "cat")
    git("config", "filter.hold.required", "true")
    merge = subprocess.Popen([git_exe, "-C", str(root), "merge", "--ff-only", target], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not gate.exists():
            assert merge.poll() is None and time.monotonic() < deadline, merge.communicate()
            time.sleep(0.01)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(merge.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only test (platforms marker)
        merge.wait()
    assert not (root / rel).exists()
    git("config", "--unset", "filter.hold.smudge")
    (root / ".git/info/attributes").unlink()

    (tmp_path / "bin").mkdir()
    launcher = _launchers._mint_shell_launcher("hermes", tmp_path / "bin", Path(sys.executable),
                                               _launchers._launcher_script("hermes", root, None))
    assert launcher is not None
    return root, env, original, launcher


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("rel", "torn"), [
    ("hermes_bootstrap.py", False),
    ("hermes_cli/_early_recovery.py", True),
    ("hermes_cli/__init__.py", True),
    ("hermes_cli/update_lock.py", True),
])
def test_a_launch_repairs_a_move_killed_while_writing_the_repairs_own_code(tmp_path, rel, torn):
    root, env, original, launcher = _killed_mid_write(tmp_path, rel)
    if torn:  # git's write cut short: a prefix of the new blob
        (root / rel).write_bytes(original[:12])
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60)
    assert launch.returncode == 0 and "APP_REACHED" in launch.stdout, launch.stderr
    assert (root / rel).read_bytes() == original
    assert not (root / ".git/hermes-update-pull").exists()


@pytest.mark.platforms("posix")
def test_the_published_repair_leaves_the_tree_to_a_live_writer_holding_the_checkout(tmp_path):
    root, env, original, launcher = _killed_mid_write(tmp_path, "hermes_bootstrap.py")
    holder = subprocess.Popen([sys.executable, "-I", "-c", _HOLDER, str(SOURCE), str(root)], env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "HELD"
        launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60)
        assert launch.returncode != 0 and "Not repairing the checkout now" in launch.stderr, launch.stderr
        assert not (root / "hermes_bootstrap.py").exists(), "repaired under a live writer"
        assert (root / ".git/hermes-update-pull").exists()
    finally:
        holder.kill()
        holder.wait()
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60)
    assert launch.returncode == 0 and "APP_REACHED" in launch.stdout, launch.stderr
    assert (root / "hermes_bootstrap.py").read_bytes() == original
