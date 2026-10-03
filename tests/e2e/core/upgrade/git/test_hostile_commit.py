"""The source swap of ``hermes update`` is one crash-safe commit point (lane LP-COMMIT).

Each cell drives a real install (HEAD's own ``scripts/install.sh`` in a bwrap sandbox) through a
real ``hermes update`` against a local origin, breaks it at one named point with a real SIGKILL or a
real git failure, and then judges the next real launch from the files the updater itself owns:

* ``kill_after_tree_moved``: git finished the fast-forward of a release with NO dependency change
  and the updater is SIGKILLed before the completion child starts. The launcher tail (launchers,
  builds, config migration, ``install-stamp.json``) must still be owed and finished by the next
  launch — a tail armed only by the child is lost for good.
* ``torn_tree_on_git_failure``: git exits non-zero halfway through writing the release (a
  read-only directory: ``unable to unlink old ...``). The checkout must end whole at its pre-update
  commit, never torn with the interrupted-pull marker already dropped.
* ``kill_during_branch_switch``: the checkout is parked on a fully merged branch, so the update
  first switches it to ``main`` (CP0); the updater dies while git is rewriting files. The next
  launch must put the parked tree back.
* ``kill_mid_zip_swap``: the ZIP swap is SIGKILLed between renames. The next launch must finish or
  roll back the swap, leaving no ``*.hermes-update-staging``/``-old`` sibling to wedge a retry.
* ``syntax_error_target``: a release whose startup module does not compile is refused BEFORE
  HEAD moves (no fast-forward in the reflog), not merely rolled back afterwards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.e2e.core.upgrade.test_upgrade_path import _RETRY_PREFIX
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

REAL_GIT = shutil.which("git") or "git"
ARTIFACT_SUFFIXES = (".hermes-update-staging", ".hermes-update-old")


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the hostile commit suite") as srv:
        yield srv


@pytest.fixture()
def world(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("hostile-commit")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    shim = sb.root / "wrap" / "git"
    world = {"sb": sb, "origin": origin, "root": root, "shim": shim, "shim_text": shim.read_text(), "n": 0}
    yield world
    shim.write_text(world["shim_text"])


def _hostile_git(world, body: str) -> None:
    """Prefix the sandbox's git shim with ``body`` (bash; ``$REAL`` is the real git, ``$PPID`` the
    updater process that spawned git)."""
    text = world["shim_text"]
    head, _, rest = text.partition("\n")
    world["shim"].write_text(f'{head}\nREAL="{REAL_GIT}"\n{body}\n{rest}')


def _release(world, files: dict[str, str]) -> str:
    world["n"] += 1
    return I.publish_commit(world["origin"], world["root"], f"release: hostile commit {world['n']}", files)


def _head(sb) -> str:
    return I.git("rev-parse", "HEAD", cwd=sb.checkout)


def _tracked_dirty(sb) -> str:
    return I.git("status", "--porcelain", "--untracked-files=no", cwd=sb.checkout)


def _stamp_commit(sb) -> str:
    try:
        return json.loads((sb.checkout / "install-stamp.json").read_text(encoding="utf-8")).get("commit") or ""
    except (OSError, ValueError):
        return ""


def _launch(sb, marker: str) -> subprocess.CompletedProcess:
    """A later real launch by the user (fresh pid, lazy installs on: the real launch path)."""
    return P.run_env(sb, [*_RETRY_PREFIX, sb.hermes, "-z", marker], P.lazy_env(sb), timeout=P.UPDATE_TIMEOUT)


def _update(sb) -> subprocess.CompletedProcess:
    return P.run_env(sb, [*_RETRY_PREFIX, sb.hermes, "update", "--yes", "--branch", "main", "--no-gateway-restart"],
                     sb.env, timeout=P.UPDATE_TIMEOUT)


def _artifacts(sb) -> list[str]:
    return sorted(p.name for p in sb.checkout.iterdir() if p.name.endswith(ARTIFACT_SUFFIXES))


def test_kill_after_tree_moved_still_owes_the_tail(world):
    sb = world["sb"]
    pre = _head(sb)
    assert _stamp_commit(sb) == pre and not P.pending_marker(sb).exists(), "harness: install not settled"
    target = _release(world, {"e2e_hostile_release.py": "RELEASE = 1\n"})
    # git completes the fast-forward, then the updater dies before it can spawn the completion child.
    _hostile_git(world, 'case " $* " in *" merge --ff-only "*) "$REAL" "$@"; rc=$?; kill -KILL $PPID; exit $rc;; esac')
    killed = _update(sb)
    _hostile_git(world, "")
    assert _head(sb) == target, "harness: the kill did not land after the tree moved\n" + I.describe(killed)
    owed = P.pending_marker(sb).exists()

    launch = _launch(sb, "first launch after a kill past the swap")
    assert owed, ("the tree moved to the release but no source-completion tail was owed: the kill "
                  "before the completion child lost it\n" + P.diagnostics(sb, killed, launch))
    assert not P.pending_marker(sb).exists() and _stamp_commit(sb) == target, (
        "the next launch did not finish the owed tail for the moved tree\n" + P.diagnostics(sb, killed, launch))


def test_torn_tree_on_git_failure_is_restored(world):
    sb = world["sb"]
    # Release 1 adds a file under a directory; the user makes that directory read-only.
    ok = _release(world, {"aaa_e2e_first.py": "V = 1\n", "zzz_e2e_locked/mod.py": "V = 1\n"})
    P.ok(_update(sb), "harness: plain update to the first release failed")
    assert _head(sb) == ok
    _release(world, {"aaa_e2e_first.py": "V = 2\n", "zzz_e2e_locked/mod.py": "V = 2\n"})
    locked = sb.checkout / "zzz_e2e_locked"
    locked.chmod(0o555)
    try:
        failed = _update(sb)
    finally:
        locked.chmod(0o755)
    assert failed.returncode != 0, "harness: git did not fail on the read-only directory\n" + I.describe(failed)
    launch = _launch(sb, "first launch after a torn pull")
    assert _head(sb) == ok and not _tracked_dirty(sb), (
        "git failed mid-pull and the checkout was left torn (marker dropped, nothing restores it):\n"
        + _tracked_dirty(sb) + "\n" + P.diagnostics(sb, failed, launch))


def test_kill_during_branch_switch_is_restored(world):
    sb = world["sb"]
    base = _head(sb)
    _release(world, {"e2e_cp0_a.py": "A = 1\n", "e2e_cp0_b.py": "B = 1\n"})
    P.ok(_update(sb), "harness: plain update failed")
    # Park the checkout on a fully merged branch one release behind main.
    I.git("checkout", "-q", "-b", "e2e-parked", base, cwd=sb.checkout)
    parked = _head(sb)
    _release(world, {"e2e_cp0_c.py": "C = 1\n"})
    # CP0: `git checkout main` dies after rewriting one file of main's tree.
    _hostile_git(world, 'if [ "${@: -2:1}" = checkout ] && [ "${@: -1}" = main ]; then '
                        '"$REAL" show main:e2e_cp0_a.py > e2e_cp0_a.py; kill -KILL $PPID; exit 137; fi')
    killed = _update(sb)
    _hostile_git(world, "")
    assert (sb.checkout / "e2e_cp0_a.py").is_file() and _head(sb) == parked, (
        "harness: the kill did not land mid-switch\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill during the branch switch")
    assert _head(sb) == parked and not (sb.checkout / "e2e_cp0_a.py").exists(), (
        "a kill during the CP0 branch switch left main's files in the parked checkout\n"
        + P.diagnostics(sb, killed, launch))


def test_syntax_error_target_is_refused_before_head_moves(world):
    sb = world["sb"]
    pre = _head(sb)
    constants = I.git("show", "main:hermes_constants.py", cwd=world["origin"])
    _release(world, {"hermes_constants.py": constants + "\ndef broken(:\n"})
    reflog_before = I.git("reflog", "-n", "20", "--format=%H %gs", cwd=sb.checkout)
    refused = _update(sb)
    reflog_after = I.git("reflog", "-n", "20", "--format=%H %gs", cwd=sb.checkout)
    assert refused.returncode != 0 and _head(sb) == pre, I.describe(refused)
    assert reflog_after == reflog_before, (
        "HEAD moved onto the uncompilable release before the syntax guard refused it:\n"
        + reflog_after.replace(reflog_before, "").strip() + "\n" + I.describe(refused))


_ZIP_DRIVER = r"""
import os, signal, sys
sys.path.insert(0, sys.argv[3])  # the install's checkout (the venv's own path points at its workspace)
from hermes_cli import update_cmd_zip as z
import hermes_cli.main as m
assert str(m.PROJECT_ROOT) == sys.argv[3], m.PROJECT_ROOT
renames, kill_after = [0], int(sys.argv[2])
real = os.rename
def rename(src, dst):
    real(src, dst)
    if str(src).endswith(".hermes-update-staging"):  # one entry swapped in
        renames[0] += 1
        if renames[0] >= kill_after:
            os.kill(os.getpid(), signal.SIGKILL)
os.rename = rename
z._download_and_swap_zip("main", sys.argv[1])
"""


def test_kill_mid_zip_swap_is_settled_by_the_next_launch(world):
    sb = world["sb"]
    pre = _head(sb)
    target = _release(world, {"e2e_zip_release.py": "Z = 1\n", "hermes_cli/e2e_zip_marker.py": "M = 1\n"})
    archive = world["root"] / "release.zip"
    I.git("archive", "--format=zip", "--prefix=hermes-agent-main/", "-o", str(archive), target, cwd=world["origin"])
    driver = world["root"] / "zip_driver.py"
    driver.write_text(_ZIP_DRIVER, encoding="utf-8")
    killed = P.run_env(sb, [sb.python, str(driver), archive.as_uri(), "12", str(sb.checkout)], sb.env, timeout=P.UPDATE_TIMEOUT)
    at_kill = {"artifacts": _artifacts(sb), "dirty": _tracked_dirty(sb),
               "new_entry": (sb.checkout / "e2e_zip_release.py").exists()}
    assert at_kill["artifacts"] and at_kill["dirty"], (
        f"harness: the ZIP swap was not killed mid-rename ({at_kill['dirty']!r})\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill mid ZIP swap")
    leftovers = _artifacts(sb)
    assert not leftovers, (f"{len(leftovers)} staging/backup siblings leaked and wedge the next update: "
                           f"{leftovers[:5]} (at kill: {len(at_kill['artifacts'])} siblings, "
                           f"tracked changes {at_kill['dirty']!r})\n" + I.describe(launch))
    assert _head(sb) == pre and not _tracked_dirty(sb), (
        "the interrupted ZIP swap left a mixed tree:\n" + _tracked_dirty(sb) + "\n" + I.describe(launch))
