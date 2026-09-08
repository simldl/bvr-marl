#!/usr/bin/env python3
"""Total this project's Linux download size, from any platform, without Docker.

Why this exists
---------------
The Linux CI job died with

    ERROR: Could not install packages due to an OSError:
    [Errno 28] No space left on device

and the cause is invisible from the Windows checkout this project is mostly
developed on: ``torch`` on PyPI declares its CUDA dependencies behind
``platform_system == "Linux"`` markers, so the Windows wheel is CPU-only and
small while the Linux wheel drags ``cuda-toolkit``, ``nvidia-cudnn-cu13``,
``nvidia-nccl-cu13`` and friends in behind it. The failure therefore reproduces
on exactly one platform, and not the developer's.

The obvious trick -- ``pip install --dry-run --platform manylinux_2_28_x86_64``
-- does NOT work here, and fails in the worst way, by reporting an all-clear.
``--platform`` selects which wheel *tags* are acceptable; it does not change the
environment that dependency *markers* are evaluated against. Those still see the
host, so on Windows every ``platform_system == "Linux"`` requirement evaluates
false and the CUDA tree silently vanishes from the report.

So this script evaluates the markers itself, against a pinned Linux environment,
and totals the real wheel sizes from the index metadata.

What it measures
----------------
DOWNLOAD size. The installed tree is larger (wheels are compressed), and pip
keeps a second copy of every wheel in its HTTP cache unless ``--no-cache-dir``
is passed, so peak disk is roughly twice the number printed here plus the
unpacked tree. A GitHub-hosted ubuntu runner was MEASURED at 9.9 GB free on
arrival (not the ~14 GB usually quoted); the workflow's reclaim step lifts
that to 32 GB before any dependency lands.

This walks the dependency graph taking the newest version that satisfies each
requirement. It does not backtrack, so it is an upper-bound sketch of a
resolution rather than a resolver. For the question it exists to answer -- "does
the CUDA tree land on the runner, and roughly how many GB is it" -- that is
enough.

Usage
-----
    python scripts/ci/check_linux_wheel_size.py --compare     # both ways
    python scripts/ci/check_linux_wheel_size.py               # PyPI as-is
    python scripts/ci/check_linux_wheel_size.py --spec ray
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

#: PyTorch's CPU wheel index. ``torch==X.Y.Z+cpu`` there has no nvidia-*
#: dependencies at all -- that is the whole point of the build.
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

#: Free space on a GitHub-hosted ubuntu runner before anything this project
#: installs lands on it. MEASURED from run 33845446530, not assumed: `df -h /`
#: reported 72G total, 62G used, **9.9G available** — appreciably tighter than
#: the ~14 GB usually quoted for these images. The workflow's reclaim step takes
#: it to 32G before the dependency tree arrives.
RUNNER_FREE_GB = 9.9
RUNNER_FREE_AFTER_RECLAIM_GB = 32.0

#: The CI job's interpreter and target platform.
PY_VERSION = "3.12"
PY_FULL = "3.12.12"
WHEEL_TAG = re.compile(r"-(cp312|py3|cp3\d+)-.*(manylinux|linux_x86_64)", re.I)

_TIMEOUT_S = 30

#: The CDN in front of the PyTorch index answers 403 to a request with no
#: User-Agent, which reads as "wheel not found" rather than as a refusal.
_HEADERS = {"User-Agent": "bvr-marl-ci-size-check"}
_session_cache: dict[str, object] = {}


def linux_environment() -> dict[str, str]:
    """Marker environment for CPython 3.12 on manylinux x86_64.

    Overrides exactly the keys that differ from the host. Everything else is
    inherited so the dict stays a valid marker environment as packaging evolves.
    """
    env = dict(default_environment())
    env.update(
        {
            "platform_system": "Linux",
            "sys_platform": "linux",
            "platform_machine": "x86_64",
            "os_name": "posix",
            "python_version": PY_VERSION,
            "python_full_version": PY_FULL,
            "implementation_name": "cpython",
        }
    )
    return env


class _LinkParser(HTMLParser):
    """Collect hrefs from a PEP 503 simple-index page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)


def _get(url: str) -> bytes | None:
    if url in _session_cache:
        return _session_cache[url]  # type: ignore[return-value]
    try:
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            body = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        body = None
    _session_cache[url] = body
    return body


def pypi_release(name: str, specifier: SpecifierSet) -> tuple[str, float, list[str]] | None:
    """Newest PyPI version matching ``specifier``: (version, MB, requires_dist)."""
    body = _get(f"https://pypi.org/pypi/{name}/json")
    if body is None:
        return None
    data = json.loads(body)

    candidates = []
    for raw, files in data.get("releases", {}).items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if version.is_prerelease or not specifier.contains(version, prereleases=False):
            continue
        if any(f.get("packagetype") == "bdist_wheel" and not f.get("yanked") for f in files):
            candidates.append((version, raw, files))
    if not candidates:
        return None
    _, raw, files = max(candidates, key=lambda c: c[0])

    wheels = [f for f in files if f.get("packagetype") == "bdist_wheel" and not f.get("yanked")]
    linux = [f for f in wheels if WHEEL_TAG.search(f["filename"])]
    pick = max(linux or wheels, key=lambda f: f.get("size") or 0)

    meta = _get(f"https://pypi.org/pypi/{name}/{raw}/json")
    requires = (json.loads(meta)["info"].get("requires_dist") or []) if meta else []
    return raw, (pick.get("size") or 0) / 1e6, requires


def cpu_index_torch() -> tuple[str, float, list[str]] | None:
    """Newest ``+cpu`` torch wheel on the PyTorch index: (version, MB, requires)."""
    body = _get(f"{CPU_INDEX}/torch/")
    if body is None:
        return None
    parser = _LinkParser()
    parser.feed(body.decode("utf-8", "replace"))

    best: tuple[Version, str] | None = None
    for href in parser.links:
        # The index percent-encodes the local-version '+' as %2B.
        filename = urllib.parse.unquote(href.split("/")[-1].split("#")[0])
        if not filename.startswith("torch-") or "+cpu" not in filename:
            continue
        # Match manylinux_2_28_x86_64 as well as the bare linux_x86_64 tag.
        # "linux_x86_64" is NOT a substring of "manylinux_2_28_x86_64".
        if "cp312" not in filename or not filename.endswith("x86_64.whl"):
            continue
        if "linux" not in filename:
            continue
        try:
            version = Version(filename.split("-")[1])
        except (InvalidVersion, IndexError):
            continue
        if best is None or version > best[0]:
            best = (version, href)
    if best is None:
        return None

    version, href = best
    url = href if href.startswith("http") else f"{CPU_INDEX}/torch/{href.lstrip('./')}"
    url = url.split("#")[0]

    # A one-byte ranged GET, not HEAD: the CDN in front of this index answers
    # HEAD without a Content-Length, which silently reported every wheel as 0 MB.
    # Content-Range carries the true total.
    size_mb = 0.0
    try:
        request = urllib.request.Request(url, headers={**_HEADERS, "Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                size_mb = int(content_range.rsplit("/", 1)[1]) / 1e6
            else:
                size_mb = int(response.headers.get("Content-Length", 0)) / 1e6
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        pass

    # PEP 658: the index serves the wheel's METADATA next to the wheel itself.
    requires: list[str] = []
    meta = _get(url + ".metadata")
    if meta is not None:
        requires = [
            line.split(":", 1)[1].strip()
            for line in meta.decode("utf-8", "replace").splitlines()
            if line.lower().startswith("requires-dist:")
        ]
    return str(version), size_mb, requires


def walk(root: str, *, cpu_index: bool) -> list[tuple[str, str, float]]:
    """Breadth-first dependency walk with Linux markers. Returns (name, ver, MB)."""
    env = linux_environment()
    seen: set[str] = set()
    out: list[tuple[str, str, float]] = []
    queue = [Requirement(root)]

    while queue:
        requirement = queue.pop(0)
        key = requirement.name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)

        if cpu_index and key == "torch":
            found = cpu_index_torch()
        else:
            found = pypi_release(requirement.name, requirement.specifier)
        if found is None:
            print(f"  (could not resolve {requirement.name}; skipped)", file=sys.stderr)
            continue

        version, size_mb, requires = found
        out.append((key, version, size_mb))

        for raw in requires:
            try:
                dependency = Requirement(raw)
            except Exception:  # noqa: BLE001 - malformed metadata must not abort the walk
                continue
            # Extras are not requested, so skip anything gated on one.
            if dependency.marker is not None and not dependency.marker.evaluate(env):
                continue
            queue.append(dependency)

    return sorted(out, key=lambda r: -r[2])


def report(spec: str, *, cpu_index: bool) -> float:
    label = "CPU index (the fix)" if cpu_index else "PyPI default (what CI did)"
    print(f"\n=== {spec}  --  {label} ===")
    table = walk(spec, cpu_index=cpu_index)
    total = sum(mb for _, _, mb in table)
    cuda = [r for r in table if r[0].startswith(("nvidia-", "cuda-")) or r[0] == "triton"]

    for name, version, mb in table[:14]:
        flag = "   <-- CUDA" if (name, version, mb) in cuda else ""
        print(f"  {mb:8.1f} MB  {name} {version}{flag}")
    if len(table) > 14:
        print(f"  {'':8}     ... and {len(table) - 14} smaller")

    print(f"  {'-' * 46}")
    print(f"  {total:8.1f} MB  TOTAL download ({len(table)} wheels)")
    if cuda:
        cuda_mb = sum(mb for _, _, mb in cuda)
        print(f"  {cuda_mb:8.1f} MB  of that is CUDA ({len(cuda)} wheels)")
    print(
        f"  {total * 2 / 1000:8.1f} GB  rough peak disk (download + pip cache), "
        f"against {RUNNER_FREE_GB:.1f} GB free on arrival / {RUNNER_FREE_AFTER_RECLAIM_GB:.0f} GB after reclaim"
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="torch>=2.7.0", help="requirement to resolve")
    parser.add_argument("--cpu-index", action="store_true", help="resolve torch from the CPU index")
    parser.add_argument("--compare", action="store_true", help="resolve both ways and diff")
    args = parser.parse_args()

    if args.compare:
        before = report(args.spec, cpu_index=False)
        after = report(args.spec, cpu_index=True)
        if before > 0:
            print(
                f"\n=== saving: {before - after:.1f} MB download, "
                f"{2 * (before - after) / 1000:.1f} GB of peak disk "
                f"({100 * (before - after) / before:.0f}% smaller) ==="
            )
    else:
        report(args.spec, cpu_index=args.cpu_index)


if __name__ == "__main__":
    main()
