#!/usr/bin/env python3
"""Can tools/startup/tester_run.ps1 use a build made from another commit? Yes when the runtime sources are the same.

  runtime_sources.py <repo> <build revision>    print why; exit 0 = same runtime sources, 1 = not the same, 2 = error

The build stamps the commit it was made from (generated/x2w_build_rev.cpp: `git rev-parse --short=8 HEAD`, with
"-dirty" for local changes). A commit that changes only tools, tests or docs builds the same eight DLLs, so a tester who
pulls it does not have to rebuild. The DLLs of `startup_runtime` are built from RUNTIME_PATHS only (CMakeLists.txt adds
src/ files, cmake/x2w_build_rev.cmake and the presets; no source includes tools/ or tests/), and the build is used when
the git tree of those paths is the same at the build's commit and at HEAD, and the checkout has no local change in them.
"""
import hashlib
import re
import subprocess
import sys

RUNTIME_PATHS = ("src", "cmake", "CMakeLists.txt", "CMakePresets.json")
SHOWN_FILES = 3  # changed runtime files named in the refusal


class GitError(Exception):
    pass


def git(repo, *args):
    done = subprocess.run(["git", "-C", repo, "--no-optional-locks", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise GitError(done.stderr.strip() or "git %s failed" % args[0])
    return done.stdout


def runtime_tree(repo, commit):
    """One id for the runtime sources of commit: a hash of `git ls-tree -r` over RUNTIME_PATHS (mode, object id and
    path of every file), so two commits have the same id exactly when those files are the same."""
    listing = git(repo, "ls-tree", "-r", "--full-tree", commit, "--", *RUNTIME_PATHS)
    return hashlib.sha1(listing.encode("utf-8")).hexdigest()


def compare(repo, built):
    """(same, reason): may the build stamped `built` run with the checkout at HEAD?"""
    if not built or built == "unknown":
        return False, "the build has no source revision"
    if built.endswith("-dirty"):
        return False, "the build %s was made from a checkout with local changes" % built
    if not re.fullmatch(r"[0-9a-f]{7,40}", built):
        return False, "the build revision %r is not a commit" % built
    head = git(repo, "rev-parse", "--short=8", "HEAD").strip()
    try:
        git(repo, "cat-file", "-e", built + "^{commit}")
    except GitError:
        return False, "the build's commit %s is not in this checkout" % built
    local = git(repo, "status", "--porcelain", "--untracked-files=no", "--", *RUNTIME_PATHS).splitlines()
    if local:
        return False, "this checkout has local changes in the runtime sources (%s)" % ", ".join(
            line[3:] for line in local[:SHOWN_FILES])
    if runtime_tree(repo, built) != runtime_tree(repo, "HEAD"):
        changed = git(repo, "diff", "--name-only", built, "HEAD", "--", *RUNTIME_PATHS).split()
        more = " and %d more" % (len(changed) - SHOWN_FILES) if len(changed) > SHOWN_FILES else ""
        return False, "the runtime sources differ between the build %s and %s (%s%s)" % (
            built, head, ", ".join(changed[:SHOWN_FILES]), more)
    return True, "the build %s has the same runtime sources as %s (%s): only tools, tests or docs differ" % (
        built, head, ", ".join(RUNTIME_PATHS))


def main(argv):
    if len(argv) != 2:
        print(__doc__.splitlines()[2].strip())
        return 2
    try:
        same, reason = compare(argv[0], argv[1])
    except (GitError, OSError) as error:
        print("cannot compare the runtime sources: %s" % error)
        return 2
    print(reason)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
