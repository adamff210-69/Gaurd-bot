"""Kaggle bootstrap helpers for GuardBot.

Running this repo inside a Kaggle notebook breaks in four specific ways.
`setup()` fixes all of them, in the right order:

1. READ-ONLY CODE — if you attach the repo as a Kaggle *Dataset* it lands in
   `/kaggle/input/...`, which is mounted read-only. The pipeline writes
   `logs/blocked_events.jsonl` and the eval harness writes
   `eval/results.json`, so we make a writable copy under `/tmp`.
2. SECRET ORDERING — `guardbot/config.py` reads `os.environ` **at import
   time**. Kaggle secrets therefore have to be injected into the environment
   *before* `import guardbot`, or the app silently stays in mock mode.
3. HF CACHE — model/GGUF downloads need a writable cache dir, and the
   tokenizer-parallelism fork warning is noise in a notebook.
4. HARDWARE — `JUDGE_N_THREADS` defaults to 2, which under-uses a Kaggle
   box; and GPU offload is opt-in because it needs a CUDA-built llama.cpp.

Typical notebook use (after the install cell):

    sys.path.append("/tmp/guardbot-src/kaggle")
    from bootstrap import setup
    info = setup()
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Iterable, Sequence

REPO_URL = "https://github.com/adamff210-69/rag.git"

# writable scratch copy of the repo (fast local disk, wiped per session)
SRC_DIR = "/tmp/guardbot-src"

# packages Kaggle images do NOT ship
BASE_PKGS: tuple[str, ...] = (
    "langgraph>=0.2",
    "langchain>=0.3",
    "langchain-groq>=0.2",
    "langchain-openai>=0.2",
    "python-dotenv>=1.0",
    "huggingface_hub>=0.24",
)

# llama.cpp prebuilt CPU wheels — avoids a 10-minute source compile
CPU_WHEEL_INDEXES: tuple[str, ...] = (
    "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    "https://parisneo.github.io/llama-cpp-python-wheels/whl/cpu/",
)

# Kaggle Add-ons -> Secrets names to pull into os.environ
DEFAULT_SECRETS: tuple[str, ...] = ("GROQ_API_KEY", "OPENROUTER_API_KEY")

COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv", "logs")


# ----------------------------------------------------------------- platform
def is_kaggle() -> bool:
    return os.path.isdir("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, echoing it (notebook-friendly, no silent failures)."""
    _log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=False, **kw)


# ------------------------------------------------------------------ install
def install_llama_cpp(verbose: bool = False) -> str:
    """Install llama-cpp-python. Returns the strategy that worked.

    Tries prebuilt CPU wheels first (seconds); falls back to compiling from
    source (minutes); reports "unavailable" instead of raising, because
    Detector 2 degrades to the API judge / heuristic without it.
    """
    try:
        import llama_cpp  # noqa: F401
        _log(f"llama-cpp-python already importable ({llama_cpp.__version__})")
        return "already-installed"
    except Exception:
        pass

    py = sys.executable
    for index in CPU_WHEEL_INDEXES:
        _log(f"trying prebuilt CPU wheel index: {index}")
        cp = _run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                   "llama-cpp-python", "--extra-index-url", index])
        if cp.returncode == 0 and _can_import("llama_cpp"):
            return "prebuilt-wheel"

    _log("no prebuilt wheel — compiling from source (this takes several minutes)")
    env = {**os.environ, "CMAKE_ARGS": "-DGGML_NATIVE=OFF"}
    cp = _run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check",
               "--no-cache-dir", "llama-cpp-python"],
              env=env, stdout=None if verbose else subprocess.DEVNULL)
    if cp.returncode == 0 and _can_import("llama_cpp"):
        return "source-build"

    _log("⚠️  llama-cpp-python unavailable. Set JUDGE_BACKEND=api (uses your "
         "Groq/OpenRouter key) or accept the regex heuristic fallback.")
    return "unavailable"


def install(packages: Iterable[str] = BASE_PKGS,
            llama_cpp: bool = True,
            torch: bool = False) -> dict:
    """pip-install what the Kaggle image is missing. Returns a status dict."""
    py = sys.executable
    pkgs = [p for p in packages]
    status: dict = {"base": None, "torch": None, "llama_cpp": None}

    if pkgs:
        cp = _run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pkgs])
        status["base"] = "ok" if cp.returncode == 0 else f"exit {cp.returncode}"

    if torch and not _can_import("torch"):
        # CPU-only wheel: Kaggle already ships CUDA torch, so this is a
        # fallback for images that don't.
        cp = _run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                   "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
        status["torch"] = "ok" if cp.returncode == 0 else f"exit {cp.returncode}"

    if llama_cpp:
        status["llama_cpp"] = install_llama_cpp()
    return status


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------- source
def clone_repo(dest: str = SRC_DIR, ref: str | None = None,
               url: str = REPO_URL, fallback_default: bool = True) -> str:
    """git clone the public repo into a writable dir. Returns the repo root.

    If `ref` no longer exists (e.g. the branch was merged and deleted) and
    fallback_default is set, retries against the default branch. Verifies the
    kaggle helpers came along, because a clone of a branch without them fails
    confusingly later on `import bootstrap`.
    """
    if os.path.isfile(os.path.join(dest, "guardbot", "config.py")):
        _log(f"repo already present at {dest}")
        return dest
    if os.path.isdir(dest):
        _force_writable(dest)
        shutil.rmtree(dest, ignore_errors=True)

    def _attempt(branch: str | None) -> bool:
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        return _run(cmd + [url, dest]).returncode == 0

    ok = _attempt(ref)
    if not ok and ref and fallback_default:
        _log(f"branch {ref!r} unavailable → retrying on the default branch")
        ok = _attempt(None)
    if not ok:
        raise RuntimeError(
            "git clone failed — is the notebook's Internet switch ON? "
            "(Settings → Internet → On)")
    if not os.path.isfile(os.path.join(dest, "kaggle", "bootstrap.py")):
        raise RuntimeError(
            f"cloned code has no kaggle/bootstrap.py. The Kaggle helpers live on "
            f"branch {ref or '(default)'} — pass a ref that has them, or merge "
            f"that branch into the default branch.")
    return dest


def locate_code(extra_paths: Iterable[str] = ()) -> str | None:
    """Find an existing GuardBot checkout: attached dataset, clone, or cwd."""
    # Order matters: prefer the writable copy we may already have made, then
    # /kaggle/working, then the read-only /kaggle/input mounts, then the cwd.
    patterns: list[str] = [os.path.join(SRC_DIR, "guardbot", "config.py")]
    for base in extra_paths:
        patterns.append(os.path.join(base, "guardbot", "config.py"))
    if is_kaggle():
        patterns += [
            "/kaggle/working/guardbot/config.py",
            "/kaggle/working/*/guardbot/config.py",
            "/kaggle/input/*/guardbot/config.py",
            "/kaggle/input/*/*/guardbot/config.py",
        ]
    patterns += [
        "guardbot/config.py",
        os.path.expanduser("~/rag/guardbot/config.py"),
    ]
    for pat in patterns:
        for hit in sorted(glob.glob(pat)):
            return os.path.dirname(os.path.dirname(os.path.abspath(hit)))
    return None


def ensure_writable(src: str, dest: str = SRC_DIR, refresh: bool = False) -> tuple[str, bool]:
    """Return (repo_root, copied?). Copies read-only mounts to writable disk.

    An existing writable copy at `dest` is REUSED (so a second `setup()` call
    doesn't throw away `eval/results.json`); pass refresh=True to re-copy.
    """
    marker = os.path.join(dest, "guardbot", "config.py")
    if not refresh and os.path.isfile(marker) and _dir_writable(dest):
        if os.path.abspath(src) != os.path.abspath(dest):
            _log(f"reusing existing writable copy at {dest} "
                 "(setup(refresh=True) to re-copy)")
            return dest, True
        return dest, False

    writable = os.access(src, os.W_OK) and _dir_writable(src)
    if writable and os.path.abspath(src) != os.path.abspath(dest):
        return src, False
    if not writable:
        _log(f"{src} is read-only → copying to {dest}")
    if os.path.isdir(dest):
        # a previous partial copy may have left read-only files behind
        _force_writable(dest)
        shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    # copyfile (not copy/copy2) so the destination does NOT inherit the
    # read-only mode bits of a /kaggle/input mount — otherwise the "writable"
    # copy would still reject writes to logs/ and eval/results.json.
    shutil.copytree(src, dest, ignore=COPY_IGNORE, dirs_exist_ok=True,
                    copy_function=shutil.copyfile)
    _force_writable(dest)
    return dest, True


def _force_writable(root: str) -> None:
    """Clear inherited read-only bits across a tree (dirs 0755, files 0644)."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for d in dirnames:
            try:
                os.chmod(os.path.join(dirpath, d), 0o755)
            except OSError:
                pass
        for f in filenames:
            try:
                os.chmod(os.path.join(dirpath, f), 0o644)
            except OSError:
                pass
    try:
        os.chmod(root, 0o755)
    except OSError:
        pass


def _dir_writable(path: str) -> bool:
    """os.access lies on some fuse mounts — verify with a real write."""
    try:
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ secrets
def load_secrets(names: Sequence[str] = DEFAULT_SECRETS,
                 required: Sequence[str] = ()) -> dict:
    """Copy Kaggle Add-on secrets into os.environ (never overwriting a value
    you already set). Returns {name: "set" | "missing" | "n/a"}."""
    client = None
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    except Exception:
        pass

    found: dict[str, str] = {}
    for name in names:
        if os.environ.get(name, "").strip():
            found[name] = "set (already in env)"
            continue
        if client is None:
            found[name] = "n/a (not on Kaggle)"
            continue
        try:
            val, _ = client.get_secret_with_label(name)
        except Exception:
            try:
                val = client.get_secret(name)
            except Exception:
                val = ""
        if val and val.strip():
            os.environ[name] = val.strip()
            found[name] = "set (from Kaggle secret)"
        else:
            found[name] = "missing"

    for name in required:
        if found.get(name, "").startswith("missing") or found.get(name) == "n/a (not on Kaggle)":
            raise RuntimeError(
                f"Kaggle secret '{name}' is empty. Add it under "
                "Add-ons → Secrets, then attach it to this notebook "
                "(right panel → Secrets).")
    return found


# -------------------------------------------------------------------- setup
def setup(src: str | None = None,
          dest: str = SRC_DIR,
          secrets: Sequence[str] = DEFAULT_SECRETS,
          judge_backend: str | None = None,
          policy: str | None = None,
          hf_home: str = "/tmp/hf",
          threads: int | None = None,
          log_dir: str | None = None,
          gpu: bool = False,
          refresh: bool = False) -> dict:
    """Make the repo runnable on Kaggle and import `guardbot`.

    Returns a dict describing what was configured. Safe to call twice.
    """
    t0 = time.perf_counter()

    # --- 3. env for HuggingFace + tokenizer noise (before any import)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # --- 4. hardware defaults
    cpus = os.cpu_count() or 2
    os.environ.setdefault("JUDGE_N_THREADS", str(threads or min(4, cpus)))
    if gpu:
        os.environ.setdefault("D1_DEVICE", "0")
        os.environ.setdefault("JUDGE_N_GPU_LAYERS", "-1")

    # --- 2. secrets BEFORE guardbot.config is imported
    secret_status = load_secrets(secrets)
    if judge_backend:
        os.environ["JUDGE_BACKEND"] = judge_backend
    if policy:
        os.environ["POLICY"] = policy

    # --- 1. writable source tree
    src = src or locate_code()
    if src is None:
        _log("no local checkout found → cloning the public repo")
        src = clone_repo(dest)
    repo_root, copied = ensure_writable(src, dest, refresh=refresh)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    kaggle_dir = os.path.join(repo_root, "kaggle")
    if os.path.isdir(kaggle_dir) and kaggle_dir not in sys.path:
        sys.path.append(kaggle_dir)

    # --- import + patch write paths
    for mod in [m for m in list(sys.modules) if m == "guardbot" or m.startswith("guardbot.")]:
        del sys.modules[mod]          # config caches env at import time
    from guardbot import config        # noqa: E402

    log_dir = log_dir or ("/kaggle/working/logs" if is_kaggle()
                          else os.path.join(repo_root, "logs"))
    os.makedirs(log_dir, exist_ok=True)
    config.LOG_PATH = os.path.join(log_dir, "blocked_events.jsonl")

    info = {
        "repo_root": repo_root,
        "copied_to_writable": copied,
        "log_path": config.LOG_PATH,
        "provider": config.provider(),
        "judge_backend": config.JUDGE_BACKEND,
        "policy": config.DEFAULT_POLICY,
        "secrets": secret_status,
        "llama_cpp": _can_import("llama_cpp"),
        "transformers": _can_import("transformers"),
        "torch": _can_import("torch"),
        "cpus": cpus,
        "judge_threads": config.JUDGE_N_THREADS,
        "setup_seconds": round(time.perf_counter() - t0, 2),
    }
    report(info)
    return info


def report(info: dict) -> None:
    _log("\n================ GuardBot on Kaggle ================")
    _log(f"code          : {info['repo_root']}"
         + ("  (writable copy)" if info["copied_to_writable"] else ""))
    _log(f"main LLM      : {info['provider']}"
         + ("   ← mock mode, add a Kaggle secret for real answers"
            if info["provider"] == "mock" else ""))
    _log(f"judge backend : {info['judge_backend']}"
         f"   (llama-cpp-python: {'yes' if info['llama_cpp'] else 'NO'})")
    _log(f"policy        : {info['policy']}")
    _log(f"blocked-log   : {info['log_path']}")
    _log(f"hardware      : {info['cpus']} CPUs, judge threads="
         f"{info['judge_threads']}, torch={info['torch']}, "
         f"transformers={info['transformers']}")
    _log(f"secrets       : {info['secrets']}")
    _log(f"setup took    : {info['setup_seconds']}s")
    _log("====================================================\n")


def publish_outputs(repo_root: str | None = None,
                    out_dir: str | None = None) -> list[str]:
    """Copy artifacts into /kaggle/working so they appear in notebook Output.

    `/tmp` is wiped when the session ends and is not downloadable; anything you
    want to keep (eval results, blocked-event log) must be published here.
    Off Kaggle (or if the dir is not writable) falls back to ./kaggle-outputs.
    """
    repo_root = repo_root or locate_code()
    if repo_root is None:
        raise RuntimeError("cannot locate the repo — call setup() first")

    if out_dir is None:
        out_dir = "/kaggle/working" if is_kaggle() else "kaggle-outputs"
    try:
        os.makedirs(out_dir, exist_ok=True)
        if not _dir_writable(out_dir):
            raise OSError("not writable")
    except OSError:
        fallback = os.path.join(os.getcwd(), "kaggle-outputs")
        _log(f"⚠️  {out_dir} is not writable → publishing to {fallback}")
        out_dir = fallback
        os.makedirs(out_dir, exist_ok=True)

    written: list[str] = []
    candidates = [
        (os.path.join(repo_root, "eval", "results.json"), "eval_results.json"),
        (os.path.join(repo_root, "logs", "blocked_events.jsonl"), "blocked_events.jsonl"),
        (os.path.join("/kaggle/working/logs", "blocked_events.jsonl"), None),
    ]
    for src, rename in candidates:
        if os.path.isfile(src):
            dst = os.path.join(out_dir, rename or os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                try:
                    shutil.copyfile(src, dst)
                except OSError as e:
                    _log(f"⚠️  could not publish {src}: {e}")
                    continue
            if dst not in written:
                written.append(dst)
    _log("published: " + (", ".join(written) if written else "(nothing yet)"))
    return written


if __name__ == "__main__":
    setup()
