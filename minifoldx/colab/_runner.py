"""Colab session orchestration and job-script generation."""

import subprocess
import tempfile
from pathlib import Path

import click

WEIGHTS_REMOTE  = "gdrive:Colab_Data/minifold_colab/weights"
JOBS_REMOTE     = "gdrive:Colab_Data/minifold_colab/jobs"
RCLONE_CONF     = Path.home() / ".config" / "rclone" / "rclone.conf"
RCLONE_VERSION  = "v1.75.0"


def _run(*cmd: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=True, **kwargs)


def colab(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return _run("colab", *args, **kwargs)


def rclone(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return _run("rclone", *args, **kwargs)


HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


def _has_drive() -> bool:
    """Return True if rclone is installed and gdrive remote is configured."""
    if not RCLONE_CONF.exists():
        return False
    if subprocess.run(["which", "rclone"], capture_output=True).returncode != 0:
        return False
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    return "gdrive:" in result.stdout


def check_deps() -> None:
    if subprocess.run(["which", "colab"], capture_output=True).returncode != 0:
        raise click.ClickException("'colab' not found on PATH.\n  pip install colab-cli")

    if _has_drive():
        scope_result = subprocess.run(
            ["rclone", "config", "show", "gdrive"],
            capture_output=True, text=True,
        )
        for line in scope_result.stdout.splitlines():
            if line.strip().startswith("scope"):
                scope = line.split("=", 1)[-1].strip()
                if scope == "drive":
                    click.echo(
                        "Warning: gdrive remote has scope = drive (full Drive access).\n"
                        "For safer operation, re-configure with drive.file scope:\n"
                        "  rclone config reconnect gdrive: --drive-scope drive.file\n"
                        "This restricts access to only files this app creates.",
                        err=True,
                    )
                break
    else:
        click.echo("Note: Drive not configured — weights will be downloaded from HuggingFace.", err=True)


def make_job_script(
    job_id: str,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    int8_esm2: bool,
    triton_kernels: bool = False,
    out_dest: str = "both",
    fasta_name: str = "input.fasta",
    has_drive: bool = True,
) -> str:
    """Return a Python script that runs on the Colab VM."""
    import subprocess as _sp
    _repo = Path(__file__).resolve().parent.parent.parent
    _commit = _sp.run(["git", "-C", str(_repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    predict_script = '"/content/MiniFoldX/minifoldx/pytorch/predict.py"'
    extra_flags = ""
    if triton_kernels:
        extra_flags += ', "--kernels"'
    if int8_esm2:
        extra_flags += ', "--int8-esm2"'

    predict_invocation = (
        f'run(sys.executable, {predict_script},\n'
        f'    FASTA, "--out_dir", OUTPUTS, "--cache", WEIGHTS,\n'
        f'    "--checkpoint", CKPT_PATH,\n'
        f'    "--model_size", MODEL_SIZE, "--token_per_batch", str(TOKEN_PER_BATCH),\n'
        f'    "--num_recycling", str(NUM_RECYCLING){extra_flags})'
    )

    int8_install = (
        'run("uv", "pip", "install", "--system", "bitsandbytes")'
        if int8_esm2 else
        '# (standard precision ESM2)'
    )

    triton_install = (
        'run("uv", "pip", "install", "--system", "triton")'
        if triton_kernels else
        '# (standard PyTorch ops)'
    )

    return f"""\
#!/usr/bin/env python3
# MiniFold job script — job {job_id}
import subprocess, sys, os

JOB_ID          = "{job_id}"
MODEL_SIZE      = "{model_size}"
TOKEN_PER_BATCH = {token_per_batch}
NUM_RECYCLING   = {num_recycling}
WEIGHTS_REMOTE  = "{WEIGHTS_REMOTE}"
JOBS_REMOTE     = "{JOBS_REMOTE}"
HAS_DRIVE       = {has_drive}

FASTA   = "/content/{fasta_name}"
WEIGHTS = "/content/weights"
OUTPUTS = "/content/outputs"
VM_CACHE      = os.path.join(WEIGHTS, "vm_cache")
UV_BIN_CACHE  = os.path.join(VM_CACHE, "uv")
MINIFOLD_TAR  = os.path.join(VM_CACHE, "minifold.tar.gz")
UV_PKG_CACHE  = os.path.join(VM_CACHE, "uv_packages")
ESM2_HF_REPO  = "z-ardern/MiniFoldX_weights"
_ESM2_FNAME   = "esm2_t36_3B_UR50D.pt"
_ESM2_CANDIDATES = [
    os.path.join(WEIGHTS, "checkpoints",        _ESM2_FNAME),
    os.path.join(WEIGHTS, "hub", "checkpoints", _ESM2_FNAME),
]
ESM2_PT_PATH = next((p for p in _ESM2_CANDIDATES if os.path.exists(p)), _ESM2_CANDIDATES[0])
ESM2_HUB_DIR = os.path.dirname(ESM2_PT_PATH)
CKPT_PATH    = os.path.join(WEIGHTS, f"minifold_{{MODEL_SIZE}}.safetensors")

os.environ["TORCH_HOME"] = WEIGHTS

def run(*cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode(errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))

def rclone_copy(src, dst, *extra):
    print(f"+ rclone copy {{src}} {{dst}}", flush=True)
    r = subprocess.run([
        "rclone", "copy", src, dst, "--progress",
        "--drive-chunk-size", "256M", "--transfers", "8", "--buffer-size", "256M",
        *extra,
    ])
    if r.returncode not in (0, 3):
        raise RuntimeError(f"rclone copy failed with exit code {{r.returncode}}")

os.makedirs(WEIGHTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(ESM2_HUB_DIR, exist_ok=True)
os.makedirs(VM_CACHE, exist_ok=True)
os.makedirs(UV_PKG_CACHE, exist_ok=True)

import time as _time
_t0 = _time.time()
_t_step = _time.time()

def _fmt(secs):
    m, s = divmod(int(secs), 60)
    return f"{{m:02d}}:{{s:02d}}"

def step(msg):
    global _t_step
    now = _time.time()
    step_elapsed = now - _t_step
    total_elapsed = now - _t0
    if total_elapsed > 1:
        print(f"  [{{_fmt(step_elapsed)}} step | {{_fmt(total_elapsed)}} total]", flush=True)
    print(f"\\n=== {{msg}} ===", flush=True)
    _t_step = now

if HAS_DRIVE:
    step("Installing rclone")
    RCLONE_VERSION = "{RCLONE_VERSION}"
    rclone_zip = f"/tmp/rclone-{{RCLONE_VERSION}}-linux-amd64.zip"
    rclone_url = f"https://github.com/rclone/rclone/releases/download/{{RCLONE_VERSION}}/rclone-{{RCLONE_VERSION}}-linux-amd64.zip"
    run("curl", "-fsSL", rclone_url, "-o", rclone_zip)
    run("unzip", "-q", "-o", rclone_zip, "-d", "/tmp/rclone_bin")
    run("cp", f"/tmp/rclone_bin/rclone-{{RCLONE_VERSION}}-linux-amd64/rclone", "/usr/local/bin/rclone")

step("GPU info")
run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")

if HAS_DRIVE:
    step("Pulling weights from Drive")
    rclone_copy(WEIGHTS_REMOTE, WEIGHTS)

step("Installing uv")
new_vm_cache = False
import shutil
if os.path.exists(UV_BIN_CACHE):
    print("Using cached uv binary", flush=True)
    shutil.copy(UV_BIN_CACHE, "/usr/local/bin/uv")
    os.chmod("/usr/local/bin/uv", 0o755)
else:
    run(sys.executable, "-m", "pip", "install", "-q", "uv")
    uv_path = shutil.which("uv")
    if uv_path and HAS_DRIVE:
        shutil.copy(uv_path, UV_BIN_CACHE)
        new_vm_cache = True

step("Setting up MiniFoldX code")
os.environ["UV_CACHE_DIR"] = UV_PKG_CACHE
MINIFOLDX_DIR = "/content/MiniFoldX"
PYTORCH_DIR   = f"{{MINIFOLDX_DIR}}/minifoldx/pytorch"
MINIFOLD_TAR_COMMIT = "{_commit}"
MINIFOLD_TAR_STAMP  = os.path.join(VM_CACHE, "minifold_commit.txt")
cached_commit = open(MINIFOLD_TAR_STAMP).read().strip() if os.path.exists(MINIFOLD_TAR_STAMP) else ""
if os.path.exists(MINIFOLD_TAR) and cached_commit == MINIFOLD_TAR_COMMIT:
    print(f"Using cached MiniFoldX repo ({{MINIFOLD_TAR_COMMIT[:8]}})", flush=True)
    run("tar", "-xzf", MINIFOLD_TAR, "-C", "/content")
else:
    if cached_commit and cached_commit != MINIFOLD_TAR_COMMIT:
        print(f"Code updated ({{cached_commit[:8]}} → {{MINIFOLD_TAR_COMMIT[:8]}}), re-cloning ...", flush=True)
    run("git", "clone", "--depth=1", "https://github.com/ZacharyArdern/MiniFoldX.git", MINIFOLDX_DIR)
    if HAS_DRIVE:
        run("tar", "-czf", MINIFOLD_TAR, "-C", "/content", "MiniFoldX")
        open(MINIFOLD_TAR_STAMP, "w").write(MINIFOLD_TAR_COMMIT)
        new_vm_cache = True

step("Installing Python dependencies")
run("uv", "pip", "install", "--system", "-e", PYTORCH_DIR,
    "huggingface_hub[hf_xet]", "fair-esm", "safetensors", "ml_collections",
    "dm-tree", "modelcif", "einops", "edit_distance")
{int8_install}
{triton_install}

step("Preparing MiniFold checkpoint")
os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "1")
from huggingface_hub import hf_hub_download
ckpt_name = f"minifold_{{MODEL_SIZE}}.safetensors"
ckpt_dest = os.path.join(WEIGHTS, ckpt_name)
if not os.path.exists(ckpt_dest):
    print(f"Downloading {{ckpt_name}} from HuggingFace ...", flush=True)
    hf_hub_download(repo_id=ESM2_HF_REPO, filename=ckpt_name, local_dir=WEIGHTS, local_dir_use_symlinks=False)
    if HAS_DRIVE:
        step("Saving MiniFold checkpoint to Drive")
        rclone_copy(WEIGHTS, WEIGHTS_REMOTE, "--exclude", "vm_cache/**", "--exclude", "hub/**")
else:
    print(f"Using cached {{ckpt_name}}", flush=True)

if HAS_DRIVE and new_vm_cache:
    step("Saving VM cache to Drive")
    rclone_copy(VM_CACHE, f"{{WEIGHTS_REMOTE}}/vm_cache")

# Re-evaluate ESM2 path after Drive pull
ESM2_PT_PATH = next((p for p in _ESM2_CANDIDATES if os.path.exists(p)), _ESM2_CANDIDATES[0])
ESM2_HUB_DIR = os.path.dirname(ESM2_PT_PATH)
esm2_cached_before = os.path.exists(ESM2_PT_PATH)
if esm2_cached_before:
    print(f"ESM2 weights cached at {{ESM2_PT_PATH}}", flush=True)
else:
    print("ESM2 not cached — will download from fbaipublicfiles during model load.", flush=True)

# --- Periodic checkpoint: push outputs every 10 min so results survive connection loss ---
import threading as _threading

OUT_DEST   = "{out_dest}"
job_remote = f"{{JOBS_REMOTE}}/{{JOB_ID}}/outputs"
CHECKPOINT_INTERVAL = 600  # seconds

_checkpoint_stop = _threading.Event()
_checkpoint_lock = _threading.Lock()

def _do_checkpoint(label="checkpoint"):
    with _checkpoint_lock:
        try:
            n = sum(1 for _, _, fs in os.walk(OUTPUTS) for f in fs if f.endswith(".pdb")) if os.path.isdir(OUTPUTS) else 0
            if n == 0:
                return
            if HAS_DRIVE and OUT_DEST in ("both", "drive"):
                r = subprocess.run([
                    "rclone", "copy", OUTPUTS, job_remote,
                    "--transfers", "4", "--drive-chunk-size", "128M",
                ], capture_output=True)
                if r.returncode in (0, 3):
                    print(f"[{{label}}] {{n}} structure(s) pushed to Drive", flush=True)
                else:
                    print(f"[{{label}}] rclone exit {{r.returncode}}: {{r.stderr.decode(errors='replace')}}", flush=True)
            else:
                import tarfile
                with tarfile.open("/tmp/outputs_checkpoint.tar.gz", "w:gz") as tf:
                    tf.add(OUTPUTS, arcname="outputs")
                print(f"[{{label}}] {{n}} structure(s) tarred to /tmp/outputs_checkpoint.tar.gz", flush=True)
        except Exception as e:
            print(f"[{{label}}] Warning: {{e}}", flush=True)

def _checkpoint_loop():
    while not _checkpoint_stop.wait(CHECKPOINT_INTERVAL):
        _do_checkpoint()

_ckpt_thread = _threading.Thread(target=_checkpoint_loop, daemon=True)
_ckpt_thread.start()

step("Running structure prediction")
{predict_invocation}

_checkpoint_stop.set()
_ckpt_thread.join(timeout=30)

ESM2_PT_PATH = next((p for p in _ESM2_CANDIDATES if os.path.exists(p)), _ESM2_CANDIDATES[0])
ESM2_HUB_DIR = os.path.dirname(ESM2_PT_PATH)
_esm2_rel = os.path.relpath(ESM2_HUB_DIR, WEIGHTS)
if HAS_DRIVE and not esm2_cached_before and os.path.exists(ESM2_PT_PATH):
    step("Caching ESM2 to Drive")
    rclone_copy(ESM2_HUB_DIR, f"{{WEIGHTS_REMOTE}}/{{_esm2_rel}}")

step("Saving results")
_do_checkpoint(label="final")
if not (HAS_DRIVE and OUT_DEST in ("both", "drive")):
    print("Results in /content/outputs (download via colab CLI).", flush=True)

print(f"\\n=== Done [{{_fmt(_time.time() - _t0)}} total] ===", flush=True)
"""


def make_training_script(
    job_id: str,
    epochs: int = 15,
    lr: float = 1e-3,
    max_len: int = 300,
    min_len: int = 40,
    ckpt_every: int = 500,
    out_dest: str = "both",
    has_drive: bool = True,
    resume_job: str = "",
) -> str:
    """Return a Python script that trains fc_s/fc_z for ESMC 600M on a Colab VM."""
    import subprocess as _sp
    _repo   = Path(__file__).resolve().parent.parent.parent
    _commit = _sp.run(["git", "-C", str(_repo), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()

    return f"""\
#!/usr/bin/env python3
# MiniFoldX ESMC training — job {job_id}
import subprocess, sys, os, time

JOB_ID         = "{job_id}"
WEIGHTS_REMOTE = "{WEIGHTS_REMOTE}"
JOBS_REMOTE    = "{JOBS_REMOTE}"
HAS_DRIVE      = {has_drive}
OUT_DEST       = "{out_dest}"
RCLONE_VERSION = "{RCLONE_VERSION}"

WEIGHTS = "/content/weights"
OUTPUTS = "/content/outputs"
VM_CACHE = os.path.join(WEIGHTS, "vm_cache")
os.makedirs(WEIGHTS, exist_ok=True); os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(VM_CACHE, exist_ok=True)

T0 = time.time()
def log(msg): print(f"[{{time.time()-T0:6.1f}}s] {{msg}}", flush=True)

def run(*cmd):
    log("+ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode(errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))

def rclone_copy(src, dst, *extra):
    r = subprocess.run(["rclone","copy",src,dst,"--progress",
                        "--drive-chunk-size","256M","--transfers","8","--buffer-size","256M",*extra])
    if r.returncode not in (0, 3):
        raise RuntimeError(f"rclone copy failed: exit {{r.returncode}}")

log("=== GPU info ===")
run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")

if HAS_DRIVE:
    log("=== Installing rclone ===")
    rclone_zip = f"/tmp/rclone-{{RCLONE_VERSION}}-linux-amd64.zip"
    run("curl","-fsSL",f"https://github.com/rclone/rclone/releases/download/{{RCLONE_VERSION}}/rclone-{{RCLONE_VERSION}}-linux-amd64.zip","-o",rclone_zip)
    run("unzip","-q","-o",rclone_zip,"-d","/tmp/rclone_bin")
    run("cp",f"/tmp/rclone_bin/rclone-{{RCLONE_VERSION}}-linux-amd64/rclone","/usr/local/bin/rclone")
    log("=== Pulling weights cache from Drive ===")
    rclone_copy(WEIGHTS_REMOTE, WEIGHTS)

log("=== Installing uv ===")
run(sys.executable, "-m", "pip", "install", "-q", "uv")

log("=== Cloning MiniFoldX ===")
MINIFOLDX_DIR = "/content/MiniFoldX"
MINIFOLD_TAR  = os.path.join(VM_CACHE, "minifold.tar.gz")
MINIFOLD_STAMP = os.path.join(VM_CACHE, "minifold_commit.txt")
cached_commit = open(MINIFOLD_STAMP).read().strip() if os.path.exists(MINIFOLD_STAMP) else ""
if os.path.exists(MINIFOLD_TAR) and cached_commit == "{_commit}":
    run("tar","-xzf",MINIFOLD_TAR,"-C","/content")
else:
    run("git","clone","--depth=1","https://github.com/ZacharyArdern/MiniFoldX.git",MINIFOLDX_DIR)
    if HAS_DRIVE:
        run("tar","-czf",MINIFOLD_TAR,"-C","/content","MiniFoldX")
        open(MINIFOLD_STAMP,"w").write("{_commit}")

log("=== Installing Python dependencies ===")
os.environ["UV_CACHE_DIR"] = os.path.join(VM_CACHE, "uv_packages")
run("uv","pip","install","--system","-q",
    "-e", os.path.join(MINIFOLDX_DIR, "minifoldx", "pytorch"),
    "git+https://github.com/evolutionaryScale/esm.git",
    "safetensors","huggingface_hub[hf_xet]","einops",
    "dm-tree","ml-collections","modelcif","edit_distance","fair-esm")

RESUME_JOB = "{resume_job}"
if RESUME_JOB and HAS_DRIVE:
    log("=== Downloading resume checkpoint from Drive ===")
    rclone_copy(f"{{JOBS_REMOTE}}/{{RESUME_JOB}}/outputs", OUTPUTS)
    log(f"  Checkpoints from job {{RESUME_JOB}} ready in {{OUTPUTS}}")

log("=== Launching training ===")
TRAIN_SCRIPT = os.path.join(MINIFOLDX_DIR, "minifoldx", "colab", "_train_esmc_fcsz.py")
resume_flag = ["--resume", "auto"] if RESUME_JOB else []
run(sys.executable, TRAIN_SCRIPT,
    "--epochs", "{epochs}",
    "--lr",     "{lr}",
    "--max-len","{max_len}",
    "--min-len","{min_len}",
    "--ckpt-every","{ckpt_every}",
    "--weights",WEIGHTS,
    "--out",    OUTPUTS,
    *resume_flag)

log("=== Saving results to Drive ===")
if HAS_DRIVE and OUT_DEST in ("both","drive"):
    rclone_copy(OUTPUTS, f"{{JOBS_REMOTE}}/{{JOB_ID}}/outputs")
    log("Results pushed to Drive.")

log(f"=== Done [{{(time.time()-T0)/60:.1f}} min total] ===")
"""


def _download_from_vm(session: str, remote_dir: str, local_dir: Path) -> None:
    """Tar outputs on VM, download the archive, extract locally."""
    import tarfile
    remote_tar = "/tmp/minifold_outputs.tar.gz"
    colab("exec", "-s", session, "--timeout", "60", "-f", "/dev/stdin",
          input=f"import subprocess; subprocess.run(['tar','-czf','{remote_tar}','-C','{remote_dir}','.'], check=True)",
          text=True)
    local_tar = local_dir / "outputs.tar.gz"
    colab("download", "-s", session, remote_tar, str(local_tar))
    with tarfile.open(local_tar) as tf:
        tf.extractall(local_dir)
    local_tar.unlink()


def run_prediction(
    fasta_path: Path,
    job_id: str,
    session: str,
    model_size: str,
    token_per_batch: int,
    num_recycling: int,
    timeout: int,
    int8_esm2: bool,
    triton_kernels: bool = False,
    out_dest: str = "both",
) -> Path:
    """Upload FASTA, execute job script, download results. Returns local output dir."""

    drive = _has_drive()

    # If Drive not available, can't push results there
    if not drive and out_dest in ("both", "drive"):
        click.echo("Note: Drive not configured — results will be downloaded locally only.")
        out_dest = "pwd"

    need_drive = drive and out_dest in ("both", "drive")
    need_local = out_dest in ("both", "pwd")

    # Upload rclone credentials if Drive is configured
    if drive:
        click.echo("[2/5] Uploading rclone credentials ...")
        colab("exec", "-s", session, "--timeout", "10",
              "-f", "/dev/stdin",
              input="import os; os.makedirs('/root/.config/rclone', exist_ok=True)",
              text=True)
        colab("upload", "-s", session,
              str(RCLONE_CONF), "/root/.config/rclone/rclone.conf")
    else:
        click.echo("[2/5] No Drive configured — skipping rclone credentials ...")

    # Upload HF token if available (speeds up downloads via XET)
    if HF_TOKEN_PATH.exists():
        colab("exec", "-s", session, "--timeout", "10",
              "-f", "/dev/stdin",
              input="import os; os.makedirs('/root/.cache/huggingface', exist_ok=True)",
              text=True)
        colab("upload", "-s", session,
              str(HF_TOKEN_PATH), "/root/.cache/huggingface/token")

    # Upload FASTA — preserve extension so gzip files are detected correctly
    fasta_suffix = "".join(fasta_path.suffixes[-2:]) or fasta_path.suffix  # e.g. .fasta.gz or .fa.gz
    fasta_name = f"input{fasta_suffix}" if fasta_suffix else "input.fasta"
    remote_fasta = f"/content/{fasta_name}"
    click.echo(f"[3/5] Uploading {fasta_path.name} ...")
    colab("upload", "-s", session, str(fasta_path), remote_fasta)

    # Generate and run job script
    click.echo(f"[4/5] Running prediction (timeout {timeout}s) ...")
    job_script = make_job_script(
        job_id, model_size, token_per_batch, num_recycling,
        int8_esm2, triton_kernels, out_dest, fasta_name, drive,
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(job_script)
        job_script_path = f.name

    colab("exec", "-s", session, "-f", job_script_path, "--timeout", str(timeout))

    # Download results
    out_dir = Path("./minifold_results") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if need_local:
        click.echo("[5/5] Downloading results ...")
        if need_drive:
            result = subprocess.run(
                ["rclone", "copy", f"{JOBS_REMOTE}/{job_id}/outputs",
                 str(out_dir), "--progress"]
            )
            if result.returncode not in (0, 3):
                raise click.ClickException(f"rclone download failed (exit {result.returncode})")
        else:
            _download_from_vm(session, "/content/outputs", out_dir)
    else:
        click.echo("[5/5] Skipping local download (--out drive) ...")

    return out_dir
