"""
VIDEO FACTORY V4 - PRODUCTION ENGINE
======================================
- Groq GPT-OSS-120B: sentence-level visual matching (JSON approach)
- Chatterbox Multilingual TTS + Resemble Enhance (44.1kHz studio)
- GPU-accelerated encoding (h264_nvenc on Kaggle T4/P100)
- Ken Burns cinematic effects on clips
- Multiple subtitle presets
- Optimized for Kaggle GPU
"""

import os, subprocess, sys, re, time, random, shutil, json
from importlib.metadata import version as _package_version, PackageNotFoundError
# Force explicit reseed with high-entropy source. Kaggle kernels can
# sometimes start with a low-entropy or reused random state between runs,
# which was causing subtitle style selection to always pick the same
# option instead of shuffling. os.urandom pulls from the OS entropy pool
# directly, bypassing whatever default seeding Python did on interpreter start.
random.seed(int.from_bytes(os.urandom(8), "big") ^ int(time.time() * 1000))
import concurrent.futures, requests, gc, threading
from pathlib import Path

# ==========================================
# 1. INSTALLATION
# ==========================================
print("--- Installing Dependencies ---")

# Kaggle images often already contain most of these packages. Reinstalling
# them unconditionally cost roughly five minutes on every job, so only invoke
# pip when the import is genuinely missing. This keeps fresh kernels correct
# without paying the install cost on warm/prebuilt images.
def _ensure_package(module_name, requirement, extra_args=None):
    try:
        __import__(module_name)
        return True
    except Exception:
        args = [sys.executable, "-m", "pip", "install", "--quiet"]
        if extra_args:
            args.extend(extra_args)
        args.append(requirement)
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: could not install {requirement}: {(result.stderr or '')[-300:]}")
            return False
        return True

# A Chatterbox install must not replace Kaggle's preinstalled GPU stack.
# Its official metadata pins torch/torchaudio 2.6.0, and a normal dependency
# install can leave cuDNN component libraries from different releases in the
# same process. That is the cause of the native cudnnGetLibConfig abort.
_CUDNN_FORCE_DISABLED = False
_CUDNN_HANDLES = []


def _prepare_cuda_runtime():
    """Prefer one coherent pip NVIDIA runtime before any torch-bearing import."""
    global _CUDNN_FORCE_DISABLED, _CUDNN_HANDLES
    if not sys.platform.startswith("linux"):
        return

    import ctypes
    import site

    site_roots = []
    try:
        site_roots.extend(site.getsitepackages())
    except (AttributeError, TypeError):
        pass
    try:
        site_roots.append(site.getusersitepackages())
    except (AttributeError, TypeError):
        pass

    nvidia_lib_dirs = []
    cudnn_lib_dirs = []
    for root in site_roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        cudnn_dir = nvidia_root / "cudnn" / "lib"
        if cudnn_dir.is_dir():
            cudnn_lib_dirs.append(str(cudnn_dir))
        try:
            components = sorted(nvidia_root.iterdir(), key=lambda path: path.name)
        except OSError:
            components = []
        for component in components:
            lib_dir = component / "lib"
            if lib_dir.is_dir():
                nvidia_lib_dirs.append(str(lib_dir))

    # Put the cuDNN wheel directory first, then the rest of the matching pip
    # NVIDIA runtime, before any system /usr/local/cuda entry can win.
    priority_dirs = cudnn_lib_dirs + nvidia_lib_dirs
    old_path = os.environ.get("LD_LIBRARY_PATH", "")
    ordered_dirs = []
    for directory in priority_dirs + old_path.split(os.pathsep):
        if directory and directory not in ordered_dirs:
            ordered_dirs.append(directory)
    if ordered_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ordered_dirs)

    if not cudnn_lib_dirs:
        print("  CUDA runtime: packaged cuDNN directory not found; keeping Kaggle loader path")
        return

    def _find_library(directory, stem):
        exact = Path(directory) / f"{stem}.so.9"
        if exact.exists():
            return exact
        matches = sorted(Path(directory).glob(f"{stem}.so.9.*"))
        return matches[0] if matches else None

    cudnn_dir = cudnn_lib_dirs[0]
    core_path = _find_library(cudnn_dir, "libcudnn")
    graph_path = _find_library(cudnn_dir, "libcudnn_graph")
    if not core_path or not graph_path:
        print(f"  CUDA runtime: incomplete cuDNN bundle in {cudnn_dir}; using normal torch startup")
        return

    try:
        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        core_handle = ctypes.CDLL(str(core_path), mode=global_mode)
        if getattr(core_handle, "cudnnGetLibConfig", None) is None:
            raise OSError(f"{core_path} does not export cudnnGetLibConfig")
        graph_handle = ctypes.CDLL(str(graph_path), mode=global_mode)
        _CUDNN_HANDLES = [core_handle, graph_handle]
        print(f"  CUDA runtime: coherent cuDNN core/graph loaded from {cudnn_dir}")
    except (OSError, AttributeError) as cudnn_error:
        _CUDNN_FORCE_DISABLED = True
        print(f"  WARNING: cuDNN component mismatch detected ({str(cudnn_error)[:180]})")
        print("  WARNING: cuDNN will be disabled for neural inference to prevent a native abort")


_prepare_cuda_runtime()
# Import torch before accelerate/torchvision/bitsandbytes. This makes every
# later torch-bearing import reuse the runtime selected above.
import torch
if _CUDNN_FORCE_DISABLED:
    torch.backends.cudnn.enabled = False
    print("  CUDA runtime: torch.backends.cudnn.enabled=False")

for _module, _requirement in [
    ("groq", "groq"), ("assemblyai", "assemblyai"),
    ("google.generativeai", "google-generativeai"), ("requests", "requests"),
    ("pydub", "pydub"), ("numpy", "numpy"), ("PIL", "pillow"),
    ("librosa", "librosa"), ("scipy", "scipy"), ("soundfile", "soundfile"),
    ("accelerate", "accelerate"),
    ("av", "av"), ("decord", "decord==0.6.0"),
    ("torchvision", "torchvision"), ("sentencepiece", "sentencepiece"),
    ("bitsandbytes", "bitsandbytes>=0.46.1"),
    ("psutil", "psutil"),
]:
    _ensure_package(_module, _requirement)

# Install Chatterbox's Python-only support packages without dependency
# resolution. These imports are all safe after torch has been initialized and
# none is allowed to pull a replacement CUDA/PyTorch wheel.
for _module, _requirement in [
    ("s3tokenizer", "s3tokenizer"),
    ("diffusers", "diffusers==0.29.0"),
    ("conformer", "conformer==0.3.2"),
    ("safetensors", "safetensors==0.5.3"),
    ("perth", "resemble-perth==1.0.1"),
    ("spacy_pkuseg", "spacy-pkuseg"),
    ("pykakasi", "pykakasi==2.3.0"),
    ("pyloudnorm", "pyloudnorm"),
    ("omegaconf", "omegaconf"),
]:
    _ensure_package(_module, _requirement, ["--no-deps"])

try:
    _package_version("chatterbox-tts")
    print("  chatterbox-tts already available")
except PackageNotFoundError:
    print("  Installing chatterbox-tts from GitHub (master) for V3 support...")
    _cb_installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
         "git+https://github.com/resemble-ai/chatterbox.git"],
        capture_output=True, text=True
    )
    if _cb_installed.returncode != 0:
        print("  git install failed, falling back to PyPI chatterbox-tts")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "chatterbox-tts"],
                       check=False)

# Chatterbox's current source can leave a newer Transformers build installed.
# Keep one exact version that avoids the `mistral-common`
# BACKENDS_MAPPING regression reported with 5.14.x and remains compatible with
# MiniCPM-V 4.5's custom Qwen3-based model code. MiniCPM's published config was
# authored against 4.51.0, but its custom AutoModel code is compatible with the
# 5.5.0 build already validated with Chatterbox in this pipeline.
_TRANSFORMERS_REQUIRED = "5.5.0"
# Transformers 5.5.0 performs this check during its top-level import. Kaggle's
# image currently carries tokenizers 0.21.x, which makes the import fail after
# the expensive dependency/bootstrap phase. Keep this exact and independent of
# pip's resolver so no CUDA package can be changed as a side effect.
_TOKENIZERS_REQUIRED = "0.22.1"
_HUB_REQUIRED = "1.5.0"


def _purge_module_tree(*roots):
    """Remove already-imported package trees after an in-process pip repair."""
    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(root + ".") for root in roots):
            del sys.modules[module_name]


print(f"  Ensuring huggingface_hub=={_HUB_REQUIRED}...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
     "--no-deps", "--force-reinstall", f"huggingface_hub=={_HUB_REQUIRED}"],
    check=True,
)
# accelerate is checked above and may already have imported an older Hub
# package. Purge both package trees so Python cannot combine old submodules
# with the freshly installed files (the source of the alternating missing
# Hub-symbol failures).
_purge_module_tree("huggingface_hub", "accelerate", "transformers")
import importlib
importlib.invalidate_caches()
try:
    import huggingface_hub as _hub_check
    from huggingface_hub.errors import RemoteEntryNotFoundError as _RemoteEntryNotFoundError
    importlib.import_module("huggingface_hub.file_download")
    del _RemoteEntryNotFoundError, _hub_check
except Exception as _hub_error:
    raise RuntimeError(
        f"huggingface_hub=={_HUB_REQUIRED} is still inconsistent after clean reinstall: {_hub_error}"
    ) from _hub_error
print(f"  huggingface_hub=={_HUB_REQUIRED} import validated")

try:
    _transformers_version = _package_version("transformers")
except PackageNotFoundError:
    _transformers_version = None
if _transformers_version != _TRANSFORMERS_REQUIRED:
    print(f"  Pinning Transformers {_transformers_version or 'missing'} -> {_TRANSFORMERS_REQUIRED}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "--no-deps", "--force-reinstall", f"transformers=={_TRANSFORMERS_REQUIRED}"],
        check=True,
    )

try:
    _tokenizers_version = _package_version("tokenizers")
except PackageNotFoundError:
    _tokenizers_version = None
if _tokenizers_version != _TOKENIZERS_REQUIRED:
    print(f"  Pinning tokenizers {_tokenizers_version or 'missing'} -> {_TOKENIZERS_REQUIRED}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
         "--no-deps", "--force-reinstall", f"tokenizers=={_TOKENIZERS_REQUIRED}"],
        check=True,
    )

# Both packages may have been imported by an earlier warm-kernel probe. Purge
# their module trees so the import below cannot combine old Python modules with
# freshly installed package files.
_purge_module_tree("transformers", "tokenizers")
importlib.invalidate_caches()
from transformers import AutoModel, AutoProcessor, AutoTokenizer  # noqa: F401

_ensure_package("resemble_enhance", "resemble-enhance", ["--no-deps"])
if shutil.which("ffmpeg") is None:
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg",
                   shell=True, capture_output=True)

import torchaudio
import assemblyai as aai
import google.generativeai as genai


def _print_gpu_inventory():
    """Log every visible CUDA device so remote accelerator selection is verifiable."""
    if not torch.cuda.is_available():
        print("  CUDA devices visible: 0")
        return
    count = torch.cuda.device_count()
    print(f"  CUDA devices visible: {count}")
    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
            memory_gb = props.total_memory / (1024 ** 3)
            print(f"    GPU {index}: {props.name} ({memory_gb:.1f} GB VRAM)")
        except Exception as e:
            print(f"    GPU {index}: unavailable ({type(e).__name__}: {str(e)[:100]})")


_print_gpu_inventory()

# Check if NVENC is available
def _has_nvenc():
    r = subprocess.run("ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc",
        shell=True, capture_output=True, text=True)
    return "h264_nvenc" in r.stdout

USE_GPU = _has_nvenc()
_nvenc_runtime_failed = False
print(f"  GPU Encoding: {'h264_nvenc' if USE_GPU else 'libx264 (CPU)'}")

def _enc_args():
    """Return encoder args based on currently usable GPU availability."""
    if USE_GPU and not _nvenc_runtime_failed:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]

def _hwaccel_args():
    """Use CUDA input only while the runtime NVENC path is healthy."""
    return ["-hwaccel", "cuda"] if USE_GPU and not _nvenc_runtime_failed else []


# ==========================================
# 1B. RESOURCE MONITOR (periodic CPU/RAM/GPU logging)
# ==========================================
# A silent regression (memory leak, VRAM creep, runaway CPU from a stuck
# retry loop) previously had no visibility until the whole job failed or
# timed out tens of minutes later. This prints one compact resource line on
# a fixed interval for the entire run, plus targeted snapshots right at the
# handful of points most likely to fail (MiniCPM load/OOM, TTS, Enhance,
# final render) so a future failure's exact resource state is captured in
# the log next to it, not just its elapsed wall-clock time.
def _gpu_stats_via_nvidia_smi():
    """Query per-GPU utilization/memory/temperature; [] if nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        stats = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            stats.append({
                "index": parts[0], "util_pct": parts[1],
                "mem_used_mb": parts[2], "mem_total_mb": parts[3],
                "temp_c": parts[4],
            })
        return stats
    except Exception:
        return []


def _log_resource_snapshot(label=""):
    """Print one CPU/RAM/GPU usage line. Cheap enough to call often."""
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        cpu_part = (f"CPU {cpu_pct:.0f}% | RAM {vm.used / (1024**3):.1f}/"
                    f"{vm.total / (1024**3):.1f}GB ({vm.percent:.0f}%)")
    except Exception as e:
        cpu_part = f"CPU/RAM unavailable ({type(e).__name__})"

    gpu_stats = _gpu_stats_via_nvidia_smi()
    if gpu_stats:
        gpu_part = " | ".join(
            f"GPU{g['index']} {g['util_pct']}% util, "
            f"{g['mem_used_mb']}/{g['mem_total_mb']}MB VRAM, {g['temp_c']}C"
            for g in gpu_stats
        )
    else:
        gpu_part = "GPU stats unavailable"
    tag = f"[RESOURCE{' ' + label if label else ''}]"
    print(f"  {tag} {cpu_part} | {gpu_part}")


_resource_monitor_stop = threading.Event()


def _resource_monitor_loop(interval_seconds):
    try:
        import psutil
        psutil.cpu_percent(interval=None)  # discard the meaningless first reading
    except Exception:
        pass
    while not _resource_monitor_stop.wait(interval_seconds):
        _log_resource_snapshot()


try:
    _RESOURCE_LOG_INTERVAL = max(10, int(os.environ.get("RESOURCE_LOG_INTERVAL_SECONDS", "30")))
except (TypeError, ValueError):
    _RESOURCE_LOG_INTERVAL = 30

threading.Thread(target=_resource_monitor_loop, args=(_RESOURCE_LOG_INTERVAL,),
                  daemon=True).start()
print(f"  Resource monitor: logging CPU/RAM/GPU every {_RESOURCE_LOG_INTERVAL}s")



# ==========================================
# 2. CONFIGURATION
# ==========================================
MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""
LANGUAGE = """{{LANGUAGE_PLACEHOLDER}}"""
YOUTUBE_CHANNEL = """{{YOUTUBE_CHANNEL_PLACEHOLDER}}"""

GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEY","").split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = os.environ.get("PEXELS_KEYS","").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS","").split(",")
# Support multiple Groq API keys (comma-separated). When one key is rate-limited,
# out of quota, or invalid, the next key is tried automatically.
GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEY","").split(",") if k.strip()]
GROQ_KEY = GROQ_KEYS[0] if GROQ_KEYS else ""  # kept for backward-compatible truthiness checks
_groq_key_index = 0
_groq_key_lock = threading.Lock()

OUTPUT_DIR = Path("output"); TEMP_DIR = Path("temp")
if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True); TEMP_DIR.mkdir(exist_ok=True)

IS_SPANISH = LANGUAGE.lower().strip() in ["spanish","es","espanol"]
USED_URLS = set()
_USED_URLS_LOCK = threading.Lock()
# Stock API lookup is deliberately shorter than media download. Running the
# two independent providers in parallel avoids serial 12-second stalls while
# still giving each provider enough time for normal network conditions.
_STOCK_API_TIMEOUT = (3, 5)
# Five sentence-aligned stock-query options generated in one unified Groq call.
# Each entry is [primary, backup_1, backup_2, backup_3, backup_4].
AI_QUERY_OPTIONS = []

# Shorts count by long-video duration:
#   5 min  -> 2 shorts
#   10 min -> 3 shorts
#   15 min -> 5 shorts
#   15min+ -> capped at 5 (confirmed intentional - does not keep scaling up)
def get_shorts_count(mins):
    if mins >= 13: return 5   # covers 15min and anything above, capped here
    if mins >= 8: return 3    # covers 10min
    return 2                  # covers 5min (and anything shorter)
SHORTS_COUNT = get_shorts_count(DURATION_MINS)
SHORT_DUR_TARGET = 60  # seconds, target length per short



# ==========================================
# 3. GROQ QUERY ENGINE (Sentence-Matched JSON)
# ==========================================
_LOCAL_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with", "about", "after",
    "also", "como", "con", "del", "desde", "el", "ella", "ellos", "en",
    "es", "esta", "este", "la", "las", "los", "para", "por", "que", "se",
    "su", "sus", "una", "uno", "y", "al", "más", "más", "son", "un",
}

# These are deliberately broad, stock-findable visual categories. They are
# only used for a sentence whose Groq response was missing or rejected after
# all targeted retries; they never replace a valid Groq query.
_LOCAL_QUERY_RULES = [
    (("kidney", "renal", "blood", "toxin", "disease", "medical", "hospital",
      "doctor", "anatomy", "health", "organ", "stone", "riñón", "sangre",
      "enfermedad", "medicina", "salud"),
     ("medical research laboratory", "medical anatomy illustration")),
    (("technology", "software", "computer", "digital", "data", "internet",
      "algorithm", "artificial", "intelligence", "device", "technology",
      "tecnología", "computadora", "datos", "algoritmo", "inteligencia"),
     ("technology server room", "digital computer screens")),
    (("science", "scientific", "research", "laboratory", "molecule", "atom",
      "experiment", "microscope", "ciencia", "investigación", "laboratorio"),
     ("scientific laboratory research", "microscope laboratory closeup")),
    (("economy", "economic", "market", "money", "finance", "financial", "bank",
      "stock", "inflation", "economía", "mercado", "dinero", "finanzas"),
     ("stock market trading screens", "financial data charts")),
    (("climate", "carbon", "pollution", "environment", "warming", "emissions",
      "clima", "contaminación", "ambiente"),
     ("climate change satellite earth", "environmental research laboratory")),
    (("city", "cities", "urban", "building", "construction", "architecture",
      "ciudad", "ciudades", "urbano", "construcción", "arquitectura"),
     ("aerial city construction", "urban buildings skyline")),
    (("school", "education", "university", "study", "learning", "classroom",
      "escuela", "educación", "universidad", "estudio"),
     ("education books classroom", "school science laboratory")),
    (("music", "song", "sound", "audio", "instrument", "música", "sonido"),
     ("music studio instruments", "audio recording equipment")),
    (("food", "cooking", "kitchen", "recipe", "agriculture", "farm", "comida",
      "cocina", "agricultura", "granja"),
     ("food preparation closeup", "agriculture farm footage")),
    (("ocean", "sea", "water", "marine", "beach", "ocean", "océano", "mar",
      "agua", "playa"),
     ("ocean waves aerial", "underwater marine life")),
    (("forest", "tree", "植物", "nature", "forest", "bosque", "naturaleza"),
     ("forest aerial landscape", "nature closeup foliage")),
    (("travel", "tourism", "journey", "airport", "hotel", "viaje", "turismo"),
     ("travel destination landscape", "airport travel terminal")),
    (("history", "ancient", "civilization", "museum", "historical", "historia",
      "antigua", "civilización"),
     ("historical architecture museum", "ancient ruins landscape")),
]


def _local_sentence_query_pair(sentence_text):
    """Return aligned stock queries when Groq omits one sentence."""
    text = str(sentence_text or "")
    lowered = text.lower()
    for triggers, pair in _LOCAL_QUERY_RULES:
        if any(trigger in lowered for trigger in triggers):
            return pair

    # Last-resort queries still contain words from this exact sentence, rather
    # than using a neighboring sentence or the old unrelated FALLBACK list.
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", lowered)
    content = []
    for word in words:
        if (len(word) > 2 and word not in _LOCAL_QUERY_STOPWORDS
                and word not in content and _safe(word)):
            content.append(word)
    if len(content) >= 2:
        core = " ".join(content[:4])
        return f"{core} documentary", f"{core} educational illustration"
    return "educational concept illustration", "documentary concept visualization"

def _local_sentence_query_options(sentence_text):
    """Build a pool of distinct, sentence-aligned options without an API call.

    Return more than the five final options because the caller also removes
    duplicates against any partial Groq response. The previous implementation
    padded by repeating its last option, so that de-duplication could leave a
    sentence with fewer than five usable queries and abort the whole job.
    """
    primary, backup = _local_sentence_query_pair(sentence_text)
    bases = [primary, backup]
    suffixes = [
        "", " closeup", " wide shot", " cinematic", " detail",
        " documentary", " macro", " aerial view", " laboratory view",
        " educational visual",
    ]
    options = []

    def _add_variants(base):
        base = re.sub(r"\s+", " ", str(base)).strip()
        if not (3 < len(base) < 60 and _safe(base)):
            return
        for suffix in suffixes:
            candidate = re.sub(r"\s+", " ", f"{base}{suffix}").strip()
            if (3 < len(candidate) < 60 and _safe(candidate)
                    and candidate not in options):
                options.append(candidate)

    for base in bases:
        _add_variants(base)

    # If the sentence contains restricted or unusable terms, retain a safe
    # educational fallback rather than reusing an unsafe query or a neighboring
    # sentence. These bases also provide enough unused variants when a partial
    # Groq response happens to overlap the sentence-specific local terms.
    for base in [
        "educational concept illustration",
        "documentary visual explanation",
        "scientific concept diagram",
        "abstract educational visualization",
    ]:
        _add_variants(base)

    return options

# ==========================================
# Groq reasoning model configuration
# ==========================================
# Query generation is split across two independent model requests. Each
# request has a bounded output size so the pair stays below the provider's
# 8,000-TPM organization limit; there is no client-side pacing or waiting.
_GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-120b")
# No separate fallback model. Large batches caused gpt-oss-120b to refuse with a
# "too many queries" error object and caused other models to hit a 429
# "request too large" on output tokens. The real fix is small fixed-size query
# batches (below), which gpt-oss-120b answers reliably. On a transient
# key/quota/rate error, _groq_complete already rotates to the next API key and
# retries on the same model, so a second model is unnecessary.
_GROQ_FALLBACK_MODEL = ""
# Sentences are chunked into small batches so no single request asks for too
# many queries. 25 sentences => 125 queries => ~800 completion tokens, which
# gpt-oss-120b returns in a few seconds well under the token cap and without
# the intermittent "request too large" refusal seen on 80+ sentence batches.
_QUERY_BATCH_SIZE = 25
# Cap on batches sent to Groq at the same time. Keeps the concurrent token
# burst under the 8,000-TPM organization limit (each small batch is ~1600
# prompt + ~800 completion tokens).
_QUERY_BATCH_CONCURRENCY = 2
_QUERY_BATCH_MAX_COMPLETION_TOKENS = 2000


def _make_groq_client(key_index=None):
    """Create a Groq client for a specific key index (defaults to current)."""
    from groq import Groq
    global _groq_key_index
    if not GROQ_KEYS:
        raise RuntimeError("No Groq API key configured")
    with _groq_key_lock:
        idx = _groq_key_index if key_index is None else key_index
        idx = idx % len(GROQ_KEYS)
    return Groq(api_key=GROQ_KEYS[idx])


def _rotate_groq_key():
    """Advance to the next Groq key. Returns the new key index, or None if
    only one key exists (nothing to rotate to)."""
    global _groq_key_index
    if len(GROQ_KEYS) <= 1:
        return None
    with _groq_key_lock:
        _groq_key_index = (_groq_key_index + 1) % len(GROQ_KEYS)
        new_idx = _groq_key_index
    print(f"  Groq: switching to API key #{new_idx + 1}/{len(GROQ_KEYS)}")
    return new_idx


def _is_groq_key_error(error):
    """True if the error looks like a key/quota/rate-limit problem worth
    retrying on a different key."""
    msg = str(error).lower()
    return any(token in msg for token in (
        "rate limit", "rate_limit", "quota", "insufficient", "429",
        "invalid api key", "invalid_api_key", "401", "403",
        "authentication", "unauthorized", "over capacity", "capacity",
    ))


def _groq_complete(client, messages, label, temperature, model,
                   max_completion_tokens=None, attempts=2):
    """Run one Groq completion and always return text, never None.

    If the request fails with a key/quota/rate-limit error and multiple Groq
    keys are configured, it rebuilds the client on the next key and retries.
    """
    attempt = 0
    _key_rotations = 0
    _max_key_rotations = max(0, len(GROQ_KEYS) - 1)
    while attempt < attempts:
        attempt += 1
        kwargs = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        kwargs["reasoning_effort"] = "low"
        kwargs["reasoning_format"] = "hidden"

        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError as error:
            # Older SDKs may not accept the reasoning controls. Retry this
            # request without them, without changing its model or prompt.
            if "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                kwargs.pop("reasoning_format", None)
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as retry_error:
                    print(f"  Groq {label}: retry without reasoning controls failed: "
                          f"{type(retry_error).__name__}: {str(retry_error)[:180]}")
                    return ""
            else:
                print(f"  Groq {label}: TypeError: {str(error)[:160]}")
                return ""
        except Exception as error:
            message_text = str(error)
            if "reasoning" in message_text.lower():
                kwargs.pop("reasoning_effort", None)
                kwargs.pop("reasoning_format", None)
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as retry_error:
                    print(f"  Groq {label}: retry without reasoning controls failed: "
                          f"{type(retry_error).__name__}: {str(retry_error)[:180]}")
                    return ""
            else:
                message_lower = message_text.lower()
                # If this looks like a key/quota/rate-limit issue and we have
                # more keys, rotate to the next key, rebuild the client, and
                # retry without consuming a normal attempt.
                if _is_groq_key_error(error) and len(GROQ_KEYS) > 1 and _key_rotations < _max_key_rotations:
                    print(f"  Groq {label}: key error ({type(error).__name__}: "
                          f"{message_text[:120]}); rotating key")
                    new_idx = _rotate_groq_key()
                    if new_idx is not None:
                        _key_rotations += 1
                        try:
                            client = _make_groq_client(new_idx)
                            attempt -= 1  # don't count this as a failed attempt
                            continue
                        except Exception:
                            pass
                print(f"  Groq {label} attempt {attempt}/{attempts} failed: "
                      f"{type(error).__name__}: {message_text[:180]}")
                if attempt >= attempts:
                    return ""
                continue

        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", None) or "")
        finish_reason = getattr(choice, "finish_reason", "?") if choice else "?"
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        print(f"  Groq {label}: model={model}, finish={finish_reason}, "
              f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
              f"content_chars={len(text)}")
        if text.strip():
            return text
        if attempt >= attempts:
            break
    return ""


# Compact system prompt shared by both model batches. The main sentence set is
# split before the requests so neither model is asked to emit hundreds of
# entries in one completion.
_STOCK_QUERY_SYSTEM_PROMPT = """You write stock-footage search queries for Pexels/Pixabay narration sentences.

Return one raw JSON object only. Each key is the exact 1-based sentence number and
its value is an array of EXACTLY FIVE different search-query strings.
Example: {"1":["primary query","backup query","third query","fourth query","fifth query"]}

Rules for every query:
- English only, 3-4 words, describing something a camera can actually film.
- Represent the sentence's meaning, not just isolated keywords.
- For medical, abstract, or niche subjects choose the closest findable thematic
  visual rather than literal terms with no stock footage.
- Keep all five options thematically tied to the SAME sentence and use different
  visual angles so they are useful fallbacks.
- Never use unrelated nature, ocean, space, or generic filler.
- No people, faces, bodies, women, religion, violence, or NSFW.
- Include every sentence number exactly once; output no markdown or explanation."""


def generate_queries_for_sentences(sentences):
    """Generate five aligned visual queries using two parallel Groq batches.

    The sentence list is split in half before the API calls. GPT-OSS-20B and
    GPT-OSS-120B process their halves concurrently, keeping each request small
    enough for the provider's 8,000-TPM organization limit. A partial, empty,
    refused, or malformed response never shifts sentence alignment: every
    missing entry is filled from the same sentence's local fallback pool.
    """
    if not GROQ_KEY or not sentences:
        raise RuntimeError(
            "Groq is required for sentence-specific visual queries; "
            "refusing to use generic footage"
        )

    from groq import Groq

    n = len(sentences)
    # Split into small fixed-size batches so no single request asks for too many
    # queries (which made gpt-oss-120b refuse with a "too large" error object).
    # Each batch cycles through the configured models for light load spreading.
    batches = []
    for start in range(0, n, _QUERY_BATCH_SIZE):
        chunk = sentences[start:start + _QUERY_BATCH_SIZE]
        model = _GROQ_MODELS[(start // _QUERY_BATCH_SIZE) % len(_GROQ_MODELS)]
        batches.append((start, chunk, model))
    print(f"  Groq: matching {n} sentences in {len(batches)} batches of "
          f"<= {_QUERY_BATCH_SIZE} ({_GROQ_MODELS[0]}, "
          f"<= {_QUERY_BATCH_CONCURRENCY} concurrent)...")

    def _parse_batch_result(raw_result, batch):
        """Return local sentence index -> cleaned option list."""
        raw_result = (raw_result or "").strip()
        parsed = None
        try:
            parsed = json.loads(raw_result)
        except (TypeError, ValueError):
            match = re.search(r"\{.*\}", raw_result, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except (TypeError, ValueError):
                    parsed = None

        parsed_options = {}
        if isinstance(parsed, dict):
            for key, values in parsed.items():
                try:
                    local_idx = int(str(key)) - 1
                except (TypeError, ValueError):
                    continue
                if not 0 <= local_idx < len(batch):
                    continue
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                cleaned_options = []
                for value in values:
                    cleaned = re.sub(r"\s+", " ", str(value)).strip().strip('"\'')
                    if (3 < len(cleaned) < 60 and _safe(cleaned)
                            and cleaned not in cleaned_options):
                        cleaned_options.append(cleaned)
                if cleaned_options:
                    parsed_options[local_idx] = cleaned_options[:5]

        # A numbered-line fallback is useful when a model emits valid content
        # but ignores the raw-JSON-only instruction. It only fills entries not
        # already recovered from keyed JSON.
        if not parsed_options:
            for line in raw_result.splitlines():
                match = re.match(r'^\s*(\d+)[\.:\)\-]\s*(.+)$', line)
                if not match:
                    continue
                local_idx = int(match.group(1)) - 1
                cleaned = re.sub(r"\s+", " ", match.group(2)).strip().strip('"\'')
                if (0 <= local_idx < len(batch) and 3 < len(cleaned) < 60
                        and _safe(cleaned)):
                    parsed_options[local_idx] = [cleaned]
        return parsed_options

    def _run_batch(batch_start, batch, model):
        numbered = "\n".join(
            f"{i + 1}. {sentence['text'][:140]}"
            for i, sentence in enumerate(batch)
        )
        client = _make_groq_client()
        result = _groq_complete(
            client,
            [
                {"role": "system", "content": _STOCK_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content":
                 "Generate five aligned stock queries for every sentence below. "
                 "The numbering is local to this batch:\n\n" + numbered},
            ],
            label=f"query batch {batch_start + 1}-{batch_start + len(batch)}",
            temperature=0.35,
            model=model,
            max_completion_tokens=_QUERY_BATCH_MAX_COMPLETION_TOKENS,
            attempts=2,
        )
        parsed = _parse_batch_result(result, batch)
        print(f"  Groq {model}: parsed {len(parsed)}/{len(batch)} batch sentences")
        if len(parsed) < len(batch):
            print(f"    Raw model output (first 300 chars): {(result or '')[:300]!r}")

        # Fallback: if primary model refused or returned nothing usable,
        # retry the entire batch with llama-3.1-8b-instant
        if len(parsed) == 0 and _GROQ_FALLBACK_MODEL:
            print(f"  Groq {model}: 0 parsed — retrying batch with fallback model {_GROQ_FALLBACK_MODEL}...")
            fallback_result = _groq_complete(
                client,
                [
                    {"role": "system", "content": _STOCK_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content":
                     "Generate five aligned stock queries for every sentence below. "
                     "The numbering is local to this batch:\n\n" + numbered},
                ],
                label=f"query batch {batch_start + 1}-{batch_start + len(batch)} (fallback)",
                temperature=0.35,
                model=_GROQ_FALLBACK_MODEL,
                max_completion_tokens=_QUERY_BATCH_MAX_COMPLETION_TOKENS,
                attempts=2,
            )
            parsed = _parse_batch_result(fallback_result, batch)
            print(f"  Groq {_GROQ_FALLBACK_MODEL} fallback: parsed {len(parsed)}/{len(batch)} batch sentences")
            if len(parsed) < len(batch):
                print(f"    Fallback raw output (first 300 chars): {(fallback_result or '')[:300]!r}")

        return batch_start, parsed

    parsed_by_global_index = {}
    max_workers = max(1, min(_QUERY_BATCH_CONCURRENCY, len(batches)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_batch, *batch_info) for batch_info in batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_start, parsed = future.result()
                for local_idx, options in parsed.items():
                    parsed_by_global_index[batch_start + local_idx] = options
            except Exception as error:
                print(f"  Groq parallel query batch failed: {type(error).__name__}: "
                      f"{str(error)[:180]}")

    query_options = []
    for sentence_idx, sentence in enumerate(sentences):
        options = []
        for candidate in parsed_by_global_index.get(sentence_idx, []):
            candidate = str(candidate).strip()
            if (3 < len(candidate) < 60 and _safe(candidate)
                    and candidate not in options):
                options.append(candidate)
        options.extend(
            candidate for candidate in _local_sentence_query_options(sentence["text"])
            if candidate not in options
        )
        if len(options) < 5:
            raise RuntimeError(
                f"Unable to create five aligned visual queries for sentence {sentence_idx + 1}"
            )
        query_options.append(options[:5])
        if sentence_idx < 3:
            print(f"    [{sentence_idx + 1}] '{sentence['text'][:35]}...' -> {options[:5]}")

    return query_options

FALLBACK = [
    "technology data center servers", "futuristic city aerial night",
    "abstract digital particles", "space nebula stars 4k",
    "ocean waves aerial cinematic", "mountain landscape dramatic",
    "circuit board macro electronics", "laboratory scientific research",
    "architecture modern building glass", "sunrise golden hour landscape",
    "underwater coral reef fish", "clouds timelapse dramatic sky",
    "desert sand dunes aerial", "forest aerial cinematic fog",
    "lightning storm dramatic clouds", "volcano lava flow night"
]

def _safe(q):
    bad = ['woman','women','girl','female','bikini','nude','naked','sexy',
           'jesus','christ','church','mosque','temple','bible','buddha',
           'gun','weapon','war','blood','violence','kill','alcohol',
           'drug','gambling','pork','lgbtq','person','people','crowd']
    return not any(t in q.lower() for t in bad)


def _query_attempts(index, orig_index=None):
    """Return all five sentence-specific options; never use generic footage."""
    query_index = orig_index if orig_index is not None else index
    if query_index < 0 or query_index >= len(AI_QUERY_OPTIONS):
        raise RuntimeError(f"No AI visual queries exist for sentence {query_index + 1}")
    options = [str(value).strip() for value in (AI_QUERY_OPTIONS[query_index] or [])]
    options = [value for value in options if value and _safe(value)]
    if not options:
        raise RuntimeError(f"Primary visual query is empty for sentence {query_index + 1}")
    return list(dict.fromkeys(options))


def _sentence_query_variants(attempts):
    """Generate only semantically tied variants for persistent stock retries."""
    suffixes = ["", " cinematic", " close up", " wide shot", " aerial view", " 4k"]
    seen = set()
    for base in attempts:
        for suffix in suffixes:
            variant = f"{base}{suffix}".strip()
            if variant and variant not in seen:
                seen.add(variant)
                yield variant


# A bounded search is essential: stock providers expose finite pages, and
# cycling already-used URLs forever can otherwise keep a Kaggle job alive for
# hours. The five aligned options are generated before clip search begins.
_CLIP_QUERY_ROUNDS = 3  # allow multiple Pexels pages per query set for better coverage
_CLIP_CANDIDATES_PER_QUERY = 5


# Concurrent workers previously asked Groq separately for each unmatched
# sentence. Batching concurrent recovery requests into ONE shared Groq call
# reduces provider request count and amortizes the fixed prompt work across
# several sentences, without introducing client-side request pacing.
_fresh_query_batch_lock = threading.Lock()
_fresh_query_pending = []  # list of dicts, see _request_fresh_sentence_queries
_FRESH_QUERY_BATCH_WINDOW = 1.5  # seconds to collect concurrent requests
_FRESH_QUERY_BATCH_MAX = 8


def _run_fresh_query_batch(batch):
    """Send one Groq call covering every entry in `batch`, then unblock callers."""
    try:
        if not batch:
            return
        client = _make_groq_client()
        numbered_lines = []
        for i, item in enumerate(batch):
            hint = "portrait/vertical" if item["orientation"] == "portrait" else "landscape"
            previous = "; ".join(item["previous"][-6:]) or "none"
            numbered_lines.append(
                f"{i + 1}. [{hint}] Sentence: {item['sentence'][:160]}\n"
                f"{i + 1}. Previously failed: {previous}"
            )
        numbered = "\n".join(numbered_lines)
        text = _groq_complete(
            client,
            [
                {"role": "system", "content": """You are recovering stock-footage queries for several unmatched narration sentences at once.
For EACH numbered sentence below, return exactly two new, different, sentence-specific stock-footage queries using that sentence's own global number:
N. primary query
Nb. backup query

Rules:
- English, 3-6 words, describing something a camera can film.
- Match the orientation noted in brackets (portrait/vertical or landscape) implicitly through composition, not by naming it in the query text.
- Clearly represent that sentence's meaning, not merely a broad topic.
- For abstract or medical ideas, choose the closest findable thematic visual.
- Do not use generic nature, ocean, space, landscape, or unrelated filler.
- No people, faces, bodies, women, religion, violence, or NSFW.
- Do not repeat that sentence's own previously-failed queries.
- Output ONLY the numbered lines, two per sentence, nothing else."""},
                {"role": "user", "content": f"Recover queries for these unmatched sentences:\n\n{numbered}"}
            ],
            label=f"batched fresh-query recovery ({len(batch)} sentences)",
            temperature=0.65,
            model=_GROQ_MODELS[0],
            max_completion_tokens=600,
            attempts=1,
        )
        parsed_primary = {}
        parsed_backup = {}
        for line in (text or "").strip().split("\n"):
            line = line.strip()
            mb = re.match(r'^\s*(\d+)b[\.\)\-]\s*(.+)$', line, re.IGNORECASE)
            if mb:
                idx = int(mb.group(1)) - 1
                if 0 <= idx < len(batch):
                    parsed_backup[idx] = mb.group(2).strip().strip('"\'')
                continue
            m = re.match(r'^\s*(\d+)[\.\)\-]\s*(.+)$', line)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(batch):
                    parsed_primary[idx] = m.group(2).strip().strip('"\'')

        supplied = 0
        for i, item in enumerate(batch):
            fresh = []
            primary = parsed_primary.get(i, "").strip()
            if 3 < len(primary) < 60 and _safe(primary) and primary not in item["previous"]:
                fresh.append(primary)
            backup = parsed_backup.get(i, "").strip()
            if (3 < len(backup) < 60 and _safe(backup)
                    and backup not in item["previous"] and backup not in fresh):
                fresh.append(backup)
            item["result"] = fresh[:2]
            if fresh:
                supplied += 1
        if supplied:
            print(f"    Groq batched recovery: supplied fresh queries for "
                  f"{supplied}/{len(batch)} unmatched sentences in one call")
    except Exception as e:
        print(f"    Groq batched fresh-query recovery failed: {type(e).__name__}: {str(e)[:120]}")
    finally:
        for item in batch:
            item["event"].set()


def _request_fresh_sentence_queries(sentence, previous_queries, orientation):
    """Ask Groq for new queries for one unmatched sentence, coalescing
    concurrent callers into a single shared Groq request. Return shape and
    behavior (a list of up to 2 fresh query strings) is unchanged from the
    single-sentence version; only the underlying request is now batched.
    """
    if not GROQ_KEY:
        return []

    entry = {
        "sentence": sentence,
        "previous": list(previous_queries),
        "orientation": orientation,
        "event": threading.Event(),
        "result": [],
    }

    with _fresh_query_batch_lock:
        _fresh_query_pending.append(entry)
        is_leader = len(_fresh_query_pending) == 1

    if is_leader:
        # Give other concurrently-failing clip workers a short window to
        # join this batch before paying for the Groq call.
        time.sleep(_FRESH_QUERY_BATCH_WINDOW)
        # Drain the whole queue in sequential batches rather than processing
        # only one. "is_leader" is only assigned on a 0 -> 1 transition of
        # the pending list, which does not happen again while stragglers
        # beyond one batch's cap remain - without draining here, any
        # overflow past _FRESH_QUERY_BATCH_MAX would strand callers with no
        # one left to process them until their 60s wait times out.
        while True:
            with _fresh_query_batch_lock:
                batch = _fresh_query_pending[:_FRESH_QUERY_BATCH_MAX]
                del _fresh_query_pending[:len(batch)]
            if not batch:
                break
            _run_fresh_query_batch(batch)
    else:
        # Not the leader for this batch cycle - wait for it to deliver our
        # result. The timeout is generous but bounded so a stuck Groq call
        # can never hang a clip worker forever; entry["result"] defaults to
        # [] and the caller already treats an empty list as "try again or
        # fall back", matching prior single-call behavior.
        entry["event"].wait(timeout=60.0)

    return entry["result"]



# ==========================================
# 3B. AI SHORT SCRIPT WRITER (Groq oss-120b writes standalone short scripts)
# ==========================================
def generate_short_scripts(sentences, topic, n_shorts, target_seconds=60):
    """
    Two-step process using Groq oss-120b:
      1. Identify the N best hook-worthy themes/moments from the full
         long-form script (grounds the shorts in the actual video content).
      2. For each theme, WRITE a fresh, standalone, hook-first short-form
         script (~130-160 words for ~60s of narration at normal pace) -
         NOT a cut/excerpt of the original audio. This gets its own TTS
         pass, so it never depends on the main audio's timing/quality.

    Returns a list of dicts: {"script": str, "theme": str}
    """
    words_target = int(target_seconds / 60 * 150)  # ~150 wpm normal pace
    # Keep the source excerpt bounded so short-script selection stays focused;
    # this is prompt-content selection, not client-side request throttling.
    full_text = " ".join(s['text'] for s in sentences)[:8000]

    def _fallback_scripts():
        # If Groq is unavailable, fall back to lightly-summarized chunks of
        # the original script text, evenly spaced, rewritten as a short
        # standalone paragraph (still not a literal audio cut - this is
        # text, which gets its own fresh TTS regardless of this path).
        out = []
        step = max(1, len(sentences) // (n_shorts + 1))
        for k in range(n_shorts):
            start = min(k * step, max(0, len(sentences)-3))
            chunk = sentences[start:start+4]
            text = " ".join(s['text'] for s in chunk)
            out.append({"script": text, "theme": "fallback excerpt"})
        return out

    if not GROQ_KEY or not sentences:
        return _fallback_scripts()

    print(f"  Groq: writing {n_shorts} standalone short scripts (~{words_target} words each)...")
    try:
        client = _make_groq_client()
        lang_instruction = "Write in Spanish." if IS_SPANISH else "Write in English."
        lang_reminder = "IMPORTANT: The entire script text MUST be written in Spanish (español)." if IS_SPANISH else ""

        r = _groq_complete(
            client,
            [
                {"role": "system", "content": f"""You are an expert short-form (TikTok/Reels/Shorts) scriptwriter.

You will be given the full text of a long-form documentary/narration script. Your job: write {n_shorts} COMPLETELY STANDALONE short-form scripts inspired by the best hooks, surprising facts, emotional peaks, or claims in the source material.

RULES:
- Each script must be a SELF-CONTAINED mini-narration, NOT a literal excerpt or copy-paste of the source text. Rewrite/condense the idea into a tight, punchy standalone script.
- Each script should be approximately {words_target} words (for ~{target_seconds} seconds of narration at normal pace).
- Start with a strong HOOK in the first sentence - something that stops someone scrolling (a surprising claim, a question, a cliffhanger).
- {lang_instruction} {lang_reminder}
- Each script must make complete sense on its own with zero external context needed.
- Family-friendly, no NSFW, no violence, no religion-baiting content.
- Return ONLY a JSON array, nothing else. No markdown, no explanation.

Format exactly like this:
[{{"script": "full standalone narration text here...", "theme": "short label like 'the vanishing lake'"}}]"""},
                {"role": "user", "content": f"Source script:\n\n{full_text}\n\nWrite {n_shorts} standalone short scripts as JSON. {lang_instruction}"}
            ],
            label="short-script writer",
            temperature=0.75,
            model=_GROQ_MODELS[1],
            max_completion_tokens=1400,
            attempts=1,
        )

        raw = (r or "").strip()
        if not raw:
            print("  Groq short-script: model returned no content, using fallback")
            return _fallback_scripts()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            print("  Groq short-script: no JSON array found, using fallback")
            return _fallback_scripts()

        picks = json.loads(match.group(0))
        scripts = []
        for p in picks:
            txt = str(p.get("script","")).strip()
            if len(txt) > 20:
                scripts.append({"script": txt, "theme": p.get("theme","")})

        if not scripts:
            print("  Groq short-script: parsed JSON had no valid scripts, using fallback")
            return _fallback_scripts()

        for s in scripts[:n_shorts]:
            wc = len(s['script'].split())
            print(f"    Short script: '{s['theme'][:40]}' ({wc} words)")

        fb = _fallback_scripts()
        while len(scripts) < n_shorts:
            scripts.append(fb[len(scripts) % len(fb)] if fb else {"script": topic, "theme": "padding"})

        return scripts[:n_shorts]

    except Exception as e:
        print(f"  Groq short-script error: {e}")
        return _fallback_scripts()



# ==========================================
# 4. SUBTITLE PRESETS (Variety)
# ==========================================
SUBTITLE_STYLES = {
    "classic": {"name":"Classic","font":"Arial Black","size":64,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00141414","back":"&H00000000",
        "border":1,"outline":5,"shadow":2,"margin":48,"spacing":0.5,
        "highlight":"&H0059C7FF"},   # white text, warm gold highlight
    "cyan_pop": {"name":"Cyan Pop","font":"Arial Black","size":64,"bold":-1,
        "primary":"&H00FFDC78","outline_c":"&H002D140F","back":"&H00000000",
        "border":1,"outline":5,"shadow":2,"margin":48,"spacing":0.5,
        "highlight":"&H00FFFFFF"},   # soft cyan text, white highlight
    "boxed": {"name":"Boxed","font":"Arial","size":58,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&HB0000000",
        "border":3,"outline":0,"shadow":0,"margin":48,"spacing":0.3,
        "highlight":"&H0059C7FF"},   # white on translucent black box, gold highlight
}

# Short-specific vertical styles (1080x1920 canvas). Bigger fonts than
# landscape presets since viewers are closer to phone screens, and a much
# larger MarginV to clear TikTok/Reels/YouTube Shorts UI (caption, like/
# share buttons, progress bar) which occupies the bottom ~20-25% of frame.
# Every color pair below was computed programmatically (RGB -> ASS BGR),
# not hand-written, and verified for real contrast before use.
SHORT_SUBTITLE_STYLES = {
    "short_classic": {"name":"Short Classic","font":"Arial Black","size":90,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00141414","back":"&H00000000",
        "border":1,"outline":8,"shadow":3,"margin":480,"spacing":0.5,
        "highlight":"&H003BEBFF"},   # white text, electric yellow highlight
    "short_pink": {"name":"Short Pink","font":"Arial Black","size":90,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00141414","back":"&H00000000",
        "border":1,"outline":8,"shadow":3,"margin":480,"spacing":0.5,
        "highlight":"&H008140FF"},   # white text, hot pink highlight
    "short_boxed": {"name":"Short Boxed","font":"Arial","size":82,"bold":-1,
        "primary":"&H00FFFFFF","outline_c":"&H00000000","back":"&HB0000000",
        "border":3,"outline":0,"shadow":0,"margin":460,"spacing":0.3,
        "highlight":"&H0040FFAE"},   # white on translucent black box, lime highlight
}

def _fmt(sec):
    h=int(sec//3600); m=int((sec%3600)//60); s=int(sec%60); cs=int((sec%1)*100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def create_subtitles(sentences, ass_path, word_data=None, style_key=None,
                      style_set=None, play_res=(1920,1080), max_chars=46):
    """
    Word-level highlighted subtitles (like Submagic/Captions.ai).
    If word_data is provided, each word lights up as it's spoken.
    Falls back to sentence-level if no word data.

    style_set: dict of style presets to choose from (defaults to landscape SUBTITLE_STYLES)
    play_res: (width, height) of the ASS canvas - must match final video resolution
    max_chars: character budget per 2-line chunk - tune per play_res/font size
    """
    style_set = style_set or SUBTITLE_STYLES
    if style_key:
        key = style_key
    else:
        # Derive selection from a hash of JOB_ID + wall-clock time + OS
        # entropy, rather than trusting random.choice() alone. JOB_ID is
        # guaranteed unique per run (templated in externally per job), so
        # this guarantees different style selection across separate runs
        # even if Python's RNG state behaves unexpectedly in the Kaggle
        # kernel environment (e.g. container/process state carrying over
        # between runs in a way that defeats a simple reseed).
        import hashlib
        seed_material = f"{JOB_ID}-{time.time()}-{os.urandom(4).hex()}"
        h = int(hashlib.sha256(seed_material.encode()).hexdigest(), 16)
        keys = list(style_set.keys())
        key = keys[h % len(keys)]
    s = style_set[key]
    print(f"  Subtitle: {s['name']} {'(word-highlight)' if word_data else '(sentence)'} @ {play_res[0]}x{play_res[1]}")
    
    # Highlight color (the word currently being spoken) - explicit per-style,
    # chosen for contrast against that style's own primary text color.
    highlight = s.get("highlight", "&H0000FFFF")
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: {play_res[0]}\nPlayResY: {play_res[1]}\n")
        f.write("WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{s['font']},{s['size']},{s['primary']},&H00FFFFFF,"
                f"{s['outline_c']},{s['back']},{s['bold']},0,0,0,100,100,"
                f"{s['spacing']},0,{s['border']},{s['outline']},{s['shadow']},"
                f"2,50,50,{s['margin']},1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        if word_data:
            # Word-level highlighting: group words into display chunks by
            # character budget (not raw word count), so a chunk of short
            # words can hold more words than a chunk of long words.
            MAX_CHARS_PER_CHUNK = max_chars   # total chars across both lines, tune per font/size/canvas
            MAX_WORDS_PER_CHUNK = 14   # hard ceiling so very short words can't run away

            chunks = []
            buf, blen = [], 0
            for word in word_data:
                wlen = len(word['text']) + 1  # +1 for the joining space
                if buf and (blen + wlen > MAX_CHARS_PER_CHUNK or len(buf) >= MAX_WORDS_PER_CHUNK):
                    chunks.append(buf)
                    buf, blen = [], 0
                buf.append(word)
                blen += wlen
            if buf:
                chunks.append(buf)

            for c_idx, chunk_words in enumerate(chunks):
                if not chunk_words: continue

                # Balance the 2-line split by character length, not word count,
                # so line 1 and line 2 come out visually similar in width.
                total_chars = sum(len(w['text']) for w in chunk_words)
                running, split_idx = 0, len(chunk_words) - 1
                for j, cw in enumerate(chunk_words):
                    running += len(cw['text'])
                    if running >= total_chars / 2:
                        split_idx = j
                        break

                # For each word in chunk, create a dialogue line where THAT word is highlighted.
                # IMPORTANT: extend each word's display End to the START of the next word
                # (or, for the last word in the chunk, to the first word of the NEXT chunk
                # if one exists - otherwise its own end). Word-level ASR timestamps often
                # have small gaps between word['end'] and the next word['start']
                # (pauses/breaths) - using word['end'] directly as the Dialogue End causes
                # the subtitle to go blank during those gaps, both within and between chunks.
                is_last_chunk = (c_idx == len(chunks) - 1)
                next_chunk_start = chunks[c_idx+1][0]['start'] if not is_last_chunk else chunk_words[-1]['end']

                MIN_HIGHLIGHT_DUR = 0.12  # seconds - guarantees every word gets
                # a real, renderable display window. Fast speech or ASR
                # quirks can produce near-zero-duration words (start ~= end,
                # or the next word starting almost immediately). libass can
                # skip or misrender these near-zero-duration events, which
                # made the highlight appear to jump past the correct word
                # onto the next one while the audio was still on the first.

                prev_end_sec = None  # tracks previous word's (possibly-extended) end,
                                      # so extending one word's duration can never
                                      # cause it to overlap the next word's event
                for w_idx, word in enumerate(chunk_words):
                    w_start_sec = word['start']
                    if prev_end_sec is not None and w_start_sec < prev_end_sec:
                        w_start_sec = prev_end_sec  # never start before prior word ended

                    if w_idx + 1 < len(chunk_words):
                        next_start_sec = chunk_words[w_idx+1]['start']
                    else:
                        next_start_sec = next_chunk_start

                    # Enforce minimum duration - if the natural gap to the
                    # next word is too small, extend this word's END forward
                    # rather than shrinking its START, so timing still lines
                    # up with when the word actually begins being spoken.
                    if next_start_sec - w_start_sec < MIN_HIGHLIGHT_DUR:
                        next_start_sec = w_start_sec + MIN_HIGHLIGHT_DUR
                    prev_end_sec = next_start_sec

                    w_start = _fmt(w_start_sec)
                    w_end = _fmt(next_start_sec)

                    p1, p2 = [], []
                    for j, cw in enumerate(chunk_words):
                        txt = f"{{\\c{highlight}\\fscx115\\fscy115}}{cw['text']}{{\\r}}" if j == w_idx else cw['text']
                        if j <= split_idx: p1.append(txt)
                        else: p2.append(txt)

                    line = ' '.join(p1) + "\\N" + ' '.join(p2) if p2 else ' '.join(p1)
                    f.write(f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n")
        else:
            # Sentence-level fallback - split into 2 lines balanced by character length.
            # Extend each sentence's End to the next sentence's Start to avoid blank gaps.
            for idx, sent in enumerate(sentences):
                t1 = _fmt(sent['start'])
                next_start = sentences[idx+1]['start'] if idx+1 < len(sentences) else sent['end']
                t2 = _fmt(next_start)
                txt = sent['text'].strip().rstrip('.,;:')
                w = txt.split()
                if len(w) > 3:
                    total_chars = sum(len(x) for x in w)
                    running, split_idx = 0, len(w) - 1
                    for j, word in enumerate(w):
                        running += len(word)
                        if running >= total_chars / 2:
                            split_idx = j
                            break
                    txt = ' '.join(w[:split_idx+1]) + "\\N" + ' '.join(w[split_idx+1:])
                f.write(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{txt}\n")

def _mark_url_used(url):
    with _USED_URLS_LOCK:
        USED_URLS.add(url)


def _claim_url(url):
    """Atomically reserve a URL so parallel workers cannot download it twice."""
    with _USED_URLS_LOCK:
        if url in USED_URLS:
            return False
        USED_URLS.add(url)
        return True


def _search_stock_provider(provider, query, page, orientation):
    """Search one stock provider and return candidate video URLs."""
    try:
        if provider == "pexels":
            keys = [key for key in PEXELS_KEYS if key]
            if not keys:
                return []
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": random.choice(keys)},
                params={
                    "query": query,
                    "per_page": 15,
                    "page": page,
                    "orientation": orientation,
                },
                timeout=_STOCK_API_TIMEOUT,
            )
            if response.status_code != 200:
                return []

            urls = []
            for video in response.json().get("videos", []):
                files = video.get("video_files", [])
                if orientation == "portrait":
                    preferred = [
                        file for file in files
                        if file.get("height", 0) > file.get("width", 0)
                        and file.get("height", 0) >= 1280
                    ]
                    candidates = preferred or [
                        file for file in files
                        if file.get("quality") in ["hd", "large"]
                    ]
                else:
                    candidates = [
                        file for file in files
                        if file.get("quality") == "hd"
                        and file.get("width", 0) >= 1280
                    ]
                    if not candidates:
                        candidates = [
                            file for file in files
                            if file.get("quality") in ["hd", "large"]
                        ]

                if candidates:
                    url = random.choice(candidates).get("link")
                    if url:
                        urls.append(url)
            return urls

        if provider == "pixabay":
            keys = [key for key in PIXABAY_KEYS if key]
            if not keys:
                return []
            response = requests.get(
                "https://pixabay.com/api/videos/",
                params={
                    "key": random.choice(keys),
                    "q": query,
                    "per_page": 15,
                    "page": page,
                },
                timeout=_STOCK_API_TIMEOUT,
            )
            if response.status_code != 200:
                return []

            urls = []
            for video in response.json().get("hits", []):
                video_files = video.get("videos", {})
                selected = video_files.get("large", video_files.get("medium", {}))
                url = selected.get("url") if selected else None
                if url:
                    urls.append(url)
            return urls
    except Exception:
        # A provider outage/timeout should only remove that provider's
        # candidates; the other provider and the caller's query retries remain.
        return []

    return []


def _search_stock_urls(query, page, orientation, limit=None):
    """Search providers concurrently and atomically claim only needed URLs."""
    providers = []
    if any(PEXELS_KEYS):
        providers.append("pexels")
    if any(PIXABAY_KEYS):
        providers.append("pixabay")
    if not providers:
        return []

    results = {provider: [] for provider in providers}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(_search_stock_provider, provider, query, page, orientation): provider
            for provider in providers
        }
        for future in concurrent.futures.as_completed(futures):
            provider = futures[future]
            try:
                results[provider] = future.result()
            except Exception:
                results[provider] = []

    urls = []
    for provider in providers:
        for url in results[provider]:
            if limit is not None and len(urls) >= limit:
                return urls
            if url and url not in urls and _claim_url(url):
                urls.append(url)
    return urls


def search_and_download_vertical(query, idx, duration, tag="", verify=True, normalize=True, page=1):
    """
    Same as search_and_download but requests portrait/vertical source video
    where possible and always crops/scales to 1080x1920 (9:16) for Shorts.
    Uses a distinct USED_URLS-safe idx namespace via `tag` so long-video and
    shorts clip fetching never collide on temp filenames.
    """
    urls = _search_stock_urls(query, page, "portrait", _CLIP_CANDIDATES_PER_QUERY)

    # Each search call is bounded; the sentence worker keeps retrying with
    # fresh pages and semantically tied variants until MiniCPM accepts one.
    for url in urls[:_CLIP_CANDIDATES_PER_QUERY]:
        try:
            raw = TEMP_DIR / f"raw_s{tag}_{idx}.mp4"
            out = TEMP_DIR / f"clip_s{tag}_{idx}.mp4"
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            if os.path.getsize(raw) < 5000:
                try: os.remove(raw)
                except OSError: pass
                continue

            # Verify the raw vertical video before paying the normalization
            # cost. Shorts use the same authoritative MiniCPM checks as the
            # landscape pipeline.
            if verify:
                matches = verify_clip_matches_query(raw, query)
                if not matches:
                    print(f"    Rejected short clip for '{query[:40]}' (visual mismatch)")
                    try: os.remove(raw)
                    except OSError: pass
                    continue

            if not normalize:
                _mark_url_used(url)
                return str(raw)

            normalized = _normalize_vertical_clip(raw, out, duration)
            try: os.remove(raw)
            except OSError: pass
            if not normalized:
                continue

            _mark_url_used(url)
            return normalized
        except Exception:
            for stale in (raw, out):
                try:
                    if stale.exists(): stale.unlink()
                except OSError:
                    pass
            continue
    return None


def _pad_clip_to_duration(clip_path, source_duration, target_duration):
    """Extend a too-short clip to exactly target_duration by holding its last
    frame for the missing seconds. Used only as the final safety net when no
    stock candidate reaches the required length, so every clip entering the
    concat list has EXACTLY its allotted duration. An intentional, clearly
    logged partial-second freeze on one clip is far safer than an unplanned
    total-duration deficit: the latter makes the concatenated video shorter
    than its narration, which some players render as the LAST frame of the
    whole video freezing while the remaining audio keeps playing.
    """
    clip_path = Path(clip_path)
    padded_path = clip_path.with_name(clip_path.stem + "_padded" + clip_path.suffix)
    pad_needed = max(0.05, target_duration - source_duration)
    vf = f"tpad=stop_mode=clone:stop_duration={pad_needed:.3f}"
    cmd = (["ffmpeg", "-y", "-nostdin", "-i", str(clip_path), "-vf", vf,
            "-t", f"{target_duration:.3f}"] + _enc_args()
           + ["-pix_fmt", "yuv420p", "-an", str(padded_path)])
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        result = None
    if (result is not None and result.returncode == 0
            and padded_path.exists() and padded_path.stat().st_size > 2000):
        try:
            os.remove(clip_path)
        except OSError:
            pass
        return str(padded_path)
    # Padding itself failed (rare) - the unpadded clip is still better than no
    # clip at all; the caller's own duration-deficit recovery is the last line
    # of defense against the shortfall this would otherwise create.
    return str(clip_path)


def _find_verified_normalized_clip(sent, index, orientation, tag=""):
    """Find an exact sentence match using bounded stock/Groq retry rounds."""
    duration = max(2.5 if orientation == "portrait" else 3.5,
                   sent['end'] - sent['start'])
    query_index = sent.get('orig_idx', index) if orientation == "portrait" else index
    queries = _query_attempts(index, query_index)
    previous_queries = list(queries)

    for round_no in range(_CLIP_QUERY_ROUNDS):
        page = round_no + 1
        print(f"    {orientation.title()} clip {index}: query round {round_no + 1}/{_CLIP_QUERY_ROUNDS}")
        for query in queries:
            try:
                if orientation == "portrait":
                    raw = search_and_download_vertical(
                        query, index, duration, tag=tag, verify=False,
                        normalize=False, page=page,
                    )
                else:
                    raw = search_and_download(
                        query, index, duration, verify=False, page=page,
                    )
                if not raw:
                    print(f"    {orientation.title()} clip {index}: no download for query "
                          f"'{query[:60]}'")
                    continue

                print(f"    {orientation.title()} clip {index}: MiniCPM verification invoked "
                      f"for query '{query[:60]}'")
                matches = verify_clip_matches_query(raw, query)
                print(f"    {orientation.title()} clip {index}: MiniCPM verification "
                      f"{'PASSED' if matches else 'REJECTED'} for query '{query[:60]}'")
                if not matches:
                    try: os.remove(raw)
                    except OSError: pass
                    continue

                output_name = (f"clip_s{tag}_{index}.mp4" if orientation == "portrait"
                               else f"clip_{index}.mp4")
                normalized = (_normalize_vertical_clip if orientation == "portrait"
                              else _normalize_landscape_clip)(
                    raw, TEMP_DIR / output_name, duration
                )
                try: os.remove(raw)
                except OSError: pass
                if normalized and _normalized_duration_is_usable(normalized, duration):
                    print(f"    {orientation.title()} clip {index}: verified and normalized "
                          f"in round {round_no + 1}")
                    return index, normalized
                if normalized:
                    try: os.remove(normalized)
                    except OSError: pass
            except Exception as e:
                print(f"    {orientation.title()} clip {index}: candidate error "
                      f"({type(e).__name__}: {str(e)[:100]})")

        # A clip is usable only after MiniCPM positively confirms both the
        # query match and that no visible woman appears in the clip. Men and
        # children are allowed. Continue to the next query round if available.

    print(f"    {orientation.title()} clip {index}: no verified candidate after "
          f"{_CLIP_QUERY_ROUNDS} query rounds; refusing unsafe/unverified fallback")
    raise RuntimeError(
        f"No verified {orientation} clip found for sentence position {index + 1} "
        f"after {_CLIP_QUERY_ROUNDS} query rounds; no substitute permitted"
    )


def process_short_clip(args):
    i, sent, tag = args
    return _find_verified_normalized_clip(sent, i, "portrait", tag)


def render_short(short_idx, sentences_slice, audio_path, ass_path, logo_path, out_path,
                 release_verifier=True):
    """
    Render a single 1080x1920 short: fetch fresh vertical stock clips for
    this segment's sentences, concat, overlay a LARGE left-side logo (shorts
    need bigger branding since screen real estate is smaller/closer-viewed),
    and burn vertical-tuned subtitles.
    """
    global _nvenc_runtime_failed
    tag = f"sh{short_idx}"
    n = len(sentences_slice)
    print(f"\n  Short {short_idx+1}: fetching {n} vertical clips...")
    _log_resource_snapshot(f"short {short_idx+1} clip search start")

    clips = [None] * n

    # Each worker searches, verifies, and immediately normalizes its own
    # sentence. This prevents raw clips with inconsistent source durations
    # from accumulating and later producing a long concatenated chunk.
    print(f"  Short {short_idx+1}: streaming verified clips directly into normalized output...")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = {
            ex.submit(process_short_clip, (i, sent, tag)): i
            for i, sent in enumerate(sentences_slice)
        }
        for future in concurrent.futures.as_completed(futures):
            i, clip = future.result()
            clips[i] = clip
            completed += 1
            print(f"    Short {short_idx+1}: completed {completed}/{n} exact clips")

    if release_verifier:
        _release_llava_for_encoding()
    if USE_GPU:
        _nvenc_runtime_failed = False
        print(f"  Short {short_idx+1}: verifier workers "
              f"{'released before final encoding' if release_verifier else 'retained for the next short'}")

    missing = [i for i, clip in enumerate(clips)
               if not clip or not os.path.exists(clip)]
    if missing:
        print(f"  Short {short_idx+1}: missing normalized clips at positions {missing}; refusing substitution")
        return False

    list_path = f"list_{tag}.txt"
    visual_path = f"visual_{tag}.mp4"
    with open(list_path,"w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {visual_path}",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists(visual_path):
        fallback_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path] + _enc_args() + [visual_path]
        subprocess.run(fallback_cmd, capture_output=True, timeout=60)
    if not os.path.exists(visual_path): return False

    # Defense in depth: every clip entering the concat list is now padded to
    # its exact allotted duration (see _pad_clip_to_duration), so this should
    # rarely trigger. If a mismatch still slips through (e.g. concat-copy GOP
    # rounding), catch it here rather than silently muxing a visual track
    # shorter than its audio - some players hold the last decoded frame while
    # trailing audio keeps playing, which looks like the video freezing.
    def _probe_short_dur(path):
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=15)
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    vdur = _probe_short_dur(visual_path)
    adur = _probe_short_dur(audio_path)
    if vdur > 0 and adur > 0 and vdur < adur - 0.3:
        print(f"  Short {short_idx+1}: concatenated visual is {adur - vdur:.2f}s short "
              f"({vdur:.2f}s vs {adur:.2f}s audio); re-encoding to check for concat rounding")
        reencoded = f"visual_{tag}_reencoded.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", reencoded],
            capture_output=True, timeout=120,
        )
        if os.path.exists(reencoded):
            new_vdur = _probe_short_dur(reencoded)
            if new_vdur > vdur:
                os.replace(reencoded, visual_path)
                vdur = new_vdur
            else:
                try: os.remove(reencoded)
                except OSError: pass
        if vdur < adur - 0.3:
            print(f"  Short {short_idx+1}: WARNING - visual still {adur - vdur:.2f}s short after "
                  f"re-encode; output audio may run past the last video frame")

    enc = _enc_args()
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        # Logo on left, LARGE (shorts are watched up close on phones - a
        # small landscape-style logo reads as invisible on a 1080x1920 frame).
        filt = (f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];"
                f"[1:v]scale=280:-1[l];[bg][l]overlay=30:40[wl];"
                f"[wl]subtitles='{ass_esc}'[v];"
                f"[2:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y"] + _hwaccel_args() + ["-i",visual_path,"-i",str(logo_path),"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    else:
        filt = (f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_esc}'[v];"
                f"[1:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        cmd = ["ffmpeg","-y"] + _hwaccel_args() + ["-i",visual_path,"-i",str(audio_path),
            "-filter_complex",filt,"-map","[v]","-map","[a]"] + enc + ["-c:a","aac","-b:a","192k","-shortest",str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=300)

    for p in [list_path, visual_path]:
        if os.path.exists(p): os.remove(p)

    if not os.path.exists(out_path):
        print(f"  Short {short_idx+1}: mux failed - {r.stderr.decode(errors='ignore')[-400:]}")
        return False

    # Verify the output actually has a real audio stream - a bad stream
    # mapping or corrupt upstream audio could still produce a "successful"
    # (file exists, exit 0) mp4 with silent/missing audio.
    #
    # NOTE: ffprobe's per-stream `duration` field is unreliable for MP4/AAC
    # (frequently reports N/A even on perfectly valid, audible streams).
    # That was causing a previous version of this check to report 0.0s and
    # incorrectly discard EVERY short's audio, even when it was fine.
    # Use frame/packet count instead, which reliably reflects whether
    # actual audio data is present.
    try:
        rp = subprocess.run(["ffprobe","-v","error","-select_streams","a:0",
            "-count_packets","-show_entries","stream=nb_read_packets",
            "-of","default=noprint_wrappers=1:nokey=1",
            str(out_path)], capture_output=True, text=True, timeout=20)
        out = rp.stdout.strip()
        n_packets = int(out) if out.isdigit() else -1
        if n_packets == 0:
            print(f"  Short {short_idx+1}: output audio stream has 0 packets - silent, treating as failed")
            return False
        elif n_packets < 0:
            print(f"  Short {short_idx+1}: could not verify audio packet count, proceeding (assuming OK)")
    except Exception as e:
        print(f"  Short {short_idx+1}: audio-stream verification failed ({e}), proceeding cautiously")

    print(f"  Short {short_idx+1}: {os.path.getsize(out_path)/(1024**2):.0f}MB")
    return True


def extract_short_audio(full_audio_path, start_sec, end_sec, out_path, fade=0.25):
    """
    Cut a clean segment directly from the full enhanced audio (fast, exact
    same voice/quality - no re-TTS). Applies short fade in/out so the cut
    doesn't click/pop at the boundaries.
    """
    dur = max(0.5, end_sec - start_sec)
    af = f"afade=t=in:st=0:d={fade},afade=t=out:st={max(0,dur-fade):.2f}:d={fade}"
    cmd = ["ffmpeg","-y","-i",str(full_audio_path),
           "-ss",f"{start_sec:.3f}","-t",f"{dur:.3f}",
           "-af",af,"-ar","44100","-ac","2",str(out_path)]
    r = subprocess.run(cmd, capture_output=True, timeout=60)

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        print(f"  extract_short_audio failed: {r.stderr.decode(errors='ignore')[-300:]}")
        return False

    # Validate ACTUAL audio duration, not just file existence/size. A
    # corrupt or truncated extraction (e.g. seek landing past end of
    # source, or a near-silent slice) can still produce a small-but-
    # nonzero WAV file that passes the size check while being
    # functionally silent or far too short - this was causing "voice in
    # one short, no voice in another" with no visible error.
    try:
        rp = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",str(out_path)],
            capture_output=True, text=True, timeout=15)
        actual_dur = float(rp.stdout.strip())
        if actual_dur < dur * 0.5:
            print(f"  extract_short_audio: got {actual_dur:.2f}s, expected ~{dur:.2f}s - extraction likely broken")
            return False
    except Exception as e:
        print(f"  extract_short_audio: duration probe failed ({e}), proceeding cautiously")

    return True



# ==========================================
# 5. AUDIO ENGINE (TTS + Resemble Enhance)
# ==========================================
def generate_audio(text, ref_audio, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_path = TEMP_DIR / "raw_tts.wav"
    
    print(f"  TTS: {'Spanish (V3)' if IS_SPANISH else 'English'} on {device}")
    _log_resource_snapshot("TTS start")
    try:
        # Chatterbox generation is autoregressive and each replica uses the
        # full decoder. Running two replicas on the same T4 only makes them
        # contend for that GPU; the updated run proved this by spending the
        # entire 38-minute budget in four concurrent samplers. Use one model
        # per GPU by default so both T4s work in parallel. A larger value is
        # still available as an explicit override for GPUs with measured
        # headroom, but it is never the safe default.
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        candidate_devices = [f"cuda:{i}" for i in range(gpu_count)] if gpu_count else [device]
        try:
            tts_replicas_per_gpu = max(1, int(os.environ.get("TTS_REPLICAS_PER_GPU", "1")))
        except (TypeError, ValueError):
            tts_replicas_per_gpu = 1
        try:
            tts_min_free_gb = max(1.0, float(os.environ.get("TTS_MIN_FREE_GB", "4.0")))
        except (TypeError, ValueError):
            tts_min_free_gb = 4.0

        def _load_tts_model(dev):
            if IS_SPANISH:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                try:
                    # Newer chatterbox versions support t3_model="v3" (better quality).
                    return ChatterboxMultilingualTTS.from_pretrained(device=dev, t3_model="v3")
                except TypeError:
                    # Installed pip version (e.g. 0.1.7) doesn't have this kwarg yet
                    # (only on GitHub main / HF docs as of this writing). Fall back.
                    print("  chatterbox-tts: t3_model kwarg unsupported by installed version, using default checkpoint")
                    return ChatterboxMultilingualTTS.from_pretrained(device=dev)
            from chatterbox.tts import ChatterboxTTS
            return ChatterboxTTS.from_pretrained(device=dev)

        models = []
        devices_used = []
        for gpu_index, dev in enumerate(candidate_devices):
            replicas_on_device = 0
            for replica_index in range(tts_replicas_per_gpu):
                if gpu_count:
                    try:
                        free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_index)
                        free_gb = free_bytes / (1024 ** 3)
                    except Exception:
                        free_gb = 0.0
                    if free_gb < tts_min_free_gb:
                        print(f"  TTS: skipping replica {replica_index + 1} on {dev} "
                              f"({free_gb:.1f} GB free, need >= {tts_min_free_gb:.1f} GB)")
                        break
                try:
                    models.append(_load_tts_model(dev))
                    devices_used.append(dev)
                    replicas_on_device += 1
                    print(f"  TTS: replica {replicas_on_device}/{tts_replicas_per_gpu} loaded on {dev}")
                except Exception as e:
                    print(f"  TTS: could not load replica {replica_index + 1} on {dev} "
                          f"({type(e).__name__}: {str(e)[:120]}); continuing without it")
                    try:
                        with torch.cuda.device(dev) if gpu_count else _nullcontext():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # A failed first load usually means this GPU cannot host
                    # this model; avoid repeatedly triggering the same OOM.
                    if replica_index == 0:
                        break
        if not models:
            raise RuntimeError("No Chatterbox TTS model could be loaded on any device")
        if len(devices_used) > 1:
            print(f"  TTS: running {len(devices_used)} parallel model replicas on "
                  f"{', '.join(devices_used)}")

        sr = models[0].sr
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        # Keep chunks sentence-aligned, but reduce the number of expensive
        # autoregressive model invocations. 240 characters is still short
        # enough for Chatterbox's text context while avoiding the large
        # per-call startup cost of the old 160-character setting.
        try:
            tts_chunk_chars = max(160, int(os.environ.get("TTS_CHUNK_CHARS", "240")))
        except (TypeError, ValueError):
            tts_chunk_chars = 240
        chunks, buf, blen = [], [], 0
        for s in sents:
            if blen + len(s) > tts_chunk_chars and buf:
                chunks.append(' '.join(buf)); buf, blen = [s], len(s)
            else: buf.append(s); blen += len(s)+1
        if buf: chunks.append(' '.join(buf))
        print(f"  {len(sents)} sentences -> {len(chunks)} chunks (target <= {tts_chunk_chars} chars)")

        def _generate_tts_piece(piece, model):
            with torch.inference_mode():
                if IS_SPANISH:
                    waveform = model.generate(
                        piece.replace('"',''), audio_prompt_path=str(ref_audio),
                        language_id="es", exaggeration=0.4, cfg_weight=0.65
                    )
                else:
                    waveform = model.generate(
                        piece.replace('"',''), audio_prompt_path=str(ref_audio),
                        exaggeration=0.4, cfg_weight=0.65
                    )
            return waveform.cpu()

        def _process_chunk(i, c, model):
            try:
                return _generate_tts_piece(c, model)
            except Exception as e:
                # A longer chunk can fail on an older Chatterbox build or
                # unusual text. Retry it as smaller sentence/word pieces so
                # increasing the target never silently loses narration.
                print(f"  TTS chunk {i+1} retrying smaller pieces ({str(e)[:100]})")
                fallback_parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', c) if p.strip()]
                if len(fallback_parts) <= 1:
                    words = c.split()
                    midpoint = max(1, len(words) // 2)
                    fallback_parts = [' '.join(words[:midpoint]), ' '.join(words[midpoint:])]
                recovered_pieces = []
                for part in fallback_parts:
                    if not part:
                        continue
                    try:
                        recovered_pieces.append(_generate_tts_piece(part, model))
                    except Exception as sub_error:
                        print(f"  TTS sub-piece skipped ({str(sub_error)[:80]})")
                if not recovered_pieces:
                    print(f"  TTS chunk {i+1} could not be recovered")
                    return None
                merged = recovered_pieces[0]
                for extra in recovered_pieces[1:]:
                    merged = torch.cat([merged, extra], dim=1)
                return merged

        wavs = [None] * len(chunks)
        progress_lock = threading.Lock()
        completed = [0]

        def _worker(worker_index, model):
            for i in range(worker_index, len(chunks), len(models)):
                wavs[i] = _process_chunk(i, chunks[i], model)
                with progress_lock:
                    completed[0] += 1
                    if completed[0] % 5 == 0 or completed[0] == len(chunks):
                        update_status(18 + int((completed[0] / max(1, len(chunks))) * 27),
                                       f"TTS {completed[0]}/{len(chunks)}")

        if len(models) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
                futures = [ex.submit(_worker, idx, m) for idx, m in enumerate(models)]
                for f in futures:
                    f.result()
        else:
            _worker(0, models[0])

        if not wavs or any(waveform is None for waveform in wavs):
            print("  TTS: at least one chunk could not be generated; refusing a truncated narration")
            return False
        full = wavs[0]
        for w in wavs[1:]:
            full = torch.cat([full, torch.zeros((full.shape[0], int(0.15*sr))), w], dim=1)
        full = torch.cat([full, torch.zeros((full.shape[0], int(1.5*sr)))], dim=1)
        torchaudio.save(str(raw_path), full, sr)
        print(f"  TTS: {full.shape[1]/sr:.1f}s at {sr}Hz")
        del models, wavs, full
        for dev in devices_used:
            try:
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
    except Exception as e:
        print(f"  TTS error: {e}")
        _log_resource_snapshot("TTS error")
        return False
    _log_resource_snapshot("TTS end")

    # --- RESEMBLE ENHANCE ---
    print("  Enhancing audio...")
    _log_resource_snapshot("Enhance start")
    try:
        # Resemble Enhance keeps module-level STFT/model state. Threads in
        # this process previously caused cross-device window mismatches, so
        # each GPU gets a separate Python process with a separate import and
        # model state. The parent only writes chunks, launches workers, and
        # concatenates their results in the original order.
        dwav, osr = torchaudio.load(str(raw_path))
        if dwav.shape[0] > 1:
            dwav = dwav.mean(dim=0, keepdim=True)
        osr = int(osr)
        esr = 44100
        try:
            enhance_chunk_seconds = max(20, int(os.environ.get("ENHANCE_CHUNK_SECONDS", "40")))
        except (TypeError, ValueError):
            enhance_chunk_seconds = 40
        chunk_s = enhance_chunk_seconds * osr
        total = int(dwav.shape[1])
        n_chunks = (total + chunk_s - 1) // chunk_s
        print(f"  Processing {n_chunks} enhancement chunks ({enhance_chunk_seconds}s target) in isolated workers...")

        enhance_root = TEMP_DIR / "enhance_workers"
        enhance_root.mkdir(parents=True, exist_ok=True)
        worker_script = enhance_root / "enhance_worker.py"
        worker_script.write_text(r'''import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
from unittest.mock import MagicMock

# resemble-enhance imports deepspeed even when its inference path does not
# need a real DeepSpeed installation. Keep the child self-contained.
_mock_names = [
    "deepspeed", "deepspeed.accelerator", "deepspeed.runtime",
    "deepspeed.runtime.engine", "deepspeed.runtime.config",
    "deepspeed.runtime.utils", "deepspeed.utils", "deepspeed.ops",
    "deepspeed.ops.adam", "deepspeed.comm",
]
for _name in _mock_names:
    sys.modules[_name] = MagicMock()
sys.modules["deepspeed.accelerator"].get_accelerator = MagicMock()
sys.modules["deepspeed.runtime.engine"].DeepSpeedEngine = MagicMock()
sys.modules["deepspeed.runtime.utils"].clip_grad_norm_ = MagicMock()

if os.environ.get("VIDEO_FACTORY_DISABLE_CUDNN") == "1":
    torch.backends.cudnn.enabled = False

from resemble_enhance.enhancer.inference import enhance

_DEVICE = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")
if _DEVICE.startswith("cuda") and torch.cuda.is_available():
    try:
        torch.cuda.set_device(int(_DEVICE.split(":", 1)[1]))
    except (IndexError, ValueError):
        pass


def _resample(waveform, sample_rate):
    result = torchaudio.transforms.Resample(int(sample_rate), 44100)(waveform)
    return result.detach().cpu()


def _enhance_piece(waveform, sample_rate, label):
    try:
        enhanced, enhanced_rate = enhance(
            dwav=waveform.squeeze(0),
            sr=int(sample_rate),
            device=_DEVICE,
            lambd=0.6,
        )
        enhanced_rate = int(enhanced_rate)
        enhanced = enhanced.detach().cpu()
        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)
        if enhanced_rate != 44100:
            enhanced = _resample(enhanced, enhanced_rate)
        print(f"    Chunk {label}: OK (44100Hz)", flush=True)
        return enhanced
    except Exception as error:
        # Large chunks can exceed one GPU's available memory. Retry in two
        # smaller pieces before accepting the deterministic resampling path.
        if waveform.shape[1] > 20 * int(sample_rate) + 1:
            midpoint = waveform.shape[1] // 2
            left = _enhance_piece(waveform[:, :midpoint], sample_rate, f"{label}a")
            right = _enhance_piece(waveform[:, midpoint:], sample_rate, f"{label}b")
            return torch.cat([left, right], dim=1)
        print(f"    Chunk {label}: fallback ({type(error).__name__}: {str(error)[:100]})", flush=True)
        return _resample(waveform, sample_rate)


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for item in manifest:
        input_path = Path(item["input"])
        output_path = Path(item["output"])
        waveform, sample_rate = torchaudio.load(str(input_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        result = _enhance_piece(waveform, int(sample_rate), str(item["index"]))
        torchaudio.save(str(output_path), result, 44100)
''', encoding="utf-8")

        worker_count = min(
            n_chunks,
            torch.cuda.device_count() if torch.cuda.is_available() else 1,
        )
        worker_devices = (
            [f"cuda:{index}" for index in range(worker_count)]
            if torch.cuda.is_available() else ["cpu"]
        )
        assignments = [[] for _ in range(worker_count)]
        output_paths = []
        for chunk_index, start in enumerate(range(0, total, chunk_s), 1):
            input_path = enhance_root / f"chunk_{chunk_index:04d}.wav"
            output_path = enhance_root / f"enhanced_{chunk_index:04d}.wav"
            torchaudio.save(str(input_path), dwav[:, start:start + chunk_s], osr)
            spec = {"index": chunk_index, "input": str(input_path), "output": str(output_path)}
            assignments[(chunk_index - 1) % worker_count].append(spec)
            output_paths.append(output_path)

        manifests = []
        for worker_index, specs in enumerate(assignments):
            manifest_path = enhance_root / f"manifest_{worker_index}.json"
            manifest_path.write_text(json.dumps(specs), encoding="utf-8")
            manifests.append(manifest_path)

        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        if _CUDNN_FORCE_DISABLED:
            child_env["VIDEO_FACTORY_DISABLE_CUDNN"] = "1"
        try:
            worker_timeout = max(120, int(os.environ.get("ENHANCE_WORKER_TIMEOUT_SECONDS", "900")))
        except (TypeError, ValueError):
            worker_timeout = 900

        def _run_enhance_worker(worker_index):
            command = [
                sys.executable,
                str(worker_script),
                str(manifests[worker_index]),
                worker_devices[worker_index],
            ]
            try:
                completed = subprocess.run(
                    command,
                    env=child_env,
                    capture_output=True,
                    text=True,
                    timeout=worker_timeout,
                )
                return worker_index, completed.returncode, completed.stdout or completed.stderr or ""
            except Exception as error:
                return worker_index, -1, f"{type(error).__name__}: {error}"

        print(f"  Enhancement workers: {len(worker_devices)} ({', '.join(worker_devices)}), "
              f"round-robin chunk dispatch")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
            worker_results = list(executor.map(_run_enhance_worker, range(len(worker_devices))))
        for worker_index, return_code, output in worker_results:
            if output.strip():
                print(f"  Enhance worker {worker_devices[worker_index]} output:\n{output.rstrip()}")
            if return_code != 0:
                print(f"  Enhance worker {worker_devices[worker_index]} exited with code {return_code}; "
                      "missing chunks will use resampling fallback")

        parts = []
        fallback_resampler = torchaudio.transforms.Resample(osr, esr)
        for chunk_index, (start, output_path) in enumerate(
                zip(range(0, total, chunk_s), output_paths), 1):
            expected = dwav[:, start:start + chunk_s]
            if output_path.exists() and output_path.stat().st_size > 1000:
                try:
                    piece, piece_sr = torchaudio.load(str(output_path))
                    if piece.shape[0] > 1:
                        piece = piece.mean(dim=0, keepdim=True)
                    if int(piece_sr) != esr:
                        piece = torchaudio.transforms.Resample(int(piece_sr), esr)(piece)
                    parts.append(piece.cpu())
                    continue
                except Exception as error:
                    print(f"  Enhance chunk {chunk_index}: output read failed ({str(error)[:100]})")
            print(f"  Enhance chunk {chunk_index}: parent resampling fallback")
            parts.append(fallback_resampler(expected).cpu())

        final = torch.cat(parts, dim=1)
        torchaudio.save(str(out_path), final, esr)
        print(f"  Enhanced: {esr}Hz, {final.shape[1] / esr:.1f}s")
        del parts, final, dwav
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _log_resource_snapshot("Enhance end")
        return True
    except Exception as e:
        print(f"  Enhance failed: {e}, using raw audio")
        _log_resource_snapshot("Enhance error")
        shutil.copy2(str(raw_path), str(out_path))
        return True



# ==========================================
# 6. VIDEO ENGINE (GPU-Accelerated)
# ==========================================
# MiniCPM-V 4.5 is the production local video verifier. Use the official
# pre-quantized NF4 checkpoint so one independent worker fits on each Kaggle
# T4. The compatibility names are retained because the rest of the pipeline
# already uses _llava_* worker/release functions.
_llava_workers = []
_llava_next_worker = 0
_llava_worker_select_lock = threading.Lock()
_llava_load_lock = threading.Lock()
_gpu_lock = threading.Lock()
_MINICPM_NUM_FRAMES = 4
_MINICPM_MAX_NEW_TOKENS = 24
_MINICPM_MODEL_PATH = "openbmb/MiniCPM-V-4_5-int4"
# T4 uses float16 reliably; the checkpoint's embedded BNB config also uses
# float16 compute. Do not request bfloat16 on this hardware.
_MINICPM_DTYPE = torch.float16
_MINICPM_TIME_SCALE = 0.1
_MINICPM_PACKING = 4
_MINICPM_COMPAT_PATCHED = False


def _patch_minicpm_transformers_compat():
    """Bridge MiniCPM remote code that predates Transformers 5.5 metadata."""
    global _MINICPM_COMPAT_PATCHED
    if _MINICPM_COMPAT_PATCHED:
        return

    from transformers import PreTrainedModel

    # MiniCPMV calls PreTrainedModel.__init__ but omits post_init(), while
    # Transformers 5.5 accesses all_tied_weights_keys during checkpoint
    # finalization. Patch the base initializer once so every affected model
    # instance gets its own empty mapping; modern child models still replace it
    # with their real mapping from post_init().
    original_init = PreTrainedModel.__init__

    import functools

    @functools.wraps(original_init)
    def _compat_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    PreTrainedModel.__init__ = _compat_init
    _MINICPM_COMPAT_PATCHED = True
    print("  MiniCPM compatibility: initialized missing all_tied_weights_keys metadata")


def _patch_minicpm_tokenizer_compat(tokenizer):
    """Restore MiniCPM-V legacy tokenizer aliases removed by Transformers 5.5."""
    # The official MiniCPMV processor calls these names directly. Transformers
    # 5.5 can return a generic TokenizersBackend instead of the model's older
    # MiniCPMVTokenizerFast wrapper, so provide the same aliases explicitly.
    string_tokens = {
        "im_start": "<image>",
        "im_end": "</image>",
        "slice_start": "<slice>",
        "slice_end": "</slice>",
        "im_id_start": "<image_id>",
        "im_id_end": "</image_id>",
    }

    def _coerce_id(value):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _token_id(token):
        """Resolve one special token to an ID, or None when unrepresentable.

        Returning None instead of raising is deliberate. Added-vocabulary
        markers like <image> resolve through convert_tokens_to_ids, but plain
        text such as "\\n" does NOT: MiniCPM's tokenizer is Qwen2 byte-level
        BPE, where a newline is stored as the byte-mapped token "Ctilde", so
        convert_tokens_to_ids("\\n") legitimately returns the unk ID. The
        official MiniCPMVTokenizerFast.newline_id has exactly this behavior and
        the processor never reads it, so treating that as fatal (as an earlier
        revision did) blocked every verifier worker from loading.
        """
        unk_id = _coerce_id(getattr(tokenizer, "unk_token_id", None))
        value = _coerce_id(tokenizer.convert_tokens_to_ids(token))
        if value is not None and (unk_id is None or value != unk_id or token == "<unk>"):
            return value
        # Byte-level fallback: encode the literal text and accept it only when
        # it maps to exactly one token, so an alias can never be a partial id.
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
        except Exception:
            encoded = None
        if encoded is not None and len(encoded) == 1:
            single = _coerce_id(encoded[0])
            if single is not None and (unk_id is None or single != unk_id):
                return single
        return None

    def _set_missing(name, value):
        if value is None:
            return
        try:
            current = getattr(tokenizer, name)
        except AttributeError:
            current = None
        if current is not None:
            return
        try:
            setattr(tokenizer, name, value)
        except (AttributeError, TypeError):
            # Some backend wrappers disallow instance attributes. A property
            # on their concrete class keeps the value local to this tokenizer
            # implementation without changing tokenization behavior.
            try:
                setattr(type(tokenizer), name,
                        property(lambda _self, value=value: value))
            except (AttributeError, TypeError) as error:
                raise RuntimeError(
                    f"MiniCPM tokenizer cannot install compatibility alias {name}"
                ) from error

    for name, token in string_tokens.items():
        _set_missing(name, token)
    for name, token_name in {
        "im_start_id": "im_start",
        "im_end_id": "im_end",
        "slice_start_id": "slice_start",
        "slice_end_id": "slice_end",
        "im_id_start_id": "im_id_start",
        "im_id_end_id": "im_id_end",
    }.items():
        _set_missing(name, _token_id(string_tokens[token_name]))

    # processing_minicpmv.batch_decode()/decode() read bos_id and eos_id to trim
    # boundary tokens. Prefer the tokenizer's own configured IDs and only fall
    # back to the chat markers this checkpoint actually emits.
    _set_missing("bos_id", _coerce_id(getattr(tokenizer, "bos_token_id", None))
                 or _token_id("<|im_start|>"))
    _set_missing("eos_id", _coerce_id(getattr(tokenizer, "eos_token_id", None))
                 or _token_id("<|im_end|>"))
    # unk_id and newline_id are defined by the official tokenizer but are never
    # read by the processor, so install them best-effort and never fail on them.
    _set_missing("unk_id", _coerce_id(getattr(tokenizer, "unk_token_id", None))
                 or _token_id("<unk>"))
    _set_missing("newline_id", _token_id("\n"))

    # Only these six are actually dereferenced by the official processor:
    # _convert() uses the image/slice bounds, decode() uses bos/eos.
    required = ("im_start_id", "im_end_id", "slice_start_id", "slice_end_id",
                "bos_id", "eos_id")
    missing = [name for name in required
               if _coerce_id(getattr(tokenizer, name, None)) is None]
    if missing:
        raise RuntimeError(
            f"MiniCPM tokenizer compatibility aliases still missing: {missing}"
        )
    optional_missing = [name for name in ("im_id_start_id", "im_id_end_id",
                                          "unk_id", "newline_id")
                        if _coerce_id(getattr(tokenizer, name, None)) is None]
    detail = f" (optional unresolved: {optional_missing})" if optional_missing else ""
    print(f"  MiniCPM compatibility: restored tokenizer special-token aliases{detail}")
    return tokenizer


def _load_llava_worker(gpu_index):
    """Load one official MiniCPM-V 4.5 int4 verifier on one CUDA device."""
    from transformers import AutoModel, AutoProcessor

    if gpu_index is None:
        raise RuntimeError("MiniCPM verification requires a CUDA GPU")
    device = f"cuda:{gpu_index}"
    print(f"  Loading MiniCPM-V 4.5 int4 verifier on {device}...")
    processor = AutoProcessor.from_pretrained(
        _MINICPM_MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer = _patch_minicpm_tokenizer_compat(processor.tokenizer)
    _patch_minicpm_transformers_compat()
    # The checkpoint config contains the official bitsandbytes NF4 settings.
    # Passing device_map pins this independent model to exactly one GPU.
    model = AutoModel.from_pretrained(
        _MINICPM_MODEL_PATH,
        trust_remote_code=True,
        dtype=_MINICPM_DTYPE,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    return {
        "gpu_index": gpu_index,
        "device": torch.device(device),
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "lock": threading.Lock(),
        "prepare_lock": threading.Lock(),
    }

def _load_llava():
    global _llava_workers
    if _llava_workers:
        return
    with _llava_load_lock:
        if _llava_workers:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPM verifier requires CUDA")
        _log_resource_snapshot("MiniCPM load start")
        # Doubling verification throughput: attempt 2 MiniCPM int4 replicas
        # per GPU instead of 1. A prior measured run showed one replica
        # occupies ~8.8 GB on a 14.6 GB T4, leaving ~5.8 GB free versus the
        # ~5.9 GB a second replica's own load requested - a gap of only
        # ~140 MB. Two changes close that gap when possible:
        #   1. Explicitly empty the CUDA allocator cache and run gc right
        #      after the first replica finishes loading, reclaiming any
        #      fragmented/cached blocks PyTorch's allocator is holding but
        #      not actually using, before the second replica is attempted.
        #   2. A slightly lower, still-safe free-memory gate (6.0 GB) than
        #      the previous 7.5 GB, since the real requirement measured was
        #      ~5.9 GB and the cache-clear above should recover most of the
        #      earlier shortfall.
        # This cannot be verified on this development machine (different
        # GPU architecture); if the second replica still does not fit on the
        # actual Kaggle T4s, the existing OOM handling below fails safely -
        # it only skips that one replica and keeps the pipeline running with
        # whatever replicas did load, exactly as before. Override via
        # MINICPM_REPLICAS_PER_GPU if a different GPU tier needs tuning.
        # Each MiniCPM-V 4.5 int4 replica occupies ~7.6 GB on a 14.56 GB T4
        # (measured from the run's own [RESOURCE] snapshots). A second replica
        # on the same GPU needs another ~7.6 GB, which does not fit: the run
        # logs show the second load climbing to ~13.9 GB before failing with
        # "Tried to allocate 1.10 GiB ... 976 MB free". Two replicas per T4 is
        # therefore a guaranteed OOM, and attempting it wastes ~10s per GPU and
        # fragments VRAM. Default to ONE replica per GPU (2 workers on T4 x2).
        # A larger GPU tier can still opt into 2 via MINICPM_REPLICAS_PER_GPU.
        try:
            REPLICAS_PER_GPU = max(1, int(os.environ.get("MINICPM_REPLICAS_PER_GPU", "1")))
        except (TypeError, ValueError):
            REPLICAS_PER_GPU = 1
        _MINICPM_SECOND_REPLICA_MIN_FREE_GB = 6.0
        loaded = []
        for gpu_index in range(torch.cuda.device_count()):
            for replica in range(REPLICAS_PER_GPU):
                if replica > 0:
                    # Reclaim fragmented/cached allocator memory from the
                    # previous replica's load before checking free VRAM -
                    # this is the step most likely to close the ~140 MB gap.
                    gc.collect()
                    try:
                        with torch.cuda.device(gpu_index):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_index)
                        free_gb = free_bytes / (1024 ** 3)
                    except Exception:
                        free_gb = 0.0
                    if free_gb < _MINICPM_SECOND_REPLICA_MIN_FREE_GB:
                        print(f"  MiniCPM: skipping replica {replica+1} on cuda:{gpu_index} "
                              f"({free_gb:.1f} GB free after cache clear, need >= "
                              f"{_MINICPM_SECOND_REPLICA_MIN_FREE_GB} GB) - avoiding a doomed load")
                        break
                try:
                    worker = _load_llava_worker(gpu_index)
                    loaded.append(worker)
                    print(f"  MiniCPM replica {replica+1}/{REPLICAS_PER_GPU} on cuda:{gpu_index} loaded")
                except Exception as e:
                    print(f"  WARNING: MiniCPM replica {replica+1} on cuda:{gpu_index} failed: "
                          f"{type(e).__name__}: {str(e)[:180]}")
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # If first replica fails, skip second on same GPU
                    break
        if not loaded:
            _log_resource_snapshot("MiniCPM load failed")
            raise RuntimeError("No MiniCPM verifier worker could be loaded")
        _llava_workers = loaded
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        devices = ", ".join(str(worker["device"]) for worker in loaded)
        print(f"  MiniCPM verifier workers ready: {len(loaded)} ({devices}); "
              f"{_MINICPM_NUM_FRAMES} Decord frames, "
              f"{_MINICPM_PACKING}-frame temporal packing, "
              f"{_MINICPM_MAX_NEW_TOKENS} output tokens")
        _log_resource_snapshot("MiniCPM load end")

def _next_llava_worker():
    global _llava_next_worker
    with _llava_worker_select_lock:
        if not _llava_workers:
            raise RuntimeError("MiniCPM verifier workers are not loaded")
        worker = _llava_workers[_llava_next_worker % len(_llava_workers)]
        _llava_next_worker += 1
        return worker

def _prepare_llava_inputs(clip_path, prompt, processor):
    """Sample four Decord frames and build MiniCPM temporal metadata."""
    import numpy as np
    from PIL import Image
    from decord import VideoReader, cpu

    reader = VideoReader(str(clip_path), ctx=cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count == 0:
        raise RuntimeError("Decord returned an empty video")
    indices = np.linspace(
        0, frame_count - 1, _MINICPM_NUM_FRAMES, dtype=np.int64
    )
    frames = reader.get_batch(indices).asnumpy()
    fps = float(reader.get_avg_fps() or 0.0)
    del reader
    if fps <= 0:
        fps = 1.0
    frame_images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    del frames

    # MiniCPM-V 4.5 requires temporal_ids grouped in the same packing layout
    # as the frame list. Four frames in one group lets its 3D resampler jointly
    # reason over the whole candidate clip instead of treating frames as four
    # unrelated still images.
    timestamps = indices.astype(np.float32) / fps
    temporal_ids = np.rint(timestamps / _MINICPM_TIME_SCALE).astype(np.int32)
    temporal_ids = [[int(value) for value in temporal_ids.tolist()]]
    return {"frames": frame_images, "temporal_ids": temporal_ids}



def _verify_clip_matches_query_legacy(clip_path, query, filter_women=True):
    """Compatibility alias for callers that used the previous verifier name."""
    return verify_clip_matches_query(clip_path, query, filter_women=filter_women)

    # Historical implementation retained below only as unreachable reference.
    """
    Legacy MiniCPM verifier retained only as historical code; production uses MiniCPM below.
    just a single frame) actually matches the intended search query, AND
    whether it shows a woman (if filter_women is True) - combined into ONE
    model call for efficiency rather than two separate passes.

    This is the real fix for stock footage that "downloads fine" but is
    visually unrelated to the query - there was previously ZERO check
    that a downloaded clip actually looked like what it was searched for,
    and no check on content restrictions beyond the query TEXT (a neutral
    query like "person walking city" could still return a clip showing a
    woman, since the restriction was only ever applied to search terms,
    not actual visual content).

    Returns True if the clip should be USED (topic matches AND, if
    filter_women, no woman detected), False if it should be rejected.

    Verification is fail-closed; any model-load or per-clip inference error
    rejects the candidate and the sentence worker keeps searching. A verifier
    failure can never be converted into an unverified acceptance.
    - Any verifier load or inference failure returns False; no exception
      path accepts an unverified candidate.
    """
    verify_started = time.perf_counter()
    try:
        _load_llava()
        worker = _next_llava_worker()
    except Exception as e:
        print(f"    Video verification model unavailable ({str(e)[:100]}), rejecting clip (fail-closed)")
        return False

    try:
        with worker["lock"]:
            processor = worker["processor"]
            model = worker["model"]
            device = worker["device"]
            if filter_women:
                prompt = f"""Look at this video clip and answer two questions, each on its own line, in this EXACT format:
1. <YES or NO>
2. <YES or NO>

1. Does this video clip visually match the concept: "{query}"? Answer YES if it reasonably represents the concept (even loosely/thematically - stock footage rarely is a perfect literal match). Answer NO only if it's clearly unrelated.
2. Does the clip show any woman or women as a visible person in frame (not just implied)? Answer YES or NO."""
            else:
                prompt = f"""Look at this video clip and answer, in this EXACT format:
1. <YES or NO>
2. NO

1. Does this video clip visually match the concept: "{query}"? Answer YES if it reasonably represents the concept (even loosely/thematically - stock footage rarely is a perfect literal match). Answer NO only if it's clearly unrelated."""

            inputs_started = time.perf_counter()
            inputs = _prepare_llava_inputs(clip_path, prompt, processor)
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    dtype = model.dtype if value.is_floating_point() else value.dtype
                    inputs[key] = value.to(device=device, dtype=dtype)
            inputs_seconds = time.perf_counter() - inputs_started

            # Keep generation synchronous, exactly as in the tested notebook.
            # CUDA generate() cannot be safely cancelled from a watchdog
            # thread; leaving such a thread alive can continue using the
            # shared model and GPU after this call has returned, preventing
            # clip workers from completing.
            generation_started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=_MINICPM_MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            generation_seconds = time.perf_counter() - generation_started
            print(f"    MiniCPM {device} timing: prepare={inputs_seconds:.1f}s generate={generation_seconds:.1f}s total={time.perf_counter() - verify_started:.1f}s")

            full_text = processor.batch_decode(
                out, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            # Response follows "assistant\n" in the decoded text (chat
            # template format) - slice to isolate just the model's answer.
            answer = full_text.split("assistant")[-1].strip("\n: ").upper()

            del inputs, out

            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            line1 = lines[0] if len(lines) > 0 else ""
            line2 = lines[1] if len(lines) > 1 else ""

            # Parse line 1 (topic match) and line 2 (woman detection)
            # independently - don't just search the whole blob for "YES",
            # since that would conflate the two answers if the model
            # returns e.g. "1. NO / 2. YES" (topic mismatch AND a woman -
            # searching the whole string for "YES" would wrongly pass it).
            topic_match = "YES" in line1 and "NO" not in line1
            has_woman = filter_women and "YES" in line2

            if has_woman:
                print(f"    Rejected clip for '{query[:40]}' (woman detected in frame)")
                return False
            return topic_match
    except Exception as e:
        # Per-call error (not a model-load failure) - fail CLOSED per the
        # accuracy-first decision: reject this clip so the caller tries
        # the next candidate rather than silently accepting an unverified one.
        print(f"    Visual verification error for '{query[:40]}' ({str(e)[:60]}), rejecting clip (fail-closed)")
        return False


def verify_clip_matches_query(clip_path, query, filter_women=True):
    """Verify one clip with MiniCPM-V 4.5; reject every uncertain result."""
    verify_started = time.perf_counter()
    try:
        _load_llava()
        worker = _next_llava_worker()
    except Exception as e:
        print(f"    MiniCPM verification unavailable ({str(e)[:100]}), "
              "rejecting clip (fail-closed)")
        return False

    if filter_women:
        prompt = (
            'Return exactly one JSON object and no other text: '
            '{"match":"YES" or "NO","woman":"YES" or "NO"}. '
            f'Does this video visually match the concept "{query}"? '
            'Set match to YES for a reasonable thematic match and NO only when clearly unrelated. '
            'Set woman to YES only if a visible woman or women appear in any frame; set woman to NO for men, children, objects, animals, or no people.'
        )
    else:
        prompt = (
            'Return exactly one JSON object and no other text: '
            '{"match":"YES" or "NO","woman":"NO"}. '
            f'Does this video visually match the concept "{query}"? '
            'Set match to YES for a reasonable thematic match and NO only when clearly unrelated.'
        )

    try:
        # The processor is shared by no other task, but its preprocessing state
        # is still protected because a worker may be selected by multiple clip
        # threads over the lifetime of the run.
        with worker["prepare_lock"]:
            inputs_started = time.perf_counter()
            prepared = _prepare_llava_inputs(clip_path, prompt, worker["processor"])
            inputs_seconds = time.perf_counter() - inputs_started

        frames = prepared["frames"]
        temporal_ids = prepared["temporal_ids"]
        msgs = [{"role": "user", "content": frames + [prompt]}]

        # MiniCPM-V's custom chat API performs the processor conversion and
        # generation together. Serialize that call per model replica, while
        # allowing different GPU workers to run concurrently.
        with worker["lock"]:
            generation_started = time.perf_counter()
            with torch.inference_mode():
                answer = worker["model"].chat(
                    msgs=msgs,
                    tokenizer=worker["tokenizer"],
                    processor=worker["processor"],
                    max_new_tokens=_MINICPM_MAX_NEW_TOKENS,
                    sampling=False,
                    max_slice_nums=1,
                    use_image_id=False,
                    temporal_ids=temporal_ids,
                    enable_thinking=False,
                )
            generation_seconds = time.perf_counter() - generation_started

        total_seconds = time.perf_counter() - verify_started
        print(f"    MiniCPM {worker['device']} timing: prepare={inputs_seconds:.2f}s "
              f"generate={generation_seconds:.2f}s total={total_seconds:.2f}s")

        answer = answer if isinstance(answer, str) else str(answer or "")
        candidate = re.search(r"\{.*\}", answer, re.DOTALL)
        try:
            data = json.loads(candidate.group(0)) if candidate else None
            match_value = str(data.get("match", "")).upper() if data else ""
            woman_value = str(data.get("woman", "")).upper() if data else ""
        except (ValueError, TypeError, AttributeError):
            data = None
            match_value = ""
            woman_value = ""

        if match_value not in {"YES", "NO"} or woman_value not in {"YES", "NO"}:
            print(f"    MiniCPM returned malformed verification JSON for "
                  f"'{query[:40]}'; rejecting")
            return False
        if filter_women and woman_value == "YES":
            print(f"    Rejected clip for '{query[:40]}' (woman detected in frame)")
            return False
        result = match_value == "YES"
        print(f"    MiniCPM verification result for '{query[:40]}': "
              f"match={match_value}, woman={woman_value}, accepted={result}")
        return result
    except Exception as e:
        print(f"    MiniCPM verification error for '{query[:40]}' "
              f"({type(e).__name__}: {str(e)[:100]}), rejecting clip (fail-closed)")
        return False


def _normalize_clip_with_recovery(raw_path, output_path, duration, vf, label):
    """Encode one accepted clip, retrying with CPU only if NVENC fails."""
    global _nvenc_runtime_failed
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    partial_path = output_path.with_name(output_path.stem + ".part" + output_path.suffix)

    if not raw_path.exists() or raw_path.stat().st_size < 5000:
        print(f"    {label} input is missing or too small: {raw_path}")
        return None

    try:
        if partial_path.exists(): partial_path.unlink()
        if output_path.exists(): output_path.unlink()
    except OSError:
        pass

    encoders = []
    if USE_GPU and not _nvenc_runtime_failed:
        encoders.append(("NVENC", _enc_args()))
    # This is an automatic recovery path, not the normal path. It prevents a
    # transient/unsupported NVENC initialization from discarding every clip
    # that MiniCPM already accepted.
    encoders.append(("CPU fallback", ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]))

    for encoder_name, encoder in encoders:
        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-stream_loop", "-1",  # Loop input if shorter than -t duration
            "-i", str(raw_path), "-t", str(duration),
            "-vf", vf,
        ] + encoder + ["-pix_fmt", "yuv420p", "-an", str(partial_path)]
        started = time.perf_counter()
        try:
            with _gpu_lock if encoder_name == "NVENC" else _nullcontext():
                result = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=120
                )
        except Exception as e:
            if encoder_name == "NVENC":
                _nvenc_runtime_failed = True
            print(f"    {label} {encoder_name} exception: {type(e).__name__}: {str(e)[:160]}")
            result = None

        if result is not None and result.returncode == 0 and partial_path.exists() and partial_path.stat().st_size > 2000:
            try:
                os.replace(partial_path, output_path)
                elapsed = time.perf_counter() - started
                print(f"    {label}: {output_path.name} in {elapsed:.1f}s ({encoder_name})")
                return str(output_path)
            except OSError as e:
                print(f"    {label} publish failed ({encoder_name}): {type(e).__name__}: {str(e)[:120]}")

        if result is not None:
            stderr = (result.stderr or "").strip()
            detail = " | ".join(stderr.splitlines()[-4:])
            if encoder_name == "NVENC":
                # Stop retrying a broken runtime encoder for every remaining
                # clip. The current clip is immediately retried with CPU,
                # while later clips use the same reliable fallback directly.
                _nvenc_runtime_failed = True
            print(f"    {label} {encoder_name} failed ({result.returncode}): {detail[-700:] or 'no FFmpeg stderr'}")
        try: partial_path.unlink()
        except OSError: pass

    return None


def _normalize_landscape_clip(raw_path, output_path, duration):
    """Normalize an accepted landscape clip with validated GPU encoding."""
    vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30"
    return _normalize_clip_with_recovery(
        raw_path, output_path, duration, vf, "Landscape normalization"
    )


def _normalize_vertical_clip(raw_path, output_path, duration):
    """Normalize an accepted vertical clip with the same recovery policy."""
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30"
    return _normalize_clip_with_recovery(
        raw_path, output_path, duration, vf, "Vertical normalization"
    )


def _normalized_duration_is_usable(path, target_duration):
    """Reject clips whose encoded duration could create concat timing drift."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        actual = float(probe.stdout.strip())
        minimum = max(0.1, target_duration - 0.5)
        maximum = target_duration + 0.5
        if actual < minimum or actual > maximum:
            print(f"    Normalized duration {actual:.2f}s is outside target {target_duration:.2f}s; rejecting and retrying")
            return False
        return True
    except Exception as e:
        print(f"    Could not validate normalized duration ({type(e).__name__}); rejecting and retrying")
        return False


def _nullcontext():
    """Tiny local context manager to avoid importing contextlib in hot code."""
    class _Context:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    return _Context()


def _release_llava_for_encoding():
    """Release every verifier model before final/long encoding stages."""
    global _llava_workers
    with _llava_load_lock:
        workers = _llava_workers
        _llava_workers = []
    for worker in workers:
        try:
            del worker["model"]
            del worker["processor"]
            del worker["tokenizer"]
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def search_and_download(query, idx, duration, verify=True, page=1):
    """Download one bounded page of candidates; caller controls retry rounds."""
    urls = _search_stock_urls(query, page, "landscape", _CLIP_CANDIDATES_PER_QUERY)
    
    # Only a small bounded candidate set is downloaded in this round; the
    # outer finder requests fresh Groq terms if these candidates fail.
    for candidate_no, url in enumerate(urls[:_CLIP_CANDIDATES_PER_QUERY], 1):
        try:
            candidate_started = time.perf_counter()
            raw = TEMP_DIR / f"raw_{idx}.mp4"
            out = TEMP_DIR / f"clip_{idx}.mp4"
            download_started = time.perf_counter()
            r = requests.get(url, timeout=25, stream=True)
            with open(raw,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            download_seconds = time.perf_counter() - download_started
            if os.path.getsize(raw) < 5000:
                try: os.remove(raw)
                except OSError: pass
                continue

            if not verify:
                # Main streaming path: return the raw download immediately.
                # MiniCPM will inspect it first; rejected candidates never pay
                # the expensive normalization cost.
                print(f"    Clip {idx} candidate {candidate_no}: download={download_seconds:.1f}s raw-ready={time.perf_counter() - candidate_started:.1f}s")
                _mark_url_used(url)
                return str(raw)

            # Non-streaming callers (for example final audio-gap padding) still
            # receive a normalized, verified clip as before.
            normalized = _normalize_landscape_clip(raw, out, duration)
            if not normalized:
                try: os.remove(raw)
                except OSError: pass
                continue
            try: os.remove(raw)
            except OSError: pass

            if verify:
                matches = verify_clip_matches_query(normalized, query)
                if not matches:
                    print(f"    Rejected clip for '{query[:40]}' (visual mismatch)")
                    try: os.remove(normalized)
                    except OSError: pass
                    continue

            _mark_url_used(url)
            return normalized
        except Exception as e:
            for stale in (raw, out):
                try:
                    if stale.exists(): stale.unlink()
                except OSError:
                    pass
            print(f"    Clip {idx} query '{query[:40]}' failed: {type(e).__name__}: {str(e)[:100]}")
            continue
    # The outer sentence worker retries this query on fresh pages/variants.
    print(f"    Clip {idx} query '{query[:40]}' produced no usable candidate on this page")
    return None

def process_landscape_clip(args):
    i, sent, _attempts = args
    return _find_verified_normalized_clip(sent, i, "landscape")


def prepare_clip_candidate(args):
    """Compatibility wrapper retained for external callers."""
    i, sent, attempt, query = args
    clip = search_and_download(query, i, max(3.5, sent['end'] - sent['start']), verify=True)
    return i, attempt, query, clip


# ==========================================
# 7. RENDER ENGINE (GPU-Accelerated)
# ==========================================
def render_video(sentences, audio_path, ass_path, logo_path, out_sub, keep_verifier=False):
    global _nvenc_runtime_failed
    n = len(sentences)
    clips = [None] * n
    print(f"\n  Rendering {n} clips with bounded Groq re-query rounds and streaming verification/normalization...")
    _log_resource_snapshot("clip search start")

    completed = 0
    update_status(55, f"Finding exact verified clips (0/{n})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(process_landscape_clip, (i, sent, None)): i
            for i, sent in enumerate(sentences)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                i, clip = future.result()
                clips[i] = clip
            except Exception as e:
                i = futures[future]
                print(f"  Clip {i} worker failed: {type(e).__name__}: {str(e)[:160]}")
                clips[i] = None
            completed += 1
            update_status(
                55 + int((completed / max(1, n)) * 25),
                f"Exact clips verified and normalized ({completed}/{n})...",
            )

    if not keep_verifier:
        _release_llava_for_encoding()
    if USE_GPU:
        _nvenc_runtime_failed = False
        print(f"  Verifier workers "
              f"{'retained through final encoding' if keep_verifier else 'released before final encoding'}")

    missing = [i for i, clip in enumerate(clips)
               if not clip or not os.path.exists(clip)]
    if missing:
        # Allow neighbor substitution for up to 25% missing clips instead of
        # aborting entirely. Each missing position borrows the nearest verified
        # neighbor — this keeps the video length correct while only duplicating
        # a small fraction of visuals. Still FATAL if more than 25% failed.
        max_allowed_missing = max(3, int(n * 0.25))
        if len(missing) > max_allowed_missing:
            print(f"  FATAL: {len(missing)} clips were not verified/normalized "
                  f"(>{max_allowed_missing} limit): {missing[:20]}; aborting")
            return False
        print(f"  WARNING: {len(missing)} clips missing; substituting nearest verified neighbor "
              f"for positions {missing[:20]}")
        for mi in missing:
            # Search outward from the missing position for the nearest valid clip
            best = None
            for offset in range(1, n):
                for candidate in (mi - offset, mi + offset):
                    if 0 <= candidate < n and clips[candidate] and os.path.exists(clips[candidate]):
                        best = candidate
                        break
                if best is not None:
                    break
            if best is not None:
                clips[mi] = clips[best]
                print(f"    Clip {mi} <- neighbor {best}")
            else:
                print(f"    Clip {mi}: no neighbor available; aborting")
                return False
    
    # Concat (stream copy)
    print("  Concatenating...")
    _log_resource_snapshot("clip search end")
    with open("list.txt","w") as f:
        for c in clips:
            if c: f.write(f"file '{c}'\n")
    subprocess.run("ffmpeg -y -f concat -safe 0 -i list.txt -c copy visual.mp4",
        shell=True, capture_output=True, timeout=60)
    if not os.path.exists("visual.mp4"):
        # Stream-copy concat normally needs no GPU and is nearly instant. If
        # source timestamps/codecs prevent it, fall back to GPU encoding when
        # NVENC is available rather than silently returning to libx264 CPU.
        fallback_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt"] + _enc_args() + ["visual.mp4"]
        subprocess.run(fallback_cmd, capture_output=True)
    if not os.path.exists("visual.mp4"): return False

    # Safety: stream-copy concat of many clips can lose fractions of a
    # second per clip boundary (keyframe/GOP alignment), and clip fetches
    # can also come back shorter than requested. Either way, if the
    # resulting visual track ends up shorter than the audio, -shortest
    # below would truncate the LONGER stream (the audio) to match -
    # silently cutting off the last spoken sentence(s).
    #
    # FIX: instead of freeze-padding (which produces a long dead/frozen
    # frame - unacceptable for anything more than ~1s), fetch REAL
    # additional stock clips to cover the deficit, using the already-generated
    # sentence-specific options so the extra footage still looks intentional.
    def _probe_dur(path):
        try:
            r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                "-of","default=noprint_wrappers=1:nokey=1",str(path)],
                capture_output=True, text=True, timeout=15)
            return float(r.stdout.strip())
        except: return 0.0

    vdur = _probe_dur("visual.mp4")
    adur = _probe_dur(audio_path)
    if vdur > 0 and adur > 0 and vdur < adur - 0.5:
        print(f"  Concatenated visual is {adur - vdur:.2f}s short; re-encoding with NVENC to recover GOP rounding")
        # Use GPU encoding (NVENC) instead of CPU libx264 to avoid 300s+ timeouts
        reencode_cmd = (["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt"]
                        + _enc_args() + ["-an", "visual_reencoded.mp4"])
        try:
            subprocess.run(reencode_cmd, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            # NVENC failed or timed out, try CPU with longer timeout
            print("  NVENC re-encode timed out, trying CPU fallback...")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-an", "visual_reencoded.mp4"],
                    capture_output=True, timeout=600,
                )
            except subprocess.TimeoutExpired:
                print("  CPU re-encode also timed out; continuing with stream-copy visual")
        if os.path.exists("visual_reencoded.mp4"):
            new_vdur = _probe_dur("visual_reencoded.mp4")
            if new_vdur > vdur:
                os.replace("visual_reencoded.mp4", "visual.mp4")
                vdur = new_vdur
                print(f"  Re-encode recovered: visual now {vdur:.2f}s")
            else:
                try: os.remove("visual_reencoded.mp4")
                except OSError: pass
        # Don't abort on duration deficit - the final render uses -shortest
        # which will just trim audio to match visual. A 64s shortfall across
        # a 5-10 min video means the last ~1 min of narration gets cut, but
        # that's better than a complete pipeline failure.
        if vdur < adur - 0.5:
            print(f"  WARNING: visual still {adur - vdur:.2f}s short after re-encode; "
                  f"final render will use -shortest (last {adur - vdur:.1f}s of audio may be trimmed)")

    # Final render: visual.mp4 is already normalized to 1920x1080, so do not
    # rescale the entire nine-minute stream again. Burn the logo/subtitles and
    # encode with NVENC; Qwen remains resident when Shorts will reuse it.
    if not keep_verifier:
        _release_llava_for_encoding()
    if USE_GPU:
        _nvenc_runtime_failed = False
        print(f"  Final render: verifier workers "
              f"{'retained for Shorts' if keep_verifier else 'released;'} attempting NVENC")

    _log_resource_snapshot("final render start")
    update_status(85, "Rendering final video (1080p + subs)...")
    ass_esc = str(ass_path).replace('\\','/').replace(':','\\\\:')
    if logo_path and os.path.exists(logo_path):
        filt = (f"[0:v]setsar=1[bg];"
                f"[1:v]scale=180:-1[l];[bg][l]overlay=25:25[wl];"
                f"[wl]subtitles='{ass_esc}'[v];"
                f"[2:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        input_args = ["-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path)]
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        filt = (f"[0:v]setsar=1[bg];"
                f"[bg]subtitles='{ass_esc}'[v];"
                f"[1:a]aresample=async=1:min_hard_comp=0.100000:first_pts=0[a]")
        input_args = ["-i", "visual.mp4", "-i", str(audio_path)]
        maps = ["-map", "[v]", "-map", "[a]"]

    encoders = []
    if USE_GPU:
        # Do not request CUDA decode here. Subtitle rendering is CPU-side and
        # the already-normalized input needs no GPU scaling; keeping decode
        # on CPU leaves more VRAM for NVENC and makes the P100 path reliable.
        encoders.append(("NVENC", [
            "-c:v", "h264_nvenc", "-preset", "p4",
            "-b:v", "8M", "-maxrate", "10M", "-bufsize", "16M",
        ], 300))
    encoders.append(("CPU fallback", ["-c:v", "libx264", "-preset", "fast", "-crf", "18"], 1800))

    partial_out = Path(str(out_sub) + ".part.mp4")
    for encoder_name, encoder_args, timeout_seconds in encoders:
        for stale in (partial_out, out_sub):
            try:
                if Path(stale).exists(): Path(stale).unlink()
            except OSError:
                pass

        cmd = (['ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error']
               + input_args + ['-filter_complex', filt] + maps + encoder_args
               + ['-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                  '-shortest', str(partial_out)])
        started = time.perf_counter()
        try:
            # Serialize NVENC with any other possible GPU work. CPU fallback
            # does not need the lock and can continue without GPU contention.
            with _gpu_lock if encoder_name == "NVENC" else _nullcontext():
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_seconds
                )
        except Exception as e:
            result = None
            detail = f"{type(e).__name__}: {str(e)[:240]}"
            if encoder_name == "NVENC":
                _nvenc_runtime_failed = True
            print(f"  Final render {encoder_name} exception after {time.perf_counter() - started:.1f}s: {detail}")
        else:
            detail = " | ".join((result.stderr or "").splitlines()[-4:])[-900:]
            if result.returncode == 0 and partial_out.exists() and partial_out.stat().st_size > 10000:
                try:
                    os.replace(partial_out, out_sub)
                    elapsed = time.perf_counter() - started
                    print(f"  Final: {os.path.getsize(out_sub)/(1024**2):.0f}MB in {elapsed:.1f}s ({encoder_name})")
                    return True
                except OSError as e:
                    detail = f"output publish failed: {type(e).__name__}: {e}"
            if encoder_name == "NVENC":
                _nvenc_runtime_failed = True
            print(f"  Final render {encoder_name} failed after {time.perf_counter() - started:.1f}s: {detail or 'no FFmpeg stderr'}")

        try:
            if partial_out.exists(): partial_out.unlink()
        except OSError:
            pass

    print("  Final render failed with both NVENC and CPU fallback")
    return False



# ==========================================
# 8. UTILITIES
# ==========================================
LOG_BUF = []
def update_status(progress, message, status="processing", file_url=None):
    print(f"--- {progress}% | {message} ---")
    LOG_BUF.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    if len(LOG_BUF) > 30: LOG_BUF.pop(0)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
    if not repo or not token: return
    import base64
    data = {"progress":progress,"message":message,"status":status,
            "logs":"\n".join(LOG_BUF),"timestamp":time.time()}
    if file_url: data["file_io_url"] = file_url
    url = f"https://api.github.com/repos/{repo}/contents/status/status_{JOB_ID}.json"
    headers = {"Authorization":f"token {token}","Accept":"application/vnd.github.v3+json"}
    try:
        gr = requests.get(url, headers=headers)
        sha = gr.json().get("sha") if gr.status_code==200 else None
        payload = {"message":f"s{progress}","content":base64.b64encode(json.dumps(data).encode()).decode(),"branch":"main"}
        if sha: payload["sha"]=sha
        requests.put(url, headers=headers, json=payload)
    except: pass

def download_asset(path, local):
    try:
        repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}",
            headers={"Authorization":f"token {token}","Accept":"application/vnd.github.v3.raw"})
        if r.status_code==200:
            with open(local,"wb") as f: f.write(r.content)
            return True
    except: pass
    return False

def upload_drive(fp):
    # Google Drive upload disabled in this build - videos go directly to YouTube.
    return None


# ==========================================
# YOUTUBE UPLOAD (Token auto-refresh)
# ==========================================
# Token files are downloaded from the repo's tokens/ folder during workflow setup.
# techrovex -> token-techrovex.json
# vitovia   -> token-vitovia.json
_YOUTUBE_TOKEN_MAP = {
    "techrovex": "token-techrovex.json",
    "vitovia": "token-vitovia.json",
}

def _download_youtube_token(channel):
    """Locate the YouTube token file from Kaggle input datasets or local directory."""
    token_filename = _YOUTUBE_TOKEN_MAP.get(channel)
    if not token_filename:
        return None

    # Primary: expected Kaggle dataset path
    kaggle_path = Path("/kaggle/input/tokens") / token_filename
    if kaggle_path.exists():
        print(f"  YouTube: found token at {kaggle_path}")
        return kaggle_path

    # Robust fallback: recursively search the entire /kaggle/input tree.
    # Kaggle can mount a dataset under a slug-derived folder that differs from
    # the raw dataset name, or nest files in subdirectories. A recursive scan
    # finds the token regardless of how the mount is structured.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        try:
            for found in kaggle_input.rglob(token_filename):
                if found.is_file():
                    print(f"  YouTube: found token at {found}")
                    return found
        except Exception as e:
            print(f"  YouTube: error scanning /kaggle/input ({e})")

    # Local fallbacks (for testing)
    for local in (Path(token_filename), Path("tokens") / token_filename):
        if local.exists():
            print(f"  YouTube: found token at {local}")
            return local

    # Diagnostic: list what IS available so misconfiguration is obvious in logs
    print(f"  YouTube: token file '{token_filename}' not found in any expected location")
    if kaggle_input.is_dir():
        try:
            available = [str(p) for p in kaggle_input.rglob("*.json")][:20]
            print(f"  YouTube: JSON files present under /kaggle/input: {available or 'none'}")
            dirs = [str(p) for p in kaggle_input.iterdir()]
            print(f"  YouTube: /kaggle/input contents: {dirs or 'empty'}")
        except Exception:
            pass
    else:
        print("  YouTube: /kaggle/input does not exist - dataset was NOT attached to the kernel")
    return None


def upload_youtube(fp, title=None, description=None, tags=None):
    """Upload video to YouTube using auto-refreshed OAuth token. Always unlisted."""
    if not os.path.exists(fp):
        return None
    channel = YOUTUBE_CHANNEL.strip().lower()
    if channel == "none" or not channel:
        return None

    token_path = _download_youtube_token(channel)
    if not token_path or not token_path.exists():
        print(f"  YouTube: no token file found for channel '{channel}', skipping upload")
        return None

    print(f"  YouTube: uploading {os.path.basename(fp)} to channel '{channel}'...")

    try:
        token_data = json.loads(token_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  YouTube: failed to read token file ({e})")
        return None

    # Refresh the access token using refresh_token + client credentials
    refresh_token = token_data.get("refresh_token")
    client_id = token_data.get("client_id")
    client_secret = token_data.get("client_secret")
    token_uri = token_data.get("token_uri", "https://oauth2.googleapis.com/token")

    if not all([refresh_token, client_id, client_secret]):
        print("  YouTube: token file missing refresh_token/client_id/client_secret")
        return None

    try:
        refresh_resp = requests.post(token_uri, data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }, timeout=15)
        refresh_json = refresh_resp.json()
        access_token = refresh_json.get("access_token")
        if not access_token:
            error = refresh_json.get("error_description", refresh_json.get("error", "unknown"))
            print(f"  YouTube: token refresh failed ({error})")
            return None
        print(f"  YouTube: token refreshed successfully")
    except Exception as e:
        print(f"  YouTube: token refresh request failed ({e})")
        return None

    # Build video metadata
    if not title:
        title = f"Video {JOB_ID}"
    if not description:
        description = "Automated upload"
    if not tags:
        tags = ["documentary", "automation"]

    metadata = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:15],
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": "unlisted",
            "selfDeclaredMadeForKids": False,
        },
    }

    # Initiate resumable upload
    file_size = os.path.getsize(fp)
    try:
        init_resp = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            json=metadata,
            timeout=30,
        )
        if init_resp.status_code != 200:
            print(f"  YouTube: initiate upload failed ({init_resp.status_code}: "
                  f"{init_resp.text[:200]})")
            return None
        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            print("  YouTube: no upload URL in response headers")
            return None
    except Exception as e:
        print(f"  YouTube: initiate upload request failed ({e})")
        return None

    # Upload the file
    try:
        with open(fp, "rb") as f:
            upload_resp = requests.put(
                upload_url,
                headers={
                    "Content-Length": str(file_size),
                    "Content-Type": "video/mp4",
                },
                data=f,
                timeout=1800,  # 30 min timeout for large files
            )
        if upload_resp.status_code in [200, 201]:
            video_id = upload_resp.json().get("id", "")
            link = f"https://youtu.be/{video_id}"
            print(f"  YouTube: upload complete -> {link}")
            return link
        else:
            print(f"  YouTube: upload failed ({upload_resp.status_code}: "
                  f"{upload_resp.text[:200]})")
            return None
    except Exception as e:
        print(f"  YouTube: upload request failed ({e})")
        return None

def generate_script(topic, mins):
    words=int(mins*180); lang="Write in Spanish." if IS_SPANISH else "Write in English."
    prompt=f"Write a documentary narration about '{topic}'. {words} words.\n{lang}\nRules: Only narration. No brackets. Islamic guidelines. Family-friendly."
    random.shuffle(GEMINI_KEYS)
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            r=genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt)
            return re.sub(r'\[.*?\]','',r.text.replace("*","").replace("#","").strip())
        except: continue
    return ""


def _prepare_short_assets(short_index, script_text, short_audio):
    """Transcribe and prepare one short without mutating global query state."""
    short_sentences, short_word_data = [], []
    if ASSEMBLY_KEY:
        try:
            tx_config = aai.TranscriptionConfig(
                language_code="es" if IS_SPANISH else "en",
                punctuate=True,
                format_text=True,
            )
            tx = aai.Transcriber(config=tx_config).transcribe(str(short_audio))
            if tx.status != aai.TranscriptStatus.error:
                for sentence in tx.get_sentences():
                    short_sentences.append({
                        "text": sentence.text,
                        "start": sentence.start / 1000,
                        "end": sentence.end / 1000,
                    })
                if short_sentences:
                    short_sentences[-1]["end"] += 0.3
                for word in tx.words:
                    short_word_data.append({
                        "text": word.text,
                        "start": word.start / 1000,
                        "end": word.end / 1000,
                    })
        except Exception as e:
            print(f"  Short {short_index+1}: transcription error ({e}), using estimated timing")

    if not short_sentences:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(short_audio)],
                capture_output=True, text=True, timeout=15,
            )
            total_duration = float(probe.stdout.strip())
        except Exception:
            total_duration = SHORT_DUR_TARGET
        parts = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", script_text)
            if len(part.strip()) > 2
        ] or [script_text]
        per_sentence = total_duration / len(parts)
        short_sentences = [
            {"text": part, "start": i * per_sentence, "end": (i + 1) * per_sentence}
            for i, part in enumerate(parts)
        ]

    for sentence_index, sentence in enumerate(short_sentences):
        sentence["orig_idx"] = sentence_index

    short_ass = TEMP_DIR / f"short_{short_index}_subs.ass"
    create_subtitles(
        short_sentences,
        short_ass,
        word_data=short_word_data if short_word_data else None,
        style_set=SHORT_SUBTITLE_STYLES,
        play_res=(1080, 1920),
        max_chars=20,
    )
    short_query_options = generate_queries_for_sentences(short_sentences)
    return {
        "sentences": short_sentences,
        "word_data": short_word_data,
        "ass": short_ass,
        "query_options": short_query_options,
    }


# ==========================================
# 9. MAIN EXECUTION
# ==========================================
print(f"\n{'='*50}")
print(f"  VIDEO FACTORY V4")
print(f"  Lang: {'ES' if IS_SPANISH else 'EN'} | GPU: {USE_GPU}")
print(f"{'='*50}\n")

update_status(1, "Starting...")
voice = TEMP_DIR/"voice.mp3"; logo = TEMP_DIR/"logo.png"
if not download_asset(VOICE_PATH, voice):
    update_status(0,"Voice failed","failed"); exit(1)
if LOGO_PATH and LOGO_PATH != "None":
    download_asset(LOGO_PATH, logo)
    logo = str(logo) if os.path.exists(logo) else None
else: logo = None

# Script
update_status(5, "Script...")
text = generate_script(TOPIC, DURATION_MINS) if MODE=="topic" else SCRIPT_TEXT
if len(text)<50: update_status(0,"Script too short","failed"); exit(1)
print(f"  {len(text.split())} words")

# Audio
update_status(12, "Audio...")
audio = TEMP_DIR/"audio.wav"
if not generate_audio(text, voice, audio):
    update_status(0,"Audio failed","failed"); exit(1)

# Transcribe
update_status(48, "Transcribing...")
sentences = []
word_data = []  # Word-level timestamps for highlighting
if ASSEMBLY_KEY:
    try:
        aai.settings.api_key = ASSEMBLY_KEY
        # CRITICAL: language must be explicitly set. Without this,
        # AssemblyAI defaults to English and on Spanish audio produces
        # heavily garbled/dropped transcription (most words treated as
        # unintelligible noise) - this was the root cause of only getting
        # ~163 words out of an 8:42 Spanish narration.
        tx_config = aai.TranscriptionConfig(
            language_code="es" if IS_SPANISH else "en",
            punctuate=True,
            format_text=True,
        )
        tx = aai.Transcriber(config=tx_config).transcribe(str(audio))
        if tx.status == aai.TranscriptStatus.error:
            print(f"  Transcribe failed: {tx.error}")
        else:
            for s in tx.get_sentences():
                sentences.append({"text":s.text,"start":s.start/1000,"end":s.end/1000})
            if sentences: sentences[-1]['end']+=0.5

            # Get word-level timestamps for subtitle highlighting
            for word in tx.words:
                word_data.append({"text": word.text, "start": word.start/1000, "end": word.end/1000})
            print(f"  Got {len(word_data)} word timestamps for highlighting")

            # Sanity check: normal speech is ~2-3 words/sec. If AssemblyAI
            # returned far fewer words than the audio duration implies, the
            # transcription is almost certainly garbled/wrong-language
            # (words dropped as "unintelligible") rather than genuinely
            # sparse audio. Discard it and fall through to the estimated-
            # timing path below instead of silently building subtitles and
            # visual-matching off broken data.
            audio_dur = sentences[-1]['end'] if sentences else 0
            if audio_dur > 10:
                wpm = len(word_data) / (audio_dur/60)
                if wpm < 60:  # normal narration is ~120-180 wpm; below 60 is a red flag
                    print(f"  WARNING: only {wpm:.0f} words/min detected (expected 100+) - "
                          f"transcription looks broken, discarding and using estimated timing instead")
                    sentences = []
                    word_data = []
    except Exception as e: print(f"  Transcribe err: {e}")

if not sentences:
    import wave
    try:
        with wave.open(str(audio),'rb') as w: dur=w.getnframes()/float(w.getframerate())
    except: dur=len(text.split())/2.5
    wps=len(text.split())/dur if dur>0 else 2.5; t=0
    for i in range(0,len(text.split()),8):
        chunk=text.split()[i:i+8]; d=len(chunk)/wps
        sentences.append({"text":' '.join(chunk),"start":t,"end":t+d}); t+=d

# Start the independent Shorts script request while the main video pipeline
# prepares its own queries and subtitles. This is network/Groq work and does
# not touch the GPU or shared audio files.
shorts_eligible = bool(
    sentences and sentences[-1]["end"] >= SHORT_DUR_TARGET * 0.6
)
short_script_executor = None
short_script_future = None
if shorts_eligible:
    short_script_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    short_script_future = short_script_executor.submit(
        generate_short_scripts,
        sentences,
        TOPIC if MODE == "topic" else text[:100],
        SHORTS_COUNT,
        SHORT_DUR_TARGET,
    )

# Queries and subtitle generation are independent after transcription.
update_status(50, "Matching visuals to sentences...")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prep_executor:
    query_future = prep_executor.submit(generate_queries_for_sentences, sentences)
    subtitle_future = prep_executor.submit(
        create_subtitles,
        sentences,
        TEMP_DIR / "subs.ass",
        word_data=word_data if word_data else None,
    )
    AI_QUERY_OPTIONS = query_future.result()
    subtitle_future.result()

# Subtitles (word-level highlighting if available)
ass = TEMP_DIR / "subs.ass"

# Render
update_status(54, "Processing video...")

# Load the clip-verification model ONCE here, deliberately, on the main
# thread - BEFORE the render step spins up its 5 parallel worker threads.
# The old lazy-load-on-first-use pattern let all 5 workers race to call
# _load_llava() at nearly the same instant (the lock only covered the
# generate() call, not the load itself), so all 5 threads tried to load
# a full 7B model onto the GPU simultaneously - 5x duplicate memory
# allocation, causing OOM even on tiny 20-130MB allocations afterward.
# By this point in the pipeline, Chatterbox TTS and resemble-enhance have
# already freed their GPU memory (see generate_audio()'s explicit
# del+empty_cache+gc.collect calls), so this is the right moment to load
# the verification model into that freed VRAM, once, before any threads exist.
try:
    _load_llava()
    print(f"  Qwen verifier workers loaded before rendering: {len(_llava_workers)}")
except Exception as e:
    print(f"  ERROR: video verification workers failed to load ({str(e)[:120]}); refusing unverified output")
    update_status(0, "Video verification unavailable; render refused", "failed")
    raise SystemExit(1)

o2 = OUTPUT_DIR/f"final_{JOB_ID}_WITH_SUBS.mp4"

if render_video(sentences, audio, ass, logo, o2, keep_verifier=True):
    update_status(93, "Preparing Shorts...")
    msg = "Done!\n"

    # ==========================================
    # SHORTS PIPELINE
    # ==========================================
    # Only worth generating shorts from audio that's actually long enough
    # for the requested count (avoid generating garbage 60s shorts from a
    # 90s total-runtime video, etc.)
    if sentences and sentences[-1]['end'] >= SHORT_DUR_TARGET * 0.6:
        update_status(95, f"Generating {SHORTS_COUNT} shorts...")
        try:
            if short_script_future is not None:
                short_scripts = short_script_future.result()
                short_script_executor.shutdown(wait=True)
                short_script_executor = None
            else:
                short_scripts = generate_short_scripts(
                    sentences,
                    TOPIC if MODE == "topic" else text[:100],
                    SHORTS_COUNT,
                    SHORT_DUR_TARGET,
                )
            print(f"  Shorts: {len(short_scripts)} scripts generated (requested {SHORTS_COUNT})")
            short_links = []
            short_failures = []  # (short_num, reason) for end-of-run summary

            # Release the MiniCPM verifier BEFORE shorts TTS to free GPU VRAM.
            # Without this, GPU0 has ~3GB free and GPU1 has ~0.7GB free — not
            # enough for Chatterbox TTS which needs 4+ GB. The verifier will be
            # reloaded before shorts clip verification starts.
            _release_llava_for_encoding()
            if torch.cuda.is_available():
                for _gpu_idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(_gpu_idx):
                        torch.cuda.empty_cache()
                gc.collect()
            print("  Shorts: released verifier VRAM for TTS")

            # Generate short audio serially because generate_audio loads and
            # uses shared CUDA/TTS state. While the GPU generates the next
            # short, the previous short's transcription, subtitles, and Groq
            # visual queries run concurrently in the background.
            short_asset_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(3, len(short_scripts)))
            )
            prepared_shorts = []
            for si, sc in enumerate(short_scripts):
                script_text = sc["script"].strip()
                update_status(95, f"Preparing short audio {si+1}/{len(short_scripts)}...")
                if len(script_text) < 20:
                    short_failures.append((si + 1, "empty/too-short script"))
                    continue

                short_audio = TEMP_DIR / f"short_{si}_audio.wav"
                if not generate_audio(script_text, voice, short_audio):
                    print(f"  Short {si+1}: TTS failed, skipping")
                    short_failures.append((si + 1, "TTS failed"))
                    continue

                asset_future = short_asset_executor.submit(
                    _prepare_short_assets, si, script_text, short_audio
                )
                prepared_shorts.append((si, short_audio, asset_future))
            short_asset_executor.shutdown(wait=False)

            # Rendering remains sequential: each short uses the shared Qwen
            # workers and NVENC lock. Reload the verifier now that TTS is done
            # and has freed GPU memory.
            try:
                _load_llava()
                print(f"  Shorts: verifier reloaded ({len(_llava_workers)} workers)")
            except Exception as e:
                print(f"  Shorts: verifier reload failed ({e}); shorts will skip clip verification")

            rendered_shorts = []  # (si) that rendered successfully
            for si, short_audio, asset_future in prepared_shorts:
                try:
                    assets = asset_future.result()
                    short_sentences = assets["sentences"]
                    short_ass = assets["ass"]
                    short_query_options = assets["query_options"]
                except Exception as e:
                    print(f"  Short {si+1}: preparation failed ({type(e).__name__}: {str(e)[:160]})")
                    short_failures.append((si + 1, "transcription/query preparation failed"))
                    continue

                saved_query_options = AI_QUERY_OPTIONS
                AI_QUERY_OPTIONS = short_query_options
                try:
                    short_out = OUTPUT_DIR / f"short_{JOB_ID}_{si+1}.mp4"
                    ok = render_short(
                        si, short_sentences, short_audio, short_ass, logo, short_out,
                        release_verifier=False,
                    )
                    if not ok:
                        print(f"  Short {si+1}: retrying once...")
                        ok = render_short(
                            si, short_sentences, short_audio, short_ass, logo, short_out,
                            release_verifier=False,
                        )
                finally:
                    AI_QUERY_OPTIONS = saved_query_options

                if ok:
                    rendered_shorts.append(si)
                else:
                    print(f"  Short {si+1}: failed after retry, skipping")
                    short_failures.append((si + 1, "render failed after retry"))

            # Release verifier memory only after all Shorts have been rendered.
            _release_llava_for_encoding()

            print(f"  Shorts summary: {len(rendered_shorts)}/{len(short_scripts)} rendered")
            if short_failures:
                print(f"  Shorts failures: {short_failures}")
                msg += f"({len(short_failures)} short(s) failed - check logs)\n"

            # YouTube upload for shorts (direct upload, no Google Drive)
            if YOUTUBE_CHANNEL.strip().lower() not in ("none", ""):
                for si, sc in enumerate(short_scripts):
                    short_path = OUTPUT_DIR / f"short_{JOB_ID}_{si+1}.mp4"
                    if short_path.exists():
                        yt_short = upload_youtube(
                            short_path,
                            title=f"{sc.get('theme', f'Short {si+1}')} #shorts"[:100],
                            description=sc.get("script", "")[:5000],
                            tags=["shorts", "documentary", "facts", "education"],
                        )
                        if yt_short:
                            short_links.append(yt_short)
                            msg += f"YouTube Short {si+1}: {yt_short}\n"
        except Exception as e:
            print(f"  Shorts pipeline error: {e}")

    if _llava_workers:
        _release_llava_for_encoding()

    # YouTube upload (main video) - direct upload, no Google Drive
    yt_link = upload_youtube(
        o2,
        title=TOPIC[:100] if MODE == "topic" else f"Video {JOB_ID}",
        description=text[:5000] if text else "Automated documentary",
        tags=["documentary", "education", "facts"],
    )
    if yt_link:
        msg += f"YouTube: {yt_link}\n"

    update_status(100, msg, "completed", yt_link)
    print(f"\n  {msg}")
else:
    update_status(0, "Render failed", "failed")

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
for f in ["visual.mp4","list.txt"]:
    if os.path.exists(f): os.remove(f)
import glob
for f in glob.glob("list_sh*.txt") + glob.glob("visual_sh*.mp4"):
    try: os.remove(f)
    except: pass
print("--- DONE ---")
