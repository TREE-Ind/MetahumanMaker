"""Install optional local inference dependencies and download public Google models."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]
models = ROOT / "models"
models.mkdir(exist_ok=True)
environment = ROOT / ".venv"
if not environment.exists():
    venv.EnvBuilder(with_pip=True).create(environment)
python = environment / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
)
subprocess.run(
    [str(python), "-m", "pip", "install", "mediapipe==1.1.1", "h5py==3.16.0", "scipy==1.18.0", "pillow==13.0.0"],
    check=True,
)
urls = {
    "gnm_head.npz": "https://huggingface.co/google/gnm-v3/resolve/main/v3_0/gnm_head.npz",
    "head_sparse_68.txt": "https://raw.githubusercontent.com/google/GNM/main/gnm/shape/data/landmarks/head_sparse_68.txt",
    "identity_decoder_model.h5": "https://raw.githubusercontent.com/google/GNM/main/gnm/shape/data/semantic_sampler/identity_decoder_model.h5",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "GNM-LICENSE": "https://raw.githubusercontent.com/google/GNM/main/LICENSE",
}
records = []
for filename, url in urls.items():
    output = models / filename
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".download")
        print("Downloading", filename, flush=True)
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(output)
    records.append(
        {
            "file": filename,
            "source": url,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
(models / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
subprocess.run([str(python), str(ROOT / "tools/convert_gnm_decoder.py")], check=True)
(models / "python-requirements.txt").write_text(
    subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True),
    encoding="utf-8",
)
print("LOCAL_MODELS_READY", python, flush=True)
