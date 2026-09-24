#!/bin/bash
# One-time environment setup for the DISI cluster: creates a venv and installs
# requirements.txt. Run this directly on giano.cs.unibo.it (not via sbatch - no
# GPU or compute node is needed just to install packages).
#
# Requires Python >= 3.12 (see requirements.txt: numpy 2.5 needs it; matplotlib,
# torch, torchvision, lightning and wandb need >= 3.10/3.11). Check what's
# available on giano BEFORE running this:
#   python3 --version
# If it's older than 3.12, find a versioned binary (python3.12, python3.13, ...)
# or a `module load` command for it and set PYTHON_BIN below - not something
# this script can guess.

set -euo pipefail

# The home quota is 400 MB (see cluster instructions); a venv with torch +
# torchvision is easily several GB, so it MUST live under /scratch.hpc/, never
# in the home directory.
VENV_DIR=/scratch.hpc/matteo.preda/industry/indus/venv
PYTHON_BIN=python3   # TODO: change if `python3 --version` on giano is < 3.12

# cd to the directory this script lives in, so requirements.txt is found
# regardless of where the script is invoked from.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"

echo "Python: $(${PYTHON_BIN} --version)"
echo "Creating venv at ${VENV_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")"
${PYTHON_BIN} -m venv "${VENV_DIR}"

# --no-cache-dir everywhere: pip's default cache lives in the home quota (400 MB)
# and fills it fast (see cluster instructions). Run `pip3 cache purge` first if
# an earlier install without --no-cache-dir already ate into the quota.
"${VENV_DIR}/bin/pip" install --upgrade pip --no-cache-dir

echo "Installing requirements.txt"
"${VENV_DIR}/bin/pip" install -r requirements.txt --no-cache-dir

# --- CUDA compatibility: unresolved, verify before relying on this for training ---------
# requirements.txt pins torch==2.14.0 from plain PyPI, which bundles its own CUDA
# runtime. The cluster instructions' own pytorch example installs from the cu118
# wheel index instead (https://download.pytorch.org/whl/cu118), which implies the
# node driver (Nvidia 535 / CUDA 11.8 libraries per the cluster doc) may not
# support whatever CUDA runtime PyPI's torch 2.14.0 bundles. This has not been
# verified either way, and torch 2.14.0 may not even publish a cu118 build (recent
# torch releases often drop old CUDA indexes). Do NOT silently switch the index or
# downgrade the pinned version here - check on an actual GPU node first:
#   ${VENV_DIR}/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# If that prints False or errors, come back and pick a working torch build
# together instead of guessing an index URL that could install an unpinned version.

echo ""
echo "Done. Activate with:   source ${VENV_DIR}/bin/activate"
echo "Or call directly:      ${VENV_DIR}/bin/python3 ..."
