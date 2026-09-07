FROM python:3.10-slim-bookworm

# ── System dependencies for DSSAT / gym-dssat-pdi build ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    build-essential \
    gfortran \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── PDI repo (required dependency of gym-dssat-pdi) ─────────────────────────
RUN echo "deb [ arch=amd64 ] https://repo.pdi.dev/debian bookworm main" \
        | tee /etc/apt/sources.list.d/pdi.list > /dev/null && \
    wget -O /etc/apt/trusted.gpg.d/pdidev-archive-keyring.gpg \
        https://repo.pdi.dev/debian/pdidev-archive-keyring.gpg && \
    chmod a+r /etc/apt/trusted.gpg.d/pdidev-archive-keyring.gpg \
               /etc/apt/sources.list.d/pdi.list && \
    apt-get update && apt-get install -y pdidev-archive-keyring && \
    rm -rf /var/lib/apt/lists/*

# ── gym-dssat-pdi repo ───────────────────────────────────────────────────────
RUN mkdir -p /usr/local/share/keyrings && \
    wget -O- https://xlab.udc.gal/gym-dssat/gym-dssat-archive-keyring.gpg \
        | tee /usr/local/share/keyrings/gym-dssat-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/local/share/keyrings/gym-dssat-archive-keyring.gpg] https://xlab.udc.gal/gym-dssat bookworm main dev" \
        | tee /etc/apt/sources.list.d/gym-dssat.list > /dev/null && \
    chmod a+r /usr/local/share/keyrings/gym-dssat-archive-keyring.gpg && \
    apt-get update && \
    apt-get install -y \
        -o Acquire::Retries=6 \
        -o Acquire::http::Timeout=180 \
        -o Acquire::https::Timeout=180 \
        --fix-missing \
        gym-dssat-pdi && \
    rm -rf /var/lib/apt/lists/*
# NOTE: xlab.udc.gal (Universidade da Coruña) hosts the gym-dssat-pdi
# package itself, a ~36 MB .deb. That server has been observed timing
# out mid-transfer on this specific package (unrelated to any repo/config
# issue -- everything else in this RUN resolves and downloads fine).
# The retry/timeout/--fix-missing flags above give apt more room to
# recover from a slow or interrupted connection rather than failing the
# whole build on one dropped transfer.

# ── Python ML stack, installed INTO gym-dssat-pdi's own venv ────────────────
# gym-dssat-pdi ships its own virtualenv at /opt/gym_dssat_pdi, built against
# the exact DSSAT/PDI binary it installed. Anything installed with the
# system pip (/usr/local/bin/pip) lives in a separate site-packages tree and
# is invisible to /opt/gym_dssat_pdi/bin/python -- this is the root cause of
# the "which crop_utils.py is actually loaded" confusion from past sessions.
# Installing here, once, and then making this python the ONLY python on PATH
# removes that footgun entirely.
#
# NOTE: numpy is intentionally left unpinned -- it's whatever version
# gym-dssat-pdi's venv was built against. compare_results.py avoids scipy
# and uses math.erf specifically because forcing a newer numpy/scipy pair
# broke compatibility with this venv's numpy. Do not "fix" that by pinning
# numpy here; if scipy is ever needed, resolve the numpy/scipy pair
# deliberately and update compare_results.py's guard accordingly.
RUN /opt/gym_dssat_pdi/bin/pip install --no-cache-dir \
        --default-timeout=120 --retries 5 \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    /opt/gym_dssat_pdi/bin/pip install --no-cache-dir \
        --default-timeout=120 --retries 5 \
        numpy==1.24.1 \
        gymnasium \
        DSSATTools \
        requests \
        pandas==1.5.3
# NOTE: matplotlib is deliberately NOT in this pip install list.
# gym-dssat-pdi's apt install (a few RUN steps up) already brings in
# matplotlib==3.3.4 as a precompiled Debian package via apt -- that's a
# real, hard dependency of gym-dssat-pdi (`pip check` fails without
# exactly this version). matplotlib 3.3.4 predates Python 3.11 and has
# no PyPI wheel for cp311, so the moment pip is asked to install/reinstall
# it, pip ignores the apt-installed copy and tries to compile from source
# -- which fails immediately on a missing Python.h (we never installed
# python3-dev/python3.11-dev, since nothing else needs it). Do not add
# matplotlib to this pip install list, with or without a version pin, and
# do not run `pip install matplotlib` by hand inside a running container
# either -- both silently corrupt the apt-managed installation in a way
# that can't be undone by pip itself (no wheel exists to reinstall from).
# If matplotlib ever needs to change, do it through apt
# (`apt-get install --reinstall matplotlib`-equivalent for whatever
# package provides it), never through pip.
# NOTE: plain `pip install torch` now pulls the CUDA-enabled build by
# default, which drags in nvidia-cudnn-cu13 (~366 MB), cuda-toolkit, and
# several other NVIDIA runtime packages -- several hundred extra MB this
# container has no use for unless a GPU is actually passed through to it.
# That's what timed out above (125/366 MB into nvidia-cudnn-cu13). The
# PyTorch project publishes a CPU-only wheel index specifically to avoid
# this; using it here cuts the download to a fraction of the size and
# removes GPU-only dependencies that would never load on a CPU-only host
# anyway. If you later run this on a machine with an NVIDIA GPU and want
# to use it, swap this back to plain `pip install torch` (or the matching
# cu12x index URL) deliberately, rather than by accident.

# Make the gym-dssat-pdi venv's python/pip the default `python`/`pip` in
# this container, so `python main.py`, `pip install`, and
# `python -c "import crop_utils; print(crop_utils.__file__)"` all resolve
# to the same interpreter without needing the full venv path spelled out
# every time.
ENV PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:${PATH}"
# NOTE: AC's dssat_configuration.py passes run_dssat_location as an
# explicit full path ("/opt/dssat_pdi/run_dssat"), so it never needed
# this. REINFORCE's main.py calls gym.make(...) without that kwarg,
# so gym_dssat_pdi falls back to its default value -- the bare name
# "run_dssat" -- and resolves it via shutil.which(), which only
# searches PATH. Adding /opt/dssat_pdi to PATH makes that bare-name
# lookup succeed too, so both projects work without editing REINFORCE's
# main.py to hardcode a path.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Writable output directories ─────────────────────────────────────────────
# Past sessions hit CSV permission errors when training_log.csv /
# daily_log.csv / checkpoints/ were created by one UID and then accessed
# via `docker cp` or a bind mount from a different host UID. Pre-creating
# these with open permissions avoids that class of failure without needing
# to match host UID/GID at build time.
RUN mkdir -p /app/checkpoints /app/outputs && \
    chmod -R 777 /app/checkpoints /app/outputs

COPY . /app

CMD ["sleep", "infinity"]