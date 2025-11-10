FROM intel/oneapi-basekit

ARG DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# System Packages & Build Tools
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    gpg \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Intel GPU Drivers & Media Stack
# ==============================================================================
RUN apt-get update \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        libze-intel-gpu1 \
        libze1 \
        libze-dev \
        intel-opencl-icd \
        intel-ocloc \
        clinfo \
        intel-media-va-driver-non-free \
        libmfx-gen1 \
        libvpl2 \
        libvpl-tools \
        libva-glx2 \
        va-driver-all \
        vainfo \
        intel-metrics-discovery \
        intel-gsc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Intel XPU Manager
# ==============================================================================
RUN groupadd xpum \
    && useradd -M -s /bin/sh -g xpum xpum \
    && wget -qO xpumanager.deb \
        https://github.com/intel/xpumanager/releases/download/V1.3.1/xpumanager_1.3.1_20250724.061629.60921e5e_u24.04_amd64.deb \
    && dpkg --unpack xpumanager.deb \
    && rm -f /var/lib/dpkg/info/xpumanager.postinst \
    && dpkg --configure xpumanager \
    && apt-get install -f -y \
    && chown -R xpum:xpum /usr/lib/xpum \
    && chmod g+x /usr/lib/xpum/keytool.sh /usr/lib/xpum/enable_restful.sh \
    && rm -f xpumanager.deb

# ==============================================================================
# Python Dependencies
# ==============================================================================
RUN pip3 install --break-system-packages \
    opencv-python \
    nncf \
    ultralytics

RUN pip3 install --break-system-packages --pre -U \
    openvino \
    openvino-dev \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

# ==============================================================================
# Prepare Models
# ==============================================================================
WORKDIR /opt/models

COPY ./utils/prepare_models.sh /tmp/prepare_models.sh

RUN /tmp/prepare_models.sh \
    && rm -f /tmp/prepare_models.sh \
    && omz_downloader --name person-detection-0303 \
    && mv intel/* . \
    && rmdir intel

# ==============================================================================
# Entrypoint Script
# ==============================================================================
COPY <<'EOF' /usr/bin/entrypoint.sh
#!/bin/bash
set -e

# Start XPU Manager daemon in background
/usr/bin/xpumd -p /var/xpum_daemon.pid -d /usr/lib/xpum/dump > /dev/null 2>&1 &

# Execute the main command
exec "$@"
EOF

RUN chmod +x /usr/bin/entrypoint.sh

# ==============================================================================
# Container Configuration
# ==============================================================================
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/entrypoint.sh"]
CMD ["bash"]
