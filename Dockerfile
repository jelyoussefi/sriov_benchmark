FROM intel/oneapi-basekit

ARG DEBIAN_FRONTEND=noninteractive

#--------------------------------------------------------------------------------------------------------------------------
#                                           ESSENTIAL SYSTEM PACKAGES & TOOLS                         
#--------------------------------------------------------------------------------------------------------------------------

RUN apt update -y && apt install -y --no-install-recommends \
    software-properties-common build-essential wget gpg  git  \
    python3-pip  && \
    rm -rf /var/lib/apt/lists/*

#--------------------------------------------------------------------------------------------------------------------------
#                                               INSTALL INTEL GPU DRIVERS 
#--------------------------------------------------------------------------------------------------------------------------
RUN apt-get update \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update || true \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        libze-intel-gpu1 libze1 libze-dev \
        intel-opencl-icd intel-ocloc clinfo \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools \
        libva-glx2 va-driver-all vainfo \
        intel-metrics-discovery intel-gsc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    opencv-python \
    nncf \
    ultralytics 

RUN pip3 install --break-system-packages --pre -U \
    openvino openvino-dev \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    
#--------------------------------------------------------------------------------------------------------------------------
#                                               PREPARE MODLES
#--------------------------------------------------------------------------------------------------------------------------
COPY ./utils/prepare_models.sh /tmp/
WORKDIR /opt/models/
RUN /tmp/prepare_models.sh
RUN rm -f /tmp/prepare_models.sh
RUN omz_downloader --name person-detection-0303
RUN mv intel/* .
#--------------------------------------------------------------------------------------------------------------------------

WORKDIR /workspace
