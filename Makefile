#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=vf_container_image
export DOCKER_BUILDKIT=1

NUM_VFS ?= 2 
VF_NUM ?= 1

MODEL ?= /opt/models/yolo11s/FP16/yolo11s.xml
DURATION ?= 10

DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr \
	--privileged  \
	-e ONEAPI_DEVICE_SELECTOR=level_zero:${VF_NUM} \
	-v ${CURRENT_DIR}:/workspace \
	-w /workspace \
	 ${DOCKER_IMAGE_NAME}

DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr  \
	--privileged  \
	-v ${CURRENT_DIR}:/workspace \
	-e HTTP_PROXY=$(HTTP_PROXY) \
	-e HTTPS_PROXY=$(HTTPS_PROXY) \
	-e NO_PROXY=$(NO_PROXY) \
	${DOCKER_IMAGE_NAME}

DOCKER_BUILD_PARAMS := \
	--rm \
	--network=host \
	--build-arg http_proxy=$(HTTP_PROXY) \
	--build-arg https_proxy=$(HTTPS_PROXY) \
	--build-arg no_proxy=$(NO_PROXY) \
	-t $(DOCKER_IMAGE_NAME) . 
	
#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: test
.PHONY:  test stop

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build ${DOCKER_BUILD_PARAMS}
	
vf:
	@$(call msg, Configuring VFs ...)
	@sudo bash -c  ' \
 		echo ${NUM_VFS} | tee -a /sys/class/drm/card1/device/sriov_numvfs && \
 		modprobe vfio-pci && \
 		echo '8086 b0b0' | tee -a /sys/bus/pci/drivers/vfio-pci/new_id && \
  		echo 4096 | tee /proc/sys/vm/nr_hugepages \
  	
ai_benchmark: 
	@$(call msg, Running the AI Benchmark ...)
	@docker run ${DOCKER_RUN_PARAMS} \
		python3 ai_benchmark.py \
				-m ${MODEL} \
				-t ${DURATION}


list-devices: build
	@docker run ${DOCKER_RUN_PARAMS} bash -c  'source /opt/intel/oneapi/setvars.sh && sycl-ls'
		
stop:
	docker ps -q

bash: build
	@docker run ${DOCKER_RUN_PARAMS} bash
	
#----------------------------------------------------------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

