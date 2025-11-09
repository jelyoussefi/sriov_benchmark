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

NUM_VFS ?= 4
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


#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: test
.PHONY:  test stop

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build . -t ${DOCKER_IMAGE_NAME} 

i915:
	@$(call msg, Configuring VFs ...)
	@sudo modprobe i915 max_vfs=31
	
vf:
	@$(call msg, Configuring VFs ...)
	@sudo bash -c  ' \
 		echo ${NUM_VFS} | tee -a /sys/class/drm/card1/device/sriov_numvfs && \
 		modprobe vfio-pci && \
 		echo '8086 4680' | tee -a /sys/bus/pci/drivers/vfio-pci/new_id && \
  		echo 4096 | tee /proc/sys/vm/nr_hugepages \
  		'

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

