#!/bin/bash
#SBATCH --account=aero_ci
#SBATCH --partition=gpu-gh200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=72 
#SBATCH --mem=96GB
#SBATCH --time=2:00:00
#SBATCH --job-name=nektar-cluster-test

# Import bash script arguments
export PRIVATE_TOKEN=${1}
# CI VARIABLES
export CI_PIPELINE_ID=${2}
export CI_MERGE_REQUEST_IID=${3}
export CI_JOB_NAME=${4}
export CI_JOB_ID=${5}
export CI_REGISTRY_USER=${6}
export CI_REGISTRY_PASSWORD=${7}
export CI_REGISTRY=${8}
export CI_REGISTRY_IMAGE=${9}
export CI_RUNNER_ID=${10}
# BUILD OPTIONS
export BUILD_SIMD=${11}
export BUILD_DEVICE=${12}
export BUILD_CC=${13}
export BUILD_CXX=${14}
export BUILD_FC=${15}
export BUILD_SINGLE_PRECISION=${16}
export DISABLE_CWIPI=${17}
export DISABLE_MCA=${18}
export ENABLE_ALIGN_MEM=${19}
export DO_COVERAGE=${20}
export LD_PRELOAD=${21}
export PYTHON_EXECUTABLE=${22}
export APPTAINER_FLAGS=${23}
export CI_COMMIT_REF_NAME=${24}
export CI_PROJECT_ID=${25}
export CI_PROJECT_URL=${26}
export CI_MERGE_REQUEST_PROJECT_URL=${27}

export NUM_CPUS=72

trigger_gate(){
    JOB_ID=""
    count=0
    POSTPROCESS_JOBNAME=$(echo $CI_JOB_NAME | rev | cut -d- -f2- | rev)-post-process
    while : ; do
        ((count++))
        sleep 1
        # Get job id
        JOB_ID=$(curl --header "PRIVATE-TOKEN: ${PRIVATE_TOKEN}" \
        --header "Content-Type: application/json" \
        "https://gitlab.nektar.info/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs?scope=manual" \
        | tac | tac | python3 -c "import sys,json; print([x['id'] for x in json.load(sys.stdin) if x['name'] == \"${POSTPROCESS_JOBNAME}\"][0])")
        if [ "$JOB_ID" != "" ]; then
            break
        fi
        if (( $count == 60 )); then
            break
        fi
    done;

    # Call pipeline
    curl -X POST --header "PRIVATE-TOKEN: ${PRIVATE_TOKEN}" \
    --header "Content-Type: application/json" \
    "https://gitlab.nektar.info/api/v4/projects/${CI_PROJECT_ID}/jobs/${JOB_ID}/play"
}
    
JOB_NAME=$(echo $CI_JOB_NAME | rev | cut -d- -f2- | rev)

# Root directory
rootdir=$(pwd)

# Create directory
mkdir -p ${CI_PIPELINE_ID} && cd ${CI_PIPELINE_ID}

# Create sub-directory
rm -rf ${JOB_NAME}
mkdir -p ${JOB_NAME} && cd ${JOB_NAME}

# Initialise an empty repository, so that only the commit under test is
# fetched (merge request head, or the branch/tag the pipeline runs on)
command="git init nektar"
echo ${command} > $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
error_code=$?
if (( $error_code )); then 
    echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    trigger_gate
    exit $error_code
fi
cd nektar
# Merge request refs live in the target project, branches and tags in the
# project the pipeline runs in (which may be a fork)
GIT_URL=${CI_MERGE_REQUEST_PROJECT_URL:-${CI_PROJECT_URL}}
command="git remote add origin ${GIT_URL}.git"
echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
error_code=$?
if (( $error_code )); then 
    echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    trigger_gate
    exit $error_code
fi

# Fetch merge request branch
if (( ${CI_MERGE_REQUEST_IID} )); then
    command="git fetch --depth 1 origin merge-requests/${CI_MERGE_REQUEST_IID}/head:MR${CI_MERGE_REQUEST_IID}" 
    echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    ${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    error_code=$?
    if (( $error_code )); then 
        echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
        trigger_gate
        exit $error_code
    fi
    command="git checkout MR${CI_MERGE_REQUEST_IID}" 
    echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    ${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    error_code=$?
    if (( $error_code )); then 
        echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
        trigger_gate
        exit $error_code
    fi
# Fetch the branch or tag the pipeline runs on
else
    command="git fetch --depth 1 origin ${CI_COMMIT_REF_NAME}"
    echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    ${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    error_code=$?
    if (( $error_code )); then 
        echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
        trigger_gate
        exit $error_code
    fi
    command="git checkout FETCH_HEAD" 
    echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    ${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    error_code=$?
    if (( $error_code )); then 
        echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
        trigger_gate
        exit $error_code
    fi
fi

# Load modules
command="source /etc/profile.d/modules.sh"
echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
error_code=$?
if (( $error_code )); then 
    echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    trigger_gate
    exit $error_code
fi
command="module load apptainer"
echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
error_code=$?
if (( $error_code )); then 
    echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    trigger_gate
    exit $error_code
fi

# Configure docker
export OS_DISTRO=$(echo $CI_JOB_NAME | cut -d- -f 1)
export OS_VERSION=$(echo $CI_JOB_NAME | cut -d- -f 2)
export BUILD_TYPE=$(echo $CI_JOB_NAME | cut -d- -f 3)

# The cluster nodes are aarch64, so they cannot run the images the build-env
# jobs push: those are built by x86-64 runners. The aarch64 image is built and
# pushed by hand instead -- see "Cluster environment images" in
# docker/nektar-env/README.md -- under its own tag rather than sharing
# env-${OS_DISTRO}-${OS_VERSION}-${BUILD_TYPE} with them, because a
# single-platform push replaces whatever is at a tag: a multi-architecture
# manifest there would last only until the next build-env job ran, and this
# job would then pull an x86-64 image without anything saying so.
export ENV_ARCH=arm64
export ENV_NAME=env-${OS_DISTRO}-${OS_VERSION}-${BUILD_TYPE}-${ENV_ARCH}

# Configure, build, and run merge request branch
command="apptainer run $APPTAINER_FLAGS \
      --env \"BUILD_TYPE=$BUILD_TYPE\" \
      --env \"BUILD_SIMD=$BUILD_SIMD\" \
      --env \"BUILD_DEVICE=$BUILD_DEVICE\" \
      --env \"BUILD_CC=$BUILD_CC\" \
      --env \"BUILD_CXX=$BUILD_CXX\" \
      --env \"BUILD_FC=$BUILD_FC\" \
      --env \"BUILD_SINGLE_PRECISION=$BUILD_SINGLE_PRECISION\" \
      --env \"DISABLE_CWIPI=$DISABLE_CWIPI\" \
      --env \"DISABLE_MCA=$DISABLE_MCA\" \
      --env \"ENABLE_ALIGN_MEM=$ENABLE_ALIGN_MEM\" \
      --env \"DO_COVERAGE=$DO_COVERAGE\" \
      --env \"NUM_CPUS=$NUM_CPUS\" \
      --env \"LD_PRELOAD=$LD_PRELOAD\" \
      --env \"PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE\" \
      docker://$CI_REGISTRY_IMAGE:$ENV_NAME \
      bash -x .gitlab-ci/build-and-test.sh"
echo ${command} >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
${command} &>> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log

error_code=$?
if (( $error_code )); then 
    echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log

    # Clean-up build directory
    cd $rootdir
    rm -rf "$rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/nektar"
    trigger_gate
    exit $error_code
fi

echo "JOB SUCCEEDED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log

# Clean-up build directory
cd $rootdir
rm -rf "$rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/nektar"
trigger_gate
exit $error_code
