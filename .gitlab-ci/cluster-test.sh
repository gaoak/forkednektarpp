#!/bin/bash
#SBATCH --account=aero_ci
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
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
export NUM_CPUS=${21}
export LD_PRELOAD=${22}
export PYTHON_EXECUTABLE=${23}
export APPTAINER_FLAGS=${24}

trigger_gate(){
    JOB_ID=""
    count=0
    POSTPROCESS_JOBNAME=$(echo $CI_JOB_NAME | rev | cut -d- -f2- | rev)-post-process
    while : ; do
        ((count++))
        sleep 1
        # Get job id
        JOB_ID=$(curl --header "Content-Type: application/json" \
        "https://gitlab.nektar.info/api/v4/projects/2/pipelines/${CI_PIPELINE_ID}/jobs?scope=manual" \
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
    "https://gitlab.nektar.info/api/v4/projects/2/jobs/${JOB_ID}/play"
}
    
JOB_NAME=$(echo $CI_JOB_NAME | rev | cut -d- -f2- | rev)

# Root directory
rootdir=$(pwd)

# Create directory
mkdir -p ${CI_PIPELINE_ID} && cd ${CI_PIPELINE_ID}

# Create sub-directory
rm -rf ${JOB_NAME}
mkdir -p ${JOB_NAME} && cd ${JOB_NAME}

# Git clone master branch
command="git clone https://gitlab.nektar.info/nektar/nektar.git --branch master nektar"
echo ${command} > $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
${command}
error_code=$?
if (( $error_code )); then 
    echo "JOB FAILED" >> $rootdir/${CI_PIPELINE_ID}/${JOB_NAME}/outfile.log
    trigger_gate
    exit $error_code
fi
cd nektar

# Fetch merge request branch
command="git fetch origin merge-requests/${CI_MERGE_REQUEST_IID}/head:MR${CI_MERGE_REQUEST_IID}" 
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
export ENV_NAME=env-${OS_DISTRO}-${OS_VERSION}-${BUILD_TYPE}

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
    trigger_gate
    exit $error_code
fi

cd ../

trigger_gate
