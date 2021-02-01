#!/bin/bash
set -eux

if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

job_conf=$1
source ${job_conf}

mpirun hostname

mkdir -p log
mkdir -p output

cd ${TRAIN_WORKSPACE}/visualdl_util/
log_pid=`ps fux | grep direct_parser.py | grep python | awk -F " " '{print $2}'`
kill -9 $log_pid
eval $SYS_PYTHON_CMD -u \
    direct_parser.py \
    --job_id="${SYS_JOB_ID}" \
    --log_file="../env_run/log/workerlog.0" \
    --sys_api_host="${SYS_API_HOST}" \
    --sys_api_port="${SYS_API_PORT}" \
    --influx_db_url="${SYS_INFLUX_DB_URL}" \
    --user_id="${SYS_USER_ID}" \
    --ak="${SYS_PRIVILEGE_AK}" \
    --sk="${SYS_PRIVILEGE_SK}" \
    --trainer_id="${PADDLE_TRAINER_ID}" \
    > ./log/log_parser.log 2>&1 & echo $! > ./log_parser.pid
cd -

mpirun sh ./scripts/setup.sh ${job_conf} &> log/setup.log

mpirun \
    --bind-to none \
    -x use_k8s="true" \
    -x distributed_args="--use_paddlecloud" \
    -x log_dir="./log" \
    -x save_path="./output" \
    -x random_seed=$RANDOM \
    -x PATH="$PWD/python/bin:$PATH" \
    -x PYTHONPATH="$PWD/python" \
    sh ${job_script:-"./scripts/distributed/train.sh"} ${job_conf}
