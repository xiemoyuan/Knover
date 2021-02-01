#!/bin/bash
set -eux

if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

job_conf=$1
source ${job_conf}

# init
data_dir=hdfs_data

HADOOP="hadoop fs -D hadoop.job.ugi=${hdfs_ugi} -D fs.default.name=${hdfs_path}"

function realpath() {
    echo $(dirname $1)/$(basename $1)
}

function process_tar_file_from_hdfs() {
    hdfs_path=$1
    local_path=$2
    echo "downloading tar file: ${hdfs_path} -> ${local_path}"

    # test whether local path exists now.
    if [[ -e ${local_path} ]]; then
        echo "WARNING: local_path: ${local_path} exists."
        return
    fi

    # download tar file from hdfs
    $HADOOP -get ${hdfs_path} ./
    the_tar=$(basename ${hdfs_path})
    the_dir=$(basename $(tar tf ${the_tar} | head -n 1))
    tar xf ${the_tar}
    rm ${the_tar}

    # create local dirname
    local_dirname=$(dirname ${local_path})
    mkdir -p ${local_dirname}

    if [[ $(realpath ${the_dir}) != $(realpath ${local_path}) ]]; then
        mv ${the_dir} ${local_path}
    fi

    echo "download tar file: ${hdfs_path} -> ${local_path} (Done)"
}

# python & paddle
if [[ ${hdfs_python_package:-""} != "" ]]; then
    process_tar_file_from_hdfs ${hdfs_python_package} python
fi

# init model
if [[ ${hdfs_init_model:-""} != "" ]]; then
    process_tar_file_from_hdfs ${hdfs_init_model} ${init_params:-${init_checkpoint}}
fi
if [[ ${hdfs_nsp_init_model:-""} != "" ]]; then
    process_tar_file_from_hdfs ${hdfs_nsp_init_model} ${nsp_init_params}
fi

# download data
if [[ ${hdfs_files:-""} != "" ]]; then
    mkdir -p ${data_dir}
    for hdfs_file in ${hdfs_files//,/ }; do
        $HADOOP -get ${hdfs_file} ./$data_dir/
    done
fi

if [[ ${remote_mount_point:-""} != "" ]]; then
    mkdir -p ${local_mount_point}
    cd /opt/afs_mount
    sed -i 's/24789/0/g' ./conf/afs_mount.conf
    IFS=',' array=(${afs_fs_ugi})
    afs_fs_username=${array[0]}
    afs_fs_password=${array[1]}
    ./bin/afs_mount \
        --username=${afs_fs_username} \
        --password=${afs_fs_password} \
        ${local_mount_point} \
        ${afs_fs_name}${remote_mount_point} \
        &> afs_mount.log &
    cd -
fi
