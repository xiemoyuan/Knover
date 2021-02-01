#!/bin/bash
set -u

if [[ $# != 1 ]];then
    exit -1
fi

path=`pwd`/$1
cd $path
echo "current path :${path}"
while true
do
    file_list=`ls 2>/dev/null`
    for file in $file_list
    do
        if [[ -d "${file}" ]] && [[ -f "${file}.finish" ]]; then
            tar_file_name="${file}.tar"
            echo "pack files to $tar_file_name"
            tar -cf ${tar_file_name} ${file}
            rm -rf ${file}
            rm ${file}.finish
        fi
    done
    #tars_size=`du -s . | awk -F "\t" '{print $1}'`
    #if [[ $tars_size -gt 104857600 ]]; then
    #    oldest_tar=`ls -t *.tar | tail -1`
    #    echo "rm $oldest_tar"
    #    rm $oldest_tar
    #fi
    sleep 1m
done
