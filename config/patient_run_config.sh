#!/usr/bin/env bash

#TODO place these in a better form ?

PEGASUS_HOME=/opt/pegasus-home
echo ${PEGASUS_HOME}
export PEGASUS_HOME

OS=MACOSX
echo ${OS}
export OS

SUBJECTS_FOLDER=/Applications/freesurfer_dev/subjects
echo ${SUBJECTS_FOLDER}
export SUBJECTS_FOLDER

SUBJECT_NAME=XYZ
export SUBJECT_NAME

OPENMP_THRDS=4
export OPENMP_THRDS

MRTRIX_THRDS=4
export MRTRIX_THRDS