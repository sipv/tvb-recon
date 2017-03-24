#!/bin/bash

# Workflow for identifying electrode contact positions in T1 space, from CT and FS recon.
# Final output file also provides nearest parcel name in default FS parcellations.
#
# Requires manual intervention to match implantation schema to labeled contacts.
# Caller will be prompted at appropriate time to create a file named schema.txt
# in which each line has the format "%d %s" % (label_integer, electrode_name).
#
# Environment variables affecting behavior:
#
#   CT_ELEC_INTENSITY_TH     - threshold applied to determin if a CT voxel is an SEEG contact
#                              default: 1000
#
#   SEEG_FOLDER              - working folder for this script; will be created if not existing
#                              default: $SUBJECTS_DIR/$SUBJECT/seeg
#
#   FS_MRI_FOLDER            - path to subject's FreeSurfer MRI folder
#                              default: $SUBJECTS_DIR/$SUBJECT/mri
# 
# It may eventually be useful to make this workflow highly interactive, so that the
# parameters of image processing can be tuned while looking at the results.


if [ -z "$SUBJECT" ] || [ -z "$CT" ]
then
    echo "CT=/path/to/ct/data SUBJECT=fs_subject_name seeg-ct.sh"
    exit 1
fi

set -eu
set -o pipefail

# setup variables
threshold=${CT_ELEC_INTENSITY_TH:-"1000"}
topic_folder=${SEEG_FOLDER:-"$SUBJECTS_DIR/$SUBJECT/seeg"}
mri_folder=${FS_MRI_FOLDER:-"$SUBJECTS_DIR/$SUBJECT/mri"}
flirt_cmd="flirt"
if [ -z $(which flirt) ]; then flirt_cmd="fsl5.0-flirt"; fi

# work in seeg folder 
mkdir -p $topic_folder
pushd $topic_folder

# all images to RAS nii gz
mri_convert $CT ./CT.nii.gz --out_orientation RAS &
mri_convert $mri_folder/T1.mgz ./T1.nii.gz --out_orientation RAS &
mri_convert $mri_folder/brain.mgz ./brain.nii --out_orientation RAS

wait # for conversions

# move CT to T1 space
$flirt_cmd -v -in ./CT.nii.gz -ref ./T1.nii.gz -omat CT_to_T1.mat -out CT_in_T1.nii.gz \
    -dof 12 -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -cost mutualinfo

# binarize & erode brain mask; erode to avoid inner skull
mri_binarize --i ./brain.nii --o ./mask.nii --min 10 --erode 4

# mask ct
mri_binarize --i CT_in_T1.nii.gz --o masked_CT.nii --min $threshold --mask mask.nii

# dilate mask for labeling
mri_binarize --i masked_CT.nii --o dilated_CT.nii --min 0.5 --dilate 2 --erode 1

# label dilated mask and apply to mask
python3 <<EOF
from tvb.recon.algo.reconutils import label_with_dilation
label_with_dilation("masked_CT.nii", "dilated_CT.nii", "labeled_CT.nii")
EOF

echo compare labels to implantation schema
echo mrview T1.nii.gz -overlay.load labeled_CT.nii -overlay.interpolation_off -overlay.colourmap 3
echo open implantation_pptx

echo TODO and put correspondence in file
echo labels w/o names will be ignored

read -n 1 -p "Press the Any key to continue.."

# find contact positions
python3 <<EOF
import nibabel, numpy as np, tvb.recon.algo.reconutils as utils
label_to_name = {}
with open("schema.txt", "r") as fd:
    for line in fd.readlines():
        label_num, label_name = line.strip().split(' ')
        label_num = int(label_num)
        assert label_num not in label_to_name
        label_to_name[int(label_num)] = label_name
nii = nibabel.load("labeled_CT.nii")
lab_bin = nii.get_data()
aff = nii.affine
ulab = np.unique(lab_bin)
ulab = ulab[ulab > 0]
# TODO find closest voxel in parcellation, provide name
fmt = '%s%d\t%f\t%f\t%f\n'
with open("seeg_name_pos.txt", "w") as fd:
    for ul in ulab[ulab > 0]:
        if ul not in label_to_name:
            print("skipping object label %d" % (ul, ))
            continue
        xyz_pos = utils.periodic_xyz_for_object(lab_bin, ul, aff)
        xyz_pos = xyz_pos[np.argsort(np.abs(xyz_pos[:, 0]))]
        name = label_to_name[ul]
        for i, (x, y, z) in enumerate(xyz_pos):
            fd.write(fmt % (name, i, x, y, z))
EOF

popd

