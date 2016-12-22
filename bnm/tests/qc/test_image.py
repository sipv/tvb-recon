# -*- coding: utf-8 -*-

import os
import pytest
from bnm.recon.model.constants import SNAPSHOTS_DIRECTORY, SNAPSHOT_NAME, AXIAL
from bnm.recon.qc.image.processor import ImageProcessor
from bnm.tests.base import get_data_file

SNAPSHOT_NUMBER = 0
TEST_SUBJECT_MODIF = "fsaverage_modified"
TEST_SUBJECT = "freesurfer_fsaverage"
TEST_MRI_FOLDER = "mri"
TEST_SURF_FOLDER = "surf"
TEST_ANNOT_FOLDER = "label"
TEST_T1 = "T1.nii.gz"
TEST_BRAIN = "brain.nii.gz"
TEST_SURF = "lh.pial"
TEST_GIFTI_SURF = "lh.pial.gii"
TEST_ANNOT = "lh.aparc.annot"


def setup_module():
    os.environ["MRI"] = os.path.join("data", TEST_SUBJECT_MODIF, TEST_MRI_FOLDER)
    os.environ["T1_RAS"] = TEST_T1
    os.environ["SUBJ_DIR"] = os.path.join("data", TEST_SUBJECT_MODIF)


def teardown_module():
    for file_path in os.listdir(SNAPSHOTS_DIRECTORY):
        os.remove(os.path.join(SNAPSHOTS_DIRECTORY, file_path))
    os.rmdir(SNAPSHOTS_DIRECTORY)


def test_show_single_volume():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    volume_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_T1)
    processor.show_single_volume(volume_path, False)
    resulted_file_name = processor.generate_file_name(AXIAL, SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name))


def test_overlap_2_volumes():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    background_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_T1)
    overlay_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_BRAIN)
    processor.overlap_2_volumes(background_path, overlay_path, False)
    resulted_file_name = processor.generate_file_name(AXIAL, SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name))


def test_overlap_3_volumes():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    background_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_T1)
    overlay1_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_BRAIN)
    overlay2_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_BRAIN)
    processor.overlap_3_volumes(background_path, overlay1_path, overlay2_path, False)
    resulted_file_name = processor.generate_file_name(AXIAL, SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name))


@pytest.mark.skip("Because it takes about 10 min to complete")
def test_overlap_surface_annotation():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    surface_path = get_data_file(TEST_SUBJECT, TEST_SURF_FOLDER, TEST_SURF)
    annotation_path = get_data_file(TEST_SUBJECT, TEST_ANNOT_FOLDER, TEST_ANNOT)
    processor.overlap_surface_annotation(surface_path, annotation_path)
    resulted_file_name = processor.generate_file_name('surface_annotation', SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name + str(0)))


def test_overlap_volume_surface():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    volume_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_T1)
    surface_path = get_data_file(TEST_SUBJECT, TEST_SURF_FOLDER, TEST_SURF)
    processor.overlap_volume_surfaces(volume_path, [surface_path], False, False)
    resulted_file_name = processor.generate_file_name(AXIAL, SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name))


def test_overlap_volume_centered_surface():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    volume_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_T1)
    surface_path = get_data_file(TEST_SUBJECT, TEST_SURF_FOLDER, TEST_SURF)
    processor.overlap_volume_surfaces(volume_path, [surface_path], True, False)
    resulted_file_name = processor.generate_file_name(AXIAL, SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name))

def test_overlap_volume_gifti_surface():
    processor = ImageProcessor(SNAPSHOTS_DIRECTORY, SNAPSHOT_NUMBER)
    volume_path = get_data_file(TEST_SUBJECT_MODIF, TEST_MRI_FOLDER, TEST_T1)
    surface_path = get_data_file(TEST_SUBJECT_MODIF, TEST_SURF_FOLDER, TEST_GIFTI_SURF)
    processor.overlap_volume_surfaces(volume_path, [surface_path], False, False)
    resulted_file_name = processor.generate_file_name(AXIAL, SNAPSHOT_NAME)
    assert os.path.exists(processor.writer.get_path(resulted_file_name))