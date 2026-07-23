import os
import subprocess


def run_command(command_list):
    """Helper function to run bash commands and check for errors."""
    try:
        result = subprocess.run(
            command_list, check=True, text=True, capture_output=True, shell=True
        )

    except subprocess.CalledProcessError as e:
        raise


def main():
    cmd_dtifit = (
        "ml fsl && dtifit "
        "--data=/data/openneuro/ds007908/sub-0040/ses-1/dwi/sub-0040_ses-1_dir-AP_run-1_dwi.nii "
        "--out=./sub-0040_dti "
        "--mask=./sub-0040_brain_mask.nii.gz "
        "--bvecs=/data/openneuro/ds007908/sub-0040/ses-1/dwi/sub-0040_ses-1_dir-AP_run-1_dwi.bvec "
        "--bvals=/data/openneuro/ds007908/sub-0040/ses-1/dwi/sub-0040_ses-1_dir-AP_run-1_dwi.bvals"
    )

    cmd_fslroi = (
        "ml fsl && fslroi "
        "/data/openneuro/ds007908/sub-0040/ses-1/dwi/sub-0040_ses-1_dir-AP_run-1_dwi.nii "
        "./sub-0040_b0.nii.gz 0 1"
    )

    cmd_bet = "ml fsl && bet sub-0040_b0.nii.gz sub-0040_brain -m -f 0.3"

    run_command([cmd_fslroi])
    run_command([cmd_bet])
    run_command([cmd_dtifit])


if __name__ == "__main__":
    main()
