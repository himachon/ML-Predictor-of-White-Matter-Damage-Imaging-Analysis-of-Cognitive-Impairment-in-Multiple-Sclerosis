"""
feature_extraction.py

Feature Extraction via the John Hopkins Universit

"""


from pathlib import Path
import numpy as np
import ants
import pandas as pd

# Setup paths
root_folder = Path("/home/jovyan/dwi_outputs")
output_root = Path("/home/jovyan/dwi_outputs") # Keeps outputs inside the same directory structure
atlas_dir = Path("/home/jovyan/JHU DTI-based white-matter atlases")

# MOVE OUTSIDE LOOP: Load static templates once to save substantial time
# Note: Reverted to 2mm for performance, change back to 1mm if your data is 1mm isotropic
template_fa = ants.image_read(str(atlas_dir / "JHU-ICBM-FA-2mm.nii.gz"))
jhu_labels = ants.image_read(str(atlas_dir / "JHU-ICBM-labels-2mm.nii.gz"))

# Region mapping directory dictionary for clean terminal output
regions = {
    4: "Genu of corpus callosum",
    5: "Body of corpus callosum",
    6: "Splenium of corpus callosum",
    7: "Fornix",
    26: "Superior corona radiata R",
    27: "Superior corona radiata L",
    43: "Superior longitudinal fasciculus R",
    44: "Superior longitudinal fasciculus L"
}
all_subjects = []
# 1. Loop through each subject folder
for subject_dir in root_folder.glob("sub-*"):
    if not subject_dir.is_dir():
        continue
        
    sub_id = subject_dir.name  # Extracts just the folder name string (e.g., "sub-9004")
    
    # Construct exact path to the subject's raw FA map
    subject_fa_path = subject_dir / f"{sub_id}_FA.nii.gz"
    
    # Guard clause: skip if the subject file is missing or named differently
    if not subject_fa_path.exists():
        print(f"Skipping {sub_id}: File {subject_fa_path.name} not found.")
        continue
        
    print(f"\n=========================================")
    print(f"Registering and extracting metrics for: {sub_id}")
    print(f"=========================================")

    # Load subject's specific native FA map
    subject_fa = ants.image_read(str(subject_fa_path))

    # 2. Register subject to the JHU template space
    # Warning: SyN can take 5-15 minutes per subject depending on your CPU power
    registration = ants.registration(fixed=template_fa, moving=subject_fa, type_of_transform='SyN')

    # 3. Bring the JHU Atlas labels back into the subject's native space
    labels_in_subject_space = ants.apply_transforms(
        fixed=subject_fa,
        moving=jhu_labels,
        transformlist=registration['invtransforms'],
        interpolation='nearestNeighbor' # Crucial for keeping label integers whole
    )

    # 4. Convert ANTs images to numpy arrays for calculation
    labels_data = labels_in_subject_space.numpy()
    subject_fa_data = subject_fa.numpy()

    # Loop through regions dynamically and cleanly extract statistics
    print(f"\nResults for {sub_id}:")
    subjects = {}
    subjects["Subject"] = sub_id
    for roi_id, roi_name in regions.items():
        # Mask out values belonging to the target region
        roi_fa_values = subject_fa_data[labels_data == roi_id]
        
        # Guard clause against empty regions due to edge alignment failures
        if roi_fa_values.size == 0:
            print(f"  {roi_name} (ID {roi_id}): No voxels found.")
            continue
            
        mean_fa = np.mean(roi_fa_values)
        std_fa = np.std(roi_fa_values)
        print(f"  {roi_name} (ID {roi_id}) -> Mean FA: {mean_fa:.4f} (±{std_fa:.4f})")
        subjects[roi_name] = mean_fa
    all_subjects.append(subjects)
    
        
df = pd.DataFrame(all_subjects)

output_csv = output_root / "white_matter_features.csv"

df.to_csv(output_csv, index=False)

print(f"\nSaved feature table to {output_csv}")