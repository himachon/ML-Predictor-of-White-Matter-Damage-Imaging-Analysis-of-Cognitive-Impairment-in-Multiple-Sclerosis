# ML-Predictor-of-White-Matter-Damage-in-Early-Multiple-Sclerosis-Patients
The following project predicts white matter damage in Multiple Sclerosis patients through mapping Fractional Anistropy trajectories using a multi-modal OpenNeuro DWI dataset. Key regions identified through literature include corpus callosum, thalamic and fornix regions. The given pipeline identifies structural degradation associated with early MS injury.


# Dataset Overview
The dataset used is a public database sourced from OpenNeuro, an open-access free platform that contains a plethora of neuroimaging datasets including fMRI, MEG, and EEG.  The dataset referred to and utilized in this project is the "Multi-scale, multi-modal imaging assessment of trajectories of cognitive impairment in Multiple Sclerosis." A total of 28 subjects including 8 healthy controls and 20 MS subjects are represented along with their baseline, session-1 data.  

### Access to Dataset
Amy Kuceyeski, Keith W. Jamison, Ke Huang, Yeona Kang, Sandra Hurtado Rua, Noel George, and Susan A. Gauthier (2026). Multi-scale, multi-modal imaging assessment of trajectories of cognitive impairment in Multiple Sclerosis. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds007908.v1.0.0

# Objectives

## Tested Variables

Multiple Sclerosis is classified as a “chronic autoimmune disease” that affects the central nervous system. Specifically, MS includes “demyelination, gliosis, inflammation” and neuronal loss at later stages. Patients may experience a wide range of symptoms including “focal weakness, bladder and bowel dysfunction, and cognitive impairment” ().
 
To reduce noise and improve the accuracy of the model, the project pinpointed specific brain regions to focus on; these regions were determined after consulting prior literature. The study “White and Gray Matter Changes in Early Multiple Sclerosis” includes 29 subjects—20 MS patients (aged 22-45) and 9 controls (aged 26-48). The study examines the distribution of white matter damage within early MS patients using MRI DTI tractography techniques such as Tract-Based Spatial Statistics-TBSS. Results show that in comparison to control subjects, MS patients displayed significant ( P <0.005) FA white matter tract abnormalities within the corpus callosum, parietal, frontal, thalamic, and fornix (Javed et al., 2012). Another experiment comparing white matter density in 38 subjects including 17 early MS patients and 21 control subjects found a 23% reduction in white matter within the parietal regions, 11% decrease in the frontal areas, and a 12% drop in the parietal-occipital regions for MS patients (Sun et al., 2017). Based on these studies, the current project focuses on the white matter tracts found in the parietal, frontal, corpus callosum, thalamic, and fornix regions of the brain.

Fractional Anisotropy (FA) refers to a neuroimaging scalar value (0 to 1) that quantifies directional alignment of water diffusion within white matter. Values tending towards 0 correspond to demyelination and weakened structure while values approaching 1 indicate clusters of healthy, myelinated nerve fibers (Diffusion Tensor Imaging (DTI)). 

## Methods


A brain mask was generated and applied to exclude non-brain tissue. Using the respective gradient directions (b-values and b-vectors), the diffusion tensors were fitted to the preprocessed diffusion-weighted images. We accessed the open source John Hopkins University DTI white matter atlas (ICBM-DTI-81 Label Atlas) as a neuroimaging reference map to nonlinearly register the resulting FA maps. Following this, the atlas labels were transformed into respective diffusion spaces for each subject. Based on previous literature, we determined 8 relevant white matter regions to extract mean FA values from; these regions include the Genu of corpus callosum, Body of corpus callosum, Splenium of corpus callosum, Fornix, Superior corona radiata (right), Superior corona radiata (left), Superior longitudinal fasciculus, Superior longitudinal fasciculus (left). These regions posed the most relevant areas of the brain as they included the largest white matter tracts that spanned the parietal, fornix, frontal, and thalamic regions. All eight regions were combined into a feature table: rows representing subjects and columns including mean FA values of a selected white matter tract for further statistical analysis and machine learning classification.


## Statistical Analysis

Subjects were divided into two different groups: one that included the 20 patients diagnosed with early signs of MS and the latter including the 9 control subjects. The regional FA values were summarized based on the comparison of these two groups. To determine whether FA value measurements could distinguish the two groups, a stratified 5-fold cross-validation was applied along with a logistic regression. Model performance was assessed using accuracy, recall, precision, F1-score, and the area under the receiver operating characteristic curve (ROC-AUC). The relative contribution of each white matter tract was then examined based on the fitted logistic regression coefficients and compared to the classification model.

## Justification for Machine Learning Modeling Approach

A logistic regression classifier was implemented utilizing the scikit-learn library. Before model fitting, features were standardized using StandardScaler, allowing for variables to be on a comparable scale. A 5-fold cross-validation was applied to evaluate model performance in order to preserve class distribution within testing and training folds. Logistic regression coefficients were averaged across folds to evaluate the relevant contribution of each white matter tract and improve model interpretability.

## Repository Structure

The following repository is organized based on modular components that distinguish between image processing, statistical analysis, feature extraction, and machine learning. All source code is located in the src/ directory for the analysis pipeline while figures and outputs are stored in respective figures/ and results/ directories to aid in reproducibility and interpretation.

# Results

## Feature Extraction

[white_matter_features.csv](/Data/white_matter.csv)


**Table 1.** Final feature table of Fractional Anisotropy (FA) values from 8 white matter regions (Genu of corpus callosum, Body of corpus callosum, Splenium of corpus callosum, Fornix, Superior corona radiata (right), Superior corona radiata (left), Superior longitudinal fasciculus (right), and Superior longitudinal fasciculus (left)) using the JHU white matter atlas. Each row represents a subject along with mean FA values for all 8 regions. Additionally, subjects were characterized as a Control Subject (LS) or early MS patient (HS) within the “Group” column.


<img width="1800" height="1200" alt="Correlation Matrix Heatmap" src="https://github.com/user-attachments/assets/cd03c5f4-8fa9-4674-86fc-dc3396145153" />

**Figure 1.** Correlation matrix depicting relationships between feature white matter regions. Regions of the corpus callosum including the genu, body, and splenium exhibited high positive correlation. Additionally, the right and left superior corona radiata presented moderate correlation (r = 0.66) while the superior longitudinal fasciculus showed relatively weak correlations with the majority of regions.

<img width="2400" height="1800" alt="ROC_curve" src="https://github.com/user-attachments/assets/e579b810-aee3-4f13-b914-efc600a52971" />

**Figure 2.** Logistic regression classifier for eight white matter regional FA measurements as predictor vectors. Large variability is present: ROC-AUC value ranges from 0.25 to 0.88 while the accuracy ranges from 0.25 to 0.88.


<img width="4771" height="2669" alt="white_matter_tracts_fa_comparison" src="https://github.com/user-attachments/assets/fa3d948f-25b1-4750-b426-940b2ec7272e" />

**Figure 3.** Averaged logistic regression coefficients across cross-validation folds as a predictor of white matter tracts that contribute heavily to classification performance. Positive coefficients suggest higher FA values increase the probability of classification as an early MS patient, whereas negative coefficients suggest the opposite. 

## Key Findings

From the given results, we found that certain white matter regions posed moderate-to-strong correlations. Specifically, the corpus callosum and the superior corona radiata on the left hemisphere displayed significant positive correlations, suggesting these two white matter pathways function closely together in mediating brain activity. Our logistic regression performed with an average classification accuracy of 66% with an ROC-AUC of 0.575, proposing only modest discriminative ability. Although the model offered relatively high recall (0.85), proposing it identified most of the early MS patients, the low ROC-AUC indicates limited overall distinction between the two groups. Findings suggest that FA measurements from the 8 selected regions solely may not accurately distinguish individuals with early MS symptoms and control subjects.

## Limitations & Future Research

There are several limitations to this work, namely the small sample size limiting the amount of data we have to train a predictive model. With the given sample size, there is high variability in our prediction model. To improve this study in the future, we should include additional diffusion metrics (AD, RD, and MD), obtain a larger sample size, or focus primarily on the largest weighted feature white matter tract variables such as the corpus callosum and the superior corona radiata on the left hemisphere. 

