# GlioblastomaAI
Biogenic amines play important roles throughout cellular metabolism. This study explores a role of biogenic amines in glioblastoma pathogenesis. Here, we characterize the plasma levels of biogenic amines in glioblastoma patients undergoing standard-of-care treatment. We examined 138 plasma samples from 36 patients with isocitrate dehydrogenase (IDH) wild-type glioblastoma at multiple stages of treatment. Untargeted gas chromatography–time of flight mass spectrometry (GC-TOF MS) was used to measure metabolite levels. Machine learning approaches were then used to develop a predictive tool based on these datasets. Surgery was associated with increased levels of 12 metabolites and decreased levels of 11 metabolites. Chemoradiation was associated with increased levels of three metabolites and decreased levels of three other metabolites. Ensemble learning models, specifically random forest (RF) and AdaBoost (AB), accurately classified treatment phases with high accuracy (RF: 0.81 ± 0.04, AB: 0.78 ± 0.05). The metabolites sorbitol and N-methylisoleucine were identified as important predictive features and confirmed via SHAP. Conclusion: To our knowledge, this is the first study to describe plasma biogenic amine signatures throughout the treatment of patients with glioblastoma. A larger study is needed to confirm these results with hopes of developing a diagnostic algorithm.

## Directories
- [`data`](./data): Repository for raw input data.
- [`src`](./src): Source code.
- [`outputs`](./outputs): Repository for intermediate and output data.
- [`scripts`](./scripts): Shell scripts.

## Getting Started
The project has been tested in the following environments:
- Ubuntu 22.04.4 LTS
- Python 3.11

### Clone this repository to your local machine.
```console
git clone https://github.com/IBPA/GlioblastomaAI
cd GlioblastomaAI
```

### Create an Anaconda environment.
Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).
```console
conda create -n glio python=3.11
conda activate glio
```
You can deactivate the environment with `conda deactivate`.

### Install the required packages.
```console
pip install -r requirements.txt
```

### Run the code.
- Step 0: The dataset is not included in this repository. Please contact the first author, Orwa Aboud (oaboud@ucdavis.edu), for the dataset.
- Step 1: Run the scripts.
```console
./scripts/0_run_data_processing.sh
./scripts/1_run_model_selection.sh
./scripts/2_run_feature_analysis.sh
./scripts/3_run_visualization.sh
```

## GitHub Contributors
- Fangzhou Li (https://github.com/fangzhouli)

## Authors
- Orwa Aboud<sup>1,2,3</sup>
- Yin Liu<sup>1,2,4</sup>
- Lina Dahabiyeh<sup>5,6</sup>
- Ahmad Abuaisheh<sup>7</sup>
- Fangzhou Li<sup>8,9,10</sup>
- John Paul Aboubechara<sup>1</sup>
- Jonathan Riess<sup>3,11</sup>
- Orin Bloch<sup>2</sup>
- Rawad Hodeify<sup>12</sup>
- Ilias Tagkopoulos<sup>8,9,10</sup>
- Oliver Fiehn<sup>5</sup>
1. Department of Neurology, University of California, Davis, Sacramento, CA 95817, USA
2. Department of Neurological Surgery, University of California, Davis, Sacramento, CA 95817, USA
3. Comprehensive Cancer Center, University of California Davis, Sacramento, CA 95817, USA
4. Department of Ophthalmology, University of California, Davis, Sacramento, CA 95817, USA
5. West Coast Metabolomics Center, University of California Davis, Davis, CA 95616, USA
6. Department of Pharmaceutical Sciences, School of Pharmacy, The University of Jordan, Amman 11942, Jordan
7. School of Medicine, Al Balqa Applied University, Al-Salt 19117, Jordan
8. Department of Computer Science, University of California, Davis, Sacramento, CA 95616, USA
9. Genome Center, University of California, Davis, Sacramento, CA 95616, USA
10. USDA/NSF AI Institute for Next Generation Food Systems (AIFS), Davis, CA 95616, USA
11. Department of Internal Medicine, Division of Hematology and Oncology, University of California, Davis, Sacramento, CA 95817, USA
12. Department of Biotechnology, School of Arts and Sciences, American University of Ras Al Khaimah, Ras Al-Khaimah 10021, United Arab Emirates

## Contact
For data-related questions, please contact Orwa Aboud (oaboud@ucdavis.edu). For code-related questions, you can contact Fangzhou Li (fzli@ucdavis.edu) or Prof. Ilias Tagkopoulos (itagkopoulos@ucdavis.edu).

## Citation
```bibtex
@article{aboud2023profile,
  title={Profile Characterization of Biogenic Amines in Glioblastoma Patients Undergoing Standard-of-Care Treatment},
  author={Aboud, Orwa and Liu, Yin and Dahabiyeh, Lina and Abuaisheh, Ahmad and Li, Fangzhou and Aboubechara, John Paul and Riess, Jonathan and Bloch, Orin and Hodeify, Rawad and Tagkopoulos, Ilias and others},
  journal={Biomedicines},
  volume={11},
  number={8},
  pages={2261},
  year={2023},
  publisher={MDPI}
}
```

## License
This project is licensed under the Apache-2.0 License. Please see the [LICENSE](./LICENSE) file for details.

## Funding
Aboud and Liu are supported in part by the UC Davis Paul Calabresi Career Development Award for Clinical Oncology as funded by the National Cancer Institute/National Institutes of Health through grant #2K12CA138464-11. Fiehn is supported by NIH U2C ES030158 funding related to the study. Tagokopoulos is supported by USDA-NIFA #2020-67021-32855.
