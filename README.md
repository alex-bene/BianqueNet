---
title: Intervertebral Disc Degeneration Analysis
emoji: 🌍
colorFrom: red
colorTo: indigo
sdk: streamlit
python_version: 3.8
sdk_version: 1.35.0
app_file: streamlit_app.py
pinned: false
---

# Quantitative analysis of intervertebral disc degeneration

This project is used for quantitative analysis of intervertebral disc(IVD) degeneration on T2MRI images
and generate structured analysis reports.

The structured analysis reports include those features:

- Geometrical parameters of IVD
- Signal intensity difference of IVD
- IVD degeneration quantitative analysis results

## File Structure

Altogether 5 sub-directory.The files in the `function` folder are used for calculating parameters and drawing.

1. `input` folder contains two subfolders, `baseline_range` folder contains statistical benchmark parameters,
   and `data_input` folder contains a case for each hospital center.
2. `network_big_apex` folder contains different network structures.
3. `output` folder stores the output results.
4. and `weights_big_apex folder` contains the training parameters of BianqueNet, please download from `https://u.pcloud.link/publink/show?code=XZ6DBkVZPdNHAOg14IHxKVDcnvq4pH1c4b1k`.

```
.
│  main.py
│  README.md
│  requirements.txt  
├─function
│  │  calcu_DHI_512.py
│  │  calcu_signal_strength.py
│  │  custom_transforms_mine.py
│  │  quantitative_analysis.py
│  │  segmentation_optimization.py
│  │  shiyan_jihe_signal_mean_std_plot_function.py
│  │  __init__.py
│  └─__pycache__     
├─input
│  ├─baseline_range
│  │  ├─DH       
│  │  ├─DHI     
│  │  ├─DWR       
│  │  └─SI   
│  └─data_input
│      ├─A40-50
│      │      110105-49007.BMP  
│      ├─B60-70
│      │      300211-63108.BMP  
│      ├─C20-30
│      │      000233-23103.BMP  
│      └─D80-90
│              200301-81106.BMP        
├─network_big_apex
│  └─network_deeplab_upernet_aspp_psp_ab_swin_skips_1288
├─output
│  └─A40-50
└─weights_big_apex
    └─deeplab_upernet_aspp_psp_ab_swin_skips_1288
```

## Requirements

All required libraries are listed in `./requirements.txt`:

```bash
pip3 install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Download and Install Tools

- Download and install **Anaconda** as instructions [here](https://docs.anaconda.com/anaconda/install/index.html).
- Create a new python environment and install the corresponding dependency packages from `./requirements.txt`.

### Quick start

```bash
python3 main.py
```

### Results

The output of the program is located in `output`, and the four subfolders correspond to different hospital
centers. Take `A40-50` as an example. There are 7 files in this folder.

1. The `.xlsx` file is the output quantitative result table. `jihe_parameter` sheet stores the Geometrical parameters of
   IVD( columns indicate L1/L2~L5/S1, and rows indicate DH, DHI, HDR), and `SI_parameter` sheet stores the Signal intensity
   difference of IVD, `quantitative_results` sheet stores the IVD degeneration quantitative analysis results( the first row
   represents the degree of degeneration of IVD, and the next four rows represent the degree of
   deviation of IVD parameters from the baseline parameters).
2. The `.png` file is the visualization file of the intervertebral disc degeneration parameters on the reference pattern spectrum.
3. and the `.BMP` file is the visualization file of the corner detection results.

```
.
├─A40-50
│      110105-49007quantitative_analysis_results.xlsx
│      DH.png
│      DHI.png
│      HDR.png
│      point_detect.BMP
│      SI_dj.png
│      SI_weizhi.png
│    
├─B60-70 
├─C20-30
└─D80-90
```
