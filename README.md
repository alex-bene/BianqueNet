<a href="https://huggingface.co/alexbene/BianqueNet" target="_blank"><img alt="HF Model Badge" src="https://img.shields.io/badge/huggingface-model-%23FFD21E?logo=huggingface"></a>
<a href="https://huggingface.co/spaces/alexbene/spine_analysis" target="_blank"><img alt="HF Space Badge" src="https://img.shields.io/badge/huggingface-space-%23FF9D00?logo=huggingface"></a>
<a href="https://pubmed.ncbi.nlm.nih.gov/35149684" target="_blank"><img alt="Paper Badge" src="https://img.shields.io/badge/paper-blue?logo=readthedocs&logoColor=white"></a>


# Quantitative Analysis of Intervertebral Disc (IVD) Degeneration
This project is used to perform quantitative analysis of IVD degeneration on spine
sagittal T2-weighted MRI images. The generated reports include:
- Geometrical parameters of IVD
- Signal intensity difference of IVD
- IVD degeneration quantitative analysis results

> **NOTE 📝**
>
> This is just a frontend for the BianqueNet model introduced in 
*"Deep learning-based high-accuracy quantitation for lumbar intervertebral disc
degeneration from MRI"* - [\[paper\]](https://pubmed.ncbi.nlm.nih.gov/35149684/)
[\[original codebase\]](https://github.com/no-saint-no-angel/BianqueNet).
> 
> Slight adjustements have been made to the original code for readability and package version compatibility. Also, docker-related files have been added for easier development.

## Quickstart
### Docker
The project includes the required docker files for running this project.
To run the application, navigate to the project directory and execute:

```bash
docker compose up
```

Then, open your browser at `http://localhost:8501` to see the application.

### Expected Input File Conventions
The system expect [spine sagittal T2-weighted MRI images](https://www.google.com/search?q=spine+sagittal+T2-weighted+MRI+images&sca_esv=668e1abf6fdfcd6c&sca_upv=1&udm=2&biw=1580&bih=1329&sxsrf=ADLYWIJ0KfhITBIB9ndT7IQp3j_F6ghuLQ%3A1716853512820&ei=CBtVZvbZMfKzi-gPo7yHgAs&ved=0ahUKEwj2_Yzega-GAxXy2QIHHSPeAbAQ4dUDCBA&uact=5&oq=spine+sagittal+T2-weighted+MRI+images&gs_lp=Egxnd3Mtd2l6LXNlcnAiJXNwaW5lIHNhZ2l0dGFsIFQyLXdlaWdodGVkIE1SSSBpbWFnZXMyBBAjGCdIzghQAFgAcAF4AJABAJgBAKABAKoBALgBA8gBAPgBApgCAaACApgDAIgGAZIHATGgBwA&sclient=gws-wiz-serp).
Images should be square and exported directly from the corresponding software
(I would advise against screenshots). For correct results, rename the images
using the integrated `Edit` button inside the app to set the correct age and gender.

### Results
The output results are located in `output` folder. For each input file inside
`inputs\data_input`, a subfolder inside `output` directory is created with the
name of the input file. Inside each subfoder are 7 files.

1. The `quantitative_analysis_results.xlsx` file contains the quantitative result table.
    - `jihe_parameter` sheet stores the Geometrical parameters of IVD (columns
    indicate L1/L2~L5/S1, and rows indicate DH, DHI, HDR)
    - `SI_parameter` sheet stores the Signal intensity difference of IVD
    - `quantitative_results` sheet stores the IVD degeneration quantitative
        analysis results (the first row represents the degree of degeneration of
        IVD, and the next four rows represent the degree of deviation of IVD
        parameters from the baseline parameters).
2. Each `.png` file is the visualization file of the IVD degeneration parameters
    on the reference pattern spectrum.
3. The `.BMP` file is the visualization file of the corner detection results.

```
├─output
│ ├─D20230410-A085-SM
│ │ │ quantitative_analysis_results.xlsx
│ │ │ DH.png
│ │ │ DHI.png
│ │ │ HDR.png
│ │ │ point_detect.BMP
│ │ │ SI_dj.png
│ │ └─SI_weizhi.png
```

## Development
I recommend using Visual Studio Code (VS Code) with the [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
extension to work on this project. This allows you to open the project as a
container and have all dependencies automatically set up.

To use VS Code, follow these steps:
1. Open VS Code.
2. Install the  Remote Development extension if you haven't already.
3. Press `Ctrl/Cmd + Shift + P` and select "Dev Containers: Open Folder in Container..."
    from the command palette.
4. Wait for the container to set up and start the Streamlit application.

Once the application is running, you can view it by navigating to
`http://localhost:8501` in your browser and make any changes to the files inside
the container.


## Codebase Structure
There are 5 main directories:
1. the `function` folder contains files for calculating parameters and drawing.
2. the `input` folder contains two subfolders, `baseline_range` folder contains
    statistical benchmark parameters, and `data_input` folder contains examples
    of IVD T2MRI images.
3. `network_big_apex` folder contains different network structures.
4. `output` folder stores the output results.
5. `weights_big_apex folder` contains the training parameters of BianqueNet. You
    can download the weights from the original repo's [pcloud link](https://u.pcloud.link/publink/show?code=XZ6DBkVZPdNHAOg14IHxKVDcnvq4pH1c4b1k) or from the [hugging face hub](https://huggingface.co/alexbene/BianqueNet). The app will auto-download the model when you run it, if it's not
    already downloaded on the expected folder.

```
.
│ main.py
│ README.md
│ requirements.txt  
├─function
│ │ calcu_DHI_512.py
│ │ calcu_signal_strength.py
│ │ custom_transforms_mine.py
│ │ quantitative_analysis.py
│ │ segmentation_optimization.py
│ │ shiyan_jihe_signal_mean_std_plot_function.py
│ │ __init__.py
│ └─__pycache__     
├─input
│ ├─baseline_range
│ │ ├─DH       
│ │ ├─DHI     
│ │ ├─DWR       
│ │ └─SI   
│ └─data_input
│   └─D20230410-A085-SM.BMP  
├─network_big_apex
│ └─network_deeplab_upernet_aspp_psp_ab_swin_skips_1288
├─output
│ └─D20230410-A085-SM
├─weights_big_apex
│ └─deeplab_upernet_aspp_psp_ab_swin_skips_1288
```

## TODO
- [ ] Replace "Date Taken" with a patient ID.
- [ ] When we save uploaded files in the app, assign a user/session id to them
    so that only  the person who uploaded can see them.


## Citation
If you use this code for your research, please cite the [original paper](https://pubmed.ncbi.nlm.nih.gov/35149684/):
```bibtex
@article{zheng2022bianquenet,
  author = {Zheng, Hua-Dong and Sun, Yue-Li and Kong, De-Wei and Yin, Meng-Chen and Chen, Jiang and Lin, Yong-Peng and Ma, Xue-Feng and Wang, Hongshen and Yuan, Guang-Jie and Yao, Min and Cui, Xue-Jun and Tian, Ying-Zhong and Wang, Yong-Jun},
  year = 2022,
  pages = 841,
  title = {Deep learning-based high-accuracy quantitation for lumbar intervertebral disc degeneration from MRI},
  volume = 13,
  journal = {Nature Communications},
}
```
