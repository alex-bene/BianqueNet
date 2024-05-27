import copy
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from function.calcu_DHI_512 import calcu_DHI
from function.calcu_signal_strength import calcu_Sigs
from function.custom_transforms_mine import Normalize_img, ToTensor_img
from function.quantitative_analysis import quantitative_analysis
from function.segmentation_optimization import seg_opt
from function.shiyan_jihe_signal_mean_std_plot_function import scatter_mean_std
from network_big_apex import (
    network_deeplab_upernet_aspp_psp_ab_swin_skips_1288 as network,
)
from streamlit_app import download_bianquenet, get_metadata

SHOW_IMAGES = False
DATA_INPUT_DIR = os.path.join(".", "input", "data_input")
RESULTS_OUTPUT_DIR = os.path.join(".", "output")
QUANTITATIVE_ANALYSIS_RESULTS_FILENAME = "quantitative_analysis_results.xlsx"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DualCompose_img:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def clahe_cv(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    output_cv = cv2.merge([b, g, r])
    return output_cv


def load_model(model_name, device):
    model_map = {
        "deeplabv3_resnet50": network.deeplabv3_resnet50,
        "deeplabv3plus_resnet50": network.deeplabv3plus_resnet50,
        "deeplabv3_resnet101": network.deeplabv3_resnet101,
        "deeplabv3plus_resnet101": network.deeplabv3plus_resnet101,
        "deeplabv3_mobilenet": network.deeplabv3_mobilenet,
        "deeplabv3plus_mobilenet": network.deeplabv3plus_mobilenet,
    }
    model = model_map[model_name](num_classes=14, output_stride=16)
    model = torch.nn.DataParallel(model)  # type: ignore
    # load model weights
    model_weight_path = os.path.join(
        ".",
        "weights_big_apex",
        "deeplab_upernet_aspp_psp_ab_swin_skips_1288",
        "deeplab_upernet_aspp_psp_ab_swin_skips_1288_0.0003.pth",
    )
    download_bianquenet(model_weight_path)
    model.load_state_dict(
        torch.load(model_weight_path, map_location=torch.device("cpu"))
    )
    model.eval()
    model.to(device)

    return model


if __name__ == "__main__":
    if not os.path.exists(RESULTS_OUTPUT_DIR):
        os.mkdir(RESULTS_OUTPUT_DIR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model("deeplabv3plus_resnet101", device)
    image_only_transform = DualCompose_img([ToTensor_img(), Normalize_img()])

    img_list = os.listdir(DATA_INPUT_DIR)

    with torch.no_grad():
        for im_name in img_list:
            data_output_dir = os.path.join(
                RESULTS_OUTPUT_DIR, im_name.split(".")[0]
            )
            if os.path.exists(data_output_dir):
                req_files = [
                    "DH.png",
                    "DHI.png",
                    "HDR.png",
                    "point_detect.BMP",
                    "SI_dj.png",
                    "SI_weizhi.png",
                    "quantitative_analysis_results.xlsx",
                ]
                if all(
                    os.path.exists(os.path.join(data_output_dir, req_file))
                    for req_file in req_files
                ):
                    continue
                else:
                    shutil.rmtree(data_output_dir)

            os.mkdir(data_output_dir)

            _, input_age, input_sex = get_metadata(im_name)
            input_sex = 0 if input_sex == "Female" else 1
            # im_name_no_suffix = (im_name.split(".")[0]).split("-")[-1]
            # input_age = int(im_name_no_suffix[0:2])
            # input_sex = int(im_name_no_suffix[3])
            im_path = os.path.join(DATA_INPUT_DIR, im_name)
            print("processing " + str(im_path) + "." * 20)
            m_input = np.array(
                Image.open(im_path).resize((512, 512), Image.BILINEAR)
            )
            out_cv = clahe_cv(m_input)
            input_img = image_only_transform(out_cv)
            input_img = torch.unsqueeze(input_img, 0)
            pred_img = model(input_img)
            output = torch.nn.Softmax2d()(pred_img)
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            output_seg_opt = output.clone()
            output_seg_opt = torch.squeeze(output_seg_opt).numpy()
            output = seg_opt(output_seg_opt)
            try:
                jihe_parameter = []
                # time_calcu_DHI_bf = time.time()
                (
                    DHI,
                    DWR,
                    DH,
                    HV,
                    point_big_h,
                    point_big_w,
                    point_fenge_h_big,
                    point_fenge_w_big,
                ) = calcu_DHI(output)
                jihe_parameter.append(DH)
                jihe_parameter.append(DHI)
                jihe_parameter.append(DWR)

                # time_calcu_DHI_af = time.time()
                # time_calcu_DHI = time_calcu_DHI_af - time_calcu_DHI_bf

                point_big_h = np.array(point_big_h)
                point_big_h = point_big_h.flatten()
                point_big_w = np.array(point_big_w)
                point_big_w = point_big_w.flatten()
                point_input_pic = copy.deepcopy(m_input)
                point_size = 1
                point_color = (0, 0, 255)  # BGR
                thickness = 4

                for p in range(len(point_big_h)):
                    point = (point_big_w[p], point_big_h[p])
                    cv2.circle(
                        point_input_pic,
                        point,
                        point_size,
                        point_color,
                        thickness,
                    )

                point_fenge_h_big = np.array(point_fenge_h_big)
                point_fenge_h_big = point_fenge_h_big.flatten()
                point_fenge_w_big = np.array(point_fenge_w_big)
                point_fenge_w_big = point_fenge_w_big.flatten()
                point_size = 1
                point_color = (0, 255, 0)  # BGR
                thickness = 4

                for s in range(len(point_fenge_w_big)):
                    point = (point_fenge_w_big[s], point_fenge_h_big[s])
                    cv2.circle(
                        point_input_pic,
                        point,
                        point_size,
                        point_color,
                        thickness,
                    )
                if SHOW_IMAGES:
                    cv2.imshow("point_input_pic", point_input_pic)
                if not cv2.imwrite(
                    os.path.join(data_output_dir, "point_detect.BMP"),
                    point_input_pic,
                ):
                    print("Could not write image")

                SI_parameter = []
                # time_calcu_Sigs_bf = time.time()
                inputs_Sigs = m_input
                output_Sigs = output.copy()
                SI_big_final, disc_si_dif_final = calcu_Sigs(
                    inputs_Sigs, output_Sigs
                )
                SI_parameter.append(disc_si_dif_final)
                # time_calcu_Sigs_af = time.time()
                # time_calcu_Sigs = time_calcu_Sigs_af - time_calcu_Sigs_bf

                scatter_mean_std(
                    input_sex,
                    input_age,
                    disc_si_dif_final,
                    metric="SI",
                    save_path=data_output_dir,
                )
                scatter_mean_std(
                    input_sex,
                    input_age,
                    DH,
                    metric="DH",
                    save_path=data_output_dir,
                )
                scatter_mean_std(
                    input_sex,
                    input_age,
                    DHI,
                    metric="DHI",
                    save_path=data_output_dir,
                )
                scatter_mean_std(
                    input_sex,
                    input_age,
                    DWR,
                    metric="HDR",
                    save_path=data_output_dir,
                )
                # scatter_mean_std(
                #     data_output_path, input_sex, input_age, disc_si_dif_final,
                #     HD, DHI, DWR
                # )

                quantitative_results = quantitative_analysis(
                    disc_si_dif_final, DH, DHI, DWR, input_sex
                )

                data_jihe_parameter = pd.DataFrame(
                    np.array(jihe_parameter).T,
                    index=["L1~L2", "L2~L3", "L3~L4", "L4~L5", "L5~S1"],
                    columns=["DH", "DHI", "HDR/DWR"],
                )
                data_SI_parameter = pd.DataFrame(
                    np.array(SI_parameter).T,
                    index=["L1~L2", "L2~L3", "L3~L4", "L4~L5", "L5~S1"],
                    columns=["SI"],
                )
                data_quantitative_results = pd.DataFrame(
                    np.array(quantitative_results).T,
                    index=["L1~L2", "L2~L3", "L3~L4", "L4~L5", "L5~S1"],
                    columns=[
                        "SI grade",
                        "SI percentage",
                        "DH percentage",
                        "DHI percentage",
                        "HDR/DWR percentage",
                    ],
                )
                data_quantitative_results["SI grade"] = (
                    data_quantitative_results["SI grade"].astype(int)
                )

                quantitative_analysis_results_output_name_path = os.path.join(
                    data_output_dir, QUANTITATIVE_ANALYSIS_RESULTS_FILENAME
                )
                with pd.ExcelWriter(
                    quantitative_analysis_results_output_name_path
                ) as writer:
                    data_jihe_parameter.to_excel(
                        writer, "jihe_parameter", float_format="%.5f"
                    )
                    data_SI_parameter.to_excel(
                        writer, "SI_parameter", float_format="%.5f"
                    )
                    data_quantitative_results.to_excel(
                        writer, "quantitative_results", float_format="%.5f"
                    )

            except Exception as e:
                print(e)
                print(f"----------the calculation of {im_path} picture failed!")
                # pass
                # continue
