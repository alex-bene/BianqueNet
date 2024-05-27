import os
import shutil
import subprocess
from datetime import datetime
from os import makedirs
from os.path import exists as pexists
from os.path import join as pjoin
from typing import Tuple

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image

from function.shiyan_jihe_signal_mean_std_plot_function import scatter_mean_std


def download_bianquenet(download_path):
    REPO_ID = "alexbene/BianqueNet"
    FILENAME = "deeplab_upernet_aspp_psp_ab_swin_skips_1288_0.0003.pth"

    if os.path.exists(download_path):
        return

    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=os.path.dirname(download_path),
    )


DATETIME_FORMAT = "%Y%m%d"

# PATHS
INPUT_DIR = pjoin("./", "input", "data_input")
OUTPUT_DIR = pjoin("./", "output")

# Create input and output directories if they don't exist
makedirs(INPUT_DIR, exist_ok=True)
makedirs(OUTPUT_DIR, exist_ok=True)

# Input files name template
# D{date.strftime("%Y%m%d")}-A{age:03d}-S{"F" if sex == "Female" else "M"}{ext}


# Parse datetime object from string
def parse_date(dt) -> datetime:
    return datetime.strptime(dt, DATETIME_FORMAT)


# Get updated filename from age, sex, date and old filename
def get_updated_filename(filename, age, sex, date) -> str:
    _, ext = os.path.splitext(filename)
    write_sex = "F" if sex == "Female" else "M"
    return f"D{date.strftime(DATETIME_FORMAT)}-A{age:03d}-S{write_sex}{ext}"


# Get date, age, sex from filename
def get_metadata(filename: str) -> Tuple[datetime, int, str]:
    filename_parts = filename.split(".")[0].split("-")
    date, age, sex = datetime.now(), int(-1), str("Unknown")

    if (
        len(filename_parts) != 3
        or len(filename_parts[0]) != 9
        or len(filename_parts[1]) != 4
        or len(filename_parts[2]) != 2
        or filename_parts[0][0] != "D"
        or filename_parts[1][0] != "A"
        or filename_parts[2][0] != "S"
    ):
        return date, age, sex

    if len(filename_parts[0]) == 9:
        date = parse_date(filename_parts[0][1:])
    if len(filename_parts[1]) == 4:
        age = int(filename_parts[1][1:])
    if len(filename_parts[2]) == 2:
        sex = "Female" if filename_parts[2][1:] == "F" else "Male"

    return date, age, sex


def update_multiselect_state(name, updated=None, deleted=None):
    if not updated and not deleted:
        return
    if deleted:
        if deleted in st.session_state[name]:
            st.session_state[name].remove(deleted)
        st.session_state[f"{name}_state"] = sorted(st.session_state[name])
    elif updated:
        st.session_state[f"{name}_state"] = [
            v if v != updated[0] else updated[1] for v in st.session_state[name]
        ]


# Define Streamlit app
def app():
    st.title("Spline Analysis")
    st.markdown(
        "The system expect [spine sagittal T2-weighted MRI images](https://www.google.com/search?sca_esv=668e1abf6fdfcd6c&sca_upv=1&sxsrf=ADLYWIKmLutts_nNNe55bLngCChY73LyDA:1716834734152&q=spine+sagittal+T2-weighted+MRI+images&uds=ADvngMg3XiBUUido4UvvhKSfTjoFyMqEg7zenUa1FhiE3LIwud3pyBZq0PhDtIOIJUYEKWxP8yVIcw6QDVNzmA6ehh2PjMjY23Z3g9Gh2ld_QPsdsZ2T5_JXsNIkZAKJtUKFWLGcqeGeIUxr9esg6YYcCUgzmgqgmVtAxaxvR0uhierPy2oLXxGXVHpgdGn9nPWlzocXA3YNxG74Lb_9YrdnEPwc8knw3q8IhuJDXEsjdJinSBd9lWJgRNyRS6OBxT4oFphddNXEyClap4xZ_BeGU3ZGxTX-heWzUAe527AjkVe3XfhBpeNvT8tIppylAmM_dtCsLFrkBHeOuWtV4m5pb6z6v3cee1QL-ogr4Wy89PNxoCfoTBM2gbjxLKBCbly1mMUYQ4gK&udm=2&prmd=ivbnz&sa=X&ved=2ahUKEwjp993ju66GAxWY87sIHYIgAmsQtKgLegQIFRAB&biw=1712&bih=1329&dpr=2). "
        "Images should be square and exported directly from the "
        "corresponding software. For correct results, **rename the images "
        " using the integrated `Edit` button to set the correct age and "
        "gender**."
    )
    st.markdown(
        "> **NOTE 📝**\n>\n"
        "> This is just a frontend for the BianqueNet model introduced in "
        '*"Deep learning-based high-accuracy quantitation for lumbar '
        'intervertebral disc degeneration from MRI"* - '
        "[\[paper\]](https://pubmed.ncbi.nlm.nih.gov/35149684/) "
        "[\[original codebase\]](https://github.com/no-saint-no-angel/BianqueNet)"
    )

    # CSS hack to vertically align the columns' content
    st.write(
        """<style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        a {
            color: #5f54c1 !important;
            text-decoration: none;
        }
        blockquote {
            border-left: 2px solid #858585;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ##################### SIDEBAR - Upload Images & Menu #####################
    # Create sidebar for input file selection
    st.sidebar.title("Upload Images")
    with st.sidebar.form("file-uploader", clear_on_submit=True):
        st.sidebar.write(
            r"""<style>[data-testid="stForm"] {
                border: 0px;
                padding: 0px;
            }</style>""",
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Select images",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Save Files", use_container_width=True
        )
        if submitted and uploaded_files is not None:
            for uploaded_file in uploaded_files:
                with open(pjoin(INPUT_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

    st.sidebar.divider()
    st.sidebar.title("Menu")
    st.sidebar.markdown(
        "### - [Overview and Rename](#images-overview-and-rename)\n"
        "### - [Process](#process-images)\n"
        "### - [Results](#results)"
    )
    st.sidebar.divider()

    # ##################### Previw Images - Edit Metadata #####################
    st.header("Images Overview and Rename")

    filenames = sorted(os.listdir(INPUT_DIR))

    if filenames:
        if "previewer_state" not in st.session_state:
            st.session_state.previewer_state = []
        # Get form button state

        container = st.container()
        show_all = st.checkbox(
            "Select all sorted", key="preview_show_all", value=True
        )

        filename_filt = container.multiselect(
            "Which samples to show?",
            filenames,
            filenames if show_all else st.session_state.previewer_state,
            key="previewer",
        )

        for filename in filename_filt:
            # Load and display image
            image = Image.open(pjoin(INPUT_DIR, filename))
            # st.subheader(f"Image {filename}")

            # COLUMN LEFT - Show image and metadata
            left_col, right_col = st.columns([1, 1])
            left_col.image(image, caption=filename, width=300)

            # COLUMN RIGHT - Show metadata and edit button
            # Get current metadata
            date, age, sex = get_metadata(filename)
            # Show current metadata and edit button
            col1, col2 = right_col.columns([2, 1])
            col1.write(f"**Age**: {age}")
            col1.write(f"**Sex**: {sex}")
            col1.write(f"**Date Taken**: {date.strftime('%Y-%m-%d')}")
            edit_button = col2.button("**Edit**", key=f"edit_button_{filename}")
            delete_button = col2.button(
                "**Delete**", key=f"delete_button_{filename}"
            )

            # Delete image if delete button is pressed
            if delete_button:
                os.remove(pjoin(INPUT_DIR, filename))
                # also remove the output folder if exists
                if os.path.exists(pjoin(OUTPUT_DIR, filename.split(".")[0])):
                    shutil.rmtree(pjoin(OUTPUT_DIR, filename.split(".")[0]))
                update_multiselect_state("previewer", deleted=filename)
                update_multiselect_state(
                    "results", deleted=filename.split(".")[0]
                )
                st.rerun()

            # Show form to edit metadata
            # Create form button state if it doesn't exist
            if f"formbtn_state_{filename}" not in st.session_state:
                st.session_state.__setattr__(f"formbtn_state_{filename}", False)
            # Get form button state
            formbtn_state = st.session_state.__getattr__(
                f"formbtn_state_{filename}"
            )
            # Show form if edit button is pressed or form button state is True
            if edit_button and formbtn_state:
                st.session_state.__setattr__(f"formbtn_state_{filename}", False)
            elif edit_button or formbtn_state:
                if formbtn_state:
                    st.session_state.__setattr__(
                        f"formbtn_state_{filename}", False
                    )
                st.session_state.__setattr__(f"formbtn_state_{filename}", True)
                # Show form to edit metadata
                with st.form(filename):
                    form_col1, form_col2, form_col3 = st.columns([0.5, 1, 1])
                    edited_age = form_col1.number_input("Age", value=age)
                    edited_sex = form_col2.selectbox(
                        "Sex",
                        ["Male", "Female"],
                        index=0 if sex == "Male" else 1,
                    )
                    edited_date = form_col3.date_input("Date", value=date)
                    submit_button = st.form_submit_button("Save")
                    # Save edited filename if submit button is pressed
                    if submit_button:
                        updated_filename = get_updated_filename(
                            filename, edited_age, edited_sex, edited_date
                        )
                        os.rename(
                            pjoin(INPUT_DIR, filename),
                            pjoin(INPUT_DIR, updated_filename),
                        )
                        if os.path.exists(
                            pjoin(OUTPUT_DIR, filename.split(".")[0])
                        ):
                            os.rename(
                                pjoin(OUTPUT_DIR, filename.split(".")[0]),
                                pjoin(
                                    OUTPUT_DIR, updated_filename.split(".")[0]
                                ),
                            )
                        update_multiselect_state(
                            "previewer", updated=(filename, updated_filename)
                        )
                        update_multiselect_state(
                            "results", updated=(filename, updated_filename)
                        )
                        st.rerun()

    # ############################ Process Images #############################
    st.divider()
    st.header("Process Images")

    if "model_downloaded" not in st.session_state:
        st.session_state["model_downloaded"] = False

    process_button = st.empty()

    if not st.session_state["model_downloaded"]:
        process_button.warning("Downloading model...")

    # ############################ Preview Results ############################
    st.divider()
    st.header("Results")
    dirnames = sorted(os.listdir(OUTPUT_DIR))
    metrics = ["DH", "DHI", "HDR", "SI"]

    if dirnames:
        if "results_state" not in st.session_state:
            st.session_state.results_state = []
        # Get form button state

        results_container = st.container()
        show_all = st.checkbox(
            "Select all sorted", key="results_show_all", value=True
        )
        # st.write(show_all)
        # st.write(dirnames)

        def hh() -> None:
            st.session_state.results_state = st.session_state.results

        dirnames_filt = results_container.multiselect(
            "Which samples to show?",
            dirnames,
            dirnames if show_all else None,
            key="results",
            on_change=hh,
        )
        for dirname in dirnames_filt:
            st.subheader(f"Sample {dirname}")
            results_dir = pjoin(OUTPUT_DIR, dirname)
            date, age, sex = get_metadata(results_dir)
            image_col, results_col = st.columns([1, 1])

            # Load and display image
            if pexists(pjoin(results_dir, "point_detect.BMP")):
                image = Image.open(pjoin(results_dir, "point_detect.BMP"))
                image_col.image(image, caption="point detections")
            else:
                image_col.error(
                    f"Image {pjoin(results_dir, 'point_detect.BMP')} \
                        can not be found"
                )

            # Read excel results from "quantitative_analysis_results.xlsx"
            quant_analysis_results = pjoin(
                results_dir, "quantitative_analysis_results.xlsx"
            )
            results = None
            results_col.write("Quantitative Results")
            if not pexists(quant_analysis_results):
                df = None
                results_col.error(
                    f"Spreadsheet {quant_analysis_results} can not be found"
                )
            else:
                df = pd.read_excel(
                    quant_analysis_results,
                    index_col=0,
                    sheet_name=[
                        "jihe_parameter",
                        "SI_parameter",
                        "quantitative_results",
                    ],
                )
                results_col.dataframe(
                    df["quantitative_results"], use_container_width=True
                )
                jihe_parameters_df = df["jihe_parameter"]
                results = {
                    "DH": jihe_parameters_df[["DH"]],
                    "DHI": jihe_parameters_df[["DHI"]],
                    "HDR": jihe_parameters_df[["HDR/DWR"]],
                    "SI": df["SI_parameter"],
                }

            tabs = st.tabs(metrics)

            for tab, metric in zip(tabs, metrics):
                plot, table = tab.columns([1.5, 1])
                if metric != "SI":
                    if pexists(pjoin(results_dir, f"{metric}.png")):
                        image = Image.open(pjoin(results_dir, f"{metric}.png"))
                        plot.image(image, use_column_width=True)
                    elif results is not None:
                        plot.pyplot(
                            scatter_mean_std(
                                0 if sex == "Female" else 1,
                                age,
                                results[metric].to_numpy(),
                                metric=metric,
                            )[0]
                        )
                    else:
                        plot.error(
                            f"Image {pjoin(results_dir, f'{metric}.png')} can \
                                not be found"
                        )
                else:
                    if pexists(pjoin(results_dir, "SI_dj.png")) and pexists(
                        pjoin(results_dir, "SI_weizhi.png")
                    ):
                        si1 = Image.open(pjoin(results_dir, "SI_dj.png"))
                        si2 = Image.open(pjoin(results_dir, "SI_weizhi.png"))
                        plot.image(si1)
                        plot.image(si2)
                    elif results is not None:
                        si1, si2 = scatter_mean_std(
                            0 if sex == "Female" else 1,
                            age,
                            results["SI"].to_numpy(),
                            metric="SI",
                        )
                        plot.pyplot(si1)
                        plot.pyplot(si2)
                    else:
                        plot.error(
                            f"Image {pjoin(results_dir, f'SI_dj.png')} can \
                                not be found"
                        )
                        plot.error(
                            f"Image {pjoin(results_dir, f'SI_weizhi.png')} \
                                can not be found"
                        )

                if results is not None:
                    table.dataframe(results[metric], use_container_width=True)
                else:
                    table.error(
                        f"Spreadsheet {quant_analysis_results} can not be found"
                    )

    download_bianquenet(
        os.path.join(
            ".",
            "weights_big_apex",
            "deeplab_upernet_aspp_psp_ab_swin_skips_1288",
            "deeplab_upernet_aspp_psp_ab_swin_skips_1288_0.0003.pth",
        )
    )
    st.session_state["model_downloaded"] = True

    # Process images if process button is pressed
    def on_process_button_click():
        with process_button, st.spinner(
            "Please wait as we process your images..."
        ):
            subprocess.call(["python", "main.py"])

    process_button.button(
        "Process Images",
        type="primary",
        use_container_width=True,
        on_click=on_process_button_click,
    )


if __name__ == "__main__":
    app()
