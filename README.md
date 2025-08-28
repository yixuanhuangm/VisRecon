# VisRecon

Contributed by Yixuan Huang

**A research-oriented toolkit for multi-view 3D reconstruction.**
VisRecon provides utility scripts and self-collected datasets to facilitate multi-view reconstruction experiments based on [COLMAP](https://colmap.github.io/).

## Features

* Utility scripts for **preprocessing, visualization, and handling 3D reconstruction data**.
* **Self-collected multi-view datasets** captured with Intel RealSense D436.
* Demo scripts to **quickly run reconstruction pipelines** and visualize results.

## Dataset

* Captured using **Intel RealSense D436**.
* Contains multiple images per scene suitable for multi-view reconstruction.
* Organized for **direct integration with COLMAP pipelines**.
* **Available datasets:**

  * [Obj3D](https://huggingface.co/datasets/yixuan-huang/Obj3D) – multi-object 3D reconstruction dataset
  * [Plant3D](https://huggingface.co/datasets/yixuan-huang/Plant3D) – plant-focused 3D reconstruction dataset

### How to download datasets

* Download human dataset:

  ```bash
  python download_dataset.py --subset human
  ```
* Download a specific human subset (`human_000`):

  ```bash
  python download_dataset.py --subset human/human_000
  ```
* Download plant dataset:

  ```bash
  python download_dataset.py --repo_id yixuan-huang/Plant3D --subset Plant
  ```

## Installation & Requirements

* **Python 3.10** or later.
* COLMAP installed and accessible from your environment.
* Install Python dependencies:

  ```bash
  pip install -r requirements.txt
  ```
* Standard Python libraries are included in `requirements.txt`: `numpy`, `opencv-python`, `open3d`, `pyrealsense2`, `huggingface-hub`

## Usage Workflow

1. **Download datasets** as shown above.

2. **Modify COLMAP run script** (`run_colmap_pipeline.sh`)

   * Update the dataset folder paths at the top of the script to point to your downloaded images.

3. **Run COLMAP reconstruction**

   ```bash
   bash run_colmap_pipeline.sh
   ```

4. **Visualize results** using the provided visualization scripts:

   ```bash
   python visualizer.py
   ```

## Contributing

* Contributions are welcome! Please open an issue or pull request for **bug fixes, improvements, or new utility scripts**.
* If you use this repository in your research, please **cite accordingly**.

## Contact

If you have any questions about this repository or would like to collaborate with me, feel free to reach out via email at yixuanhm@gmail.com.
