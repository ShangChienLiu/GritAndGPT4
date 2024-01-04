<div align="center">
<h1>
   <img src="https://img.icons8.com/pulsar-color/96/markdown.png" width="100" height="100" />
   <br>
   GRITANDGPT4
</h1>
<h3>◦ Power up your AI with GritAndGPT4!</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=flat&logo=JavaScript&logoColor=black" alt="JavaScript">
<img src="https://img.shields.io/badge/Prettier-F7B93E.svg?style=flat&logo=Prettier&logoColor=black" alt="Prettier">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat&logo=YAML&logoColor=white" alt="YAML">
<img src="https://img.shields.io/badge/Jest-C21325.svg?style=flat&logo=Jest&logoColor=white" alt="Jest">
<img src="https://img.shields.io/badge/C-A8B9CC.svg?style=flat&logo=C&logoColor=black" alt="C">
<img src="https://img.shields.io/badge/ESLint-4B32C3.svg?style=flat&logo=ESLint&logoColor=white" alt="ESLint">

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/TypeScript-3178C6.svg?style=flat&logo=TypeScript&logoColor=white" alt="TypeScript">
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=flat&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
<img src="https://img.shields.io/badge/GitHub-181717.svg?style=flat&logo=GitHub&logoColor=white" alt="GitHub">
<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
</p>

![license](https://img.shields.io/github/license/kennysuper007/GritAndGPT4?style=flat&labelColor=E5E4E2&color=869BB3)
![repo-language-count](https://img.shields.io/github/languages/count/kennysuper007/GritAndGPT4?style=flat&labelColor=E5E4E2&color=869BB3)
![repo-top-language](https://img.shields.io/github/languages/top/kennysuper007/GritAndGPT4?style=flat&labelColor=E5E4E2&color=869BB3)
![last-commit](https://img.shields.io/github/last-commit/kennysuper007/GritAndGPT4?style=flat&labelColor=E5E4E2&color=869BB3)
</div>

---

## 🔗 Quick Links
- [🔗 Quick Links](#-quick-links)
- [📍 Overview](#-overview)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
  - [⚙️ Installation](#️-installation)
  - [🤖 Running GritAndGPT4](#-running-gritandgpt4)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

GritAndGPT4 is a project designed to harness the capabilities of the GRIT (Generative Region-to-Text Transformer for Object Understanding) model to capture object descriptions. These descriptions are then processed by GPT-4, which generates safety suggestions to enhance driver awareness.

---

## 📂 Repository Structure

```sh
└── GritAndGPT4/
    ├── LICENSE 2
    ├── configs/
    │   ├── Base.yaml
    │   ├── GRiT_B_DenseCap.yaml
    │   ├── GRiT_B_DenseCap_ObjectDet.yaml
    │   ├── GRiT_B_ObjectDet.yaml
    │   ├── GRiT_H_ObjectDet.yaml
    │   └── GRiT_L_ObjectDet.yaml
    ├── datasets/
    ├── demo.py
    ├── demo_images/
    ├── detectron2/
    │   ├── .circleci/
    │   │   ├── config.yml
    │   │   └── import-tests.sh
    │   ├── .clang-format
    │   ├── configs/
    │   │   ├── Base-RCNN-C4.yaml
    │   │   ├── Base-RCNN-DilatedC5.yaml
    │   │   ├── Base-RCNN-FPN.yaml
    │   │   ├── Base-RetinaNet.yaml
    │   │   ├── COCO-Detection/
    │   │   ├── COCO-InstanceSegmentation/
    │   │   ├── COCO-Keypoints/
    │   │   ├── COCO-PanopticSegmentation/
    │   │   ├── Cityscapes/
    │   │   ├── Detectron1-Comparisons/
    │   │   ├── LVISv0.5-InstanceSegmentation/
    │   │   ├── LVISv1-InstanceSegmentation/
    │   │   ├── Misc/
    │   │   ├── PascalVOC-Detection/
    │   │   ├── common/
    │   │   ├── new_baselines/
    │   │   └── quick_schedules/
    │   ├── datasets/
    │   │   ├── prepare_ade20k_sem_seg.py
    │   │   ├── prepare_cocofied_lvis.py
    │   │   ├── prepare_for_tests.sh
    │   │   └── prepare_panoptic_fpn.py
    │   ├── demo/
    │   │   ├── demo.py
    │   │   └── predictor.py
    │   ├── detectron2/
    │   │   ├── checkpoint/
    │   │   ├── config/
    │   │   ├── engine/
    │   │   ├── evaluation/
    │   │   ├── export/
    │   │   ├── layers/
    │   │   ├── model_zoo/
    │   │   ├── modeling/
    │   │   ├── projects/
    │   │   ├── solver/
    │   │   ├── structures/
    │   │   ├── tracking/
    │   │   └── utils/
    │   ├── dev/
    │   │   ├── linter.sh
    │   │   ├── packaging/
    │   │   ├── parse_results.sh
    │   │   ├── run_inference_tests.sh
    │   │   └── run_instant_tests.sh
    │   ├── docker/
    │   │   ├── Dockerfile
    │   │   ├── deploy.Dockerfile
    │   │   └── docker-compose.yml
    │   ├── projects/
    │   │   ├── DeepLab/
    │   │   ├── DensePose/
    │   │   ├── Panoptic-DeepLab/
    │   │   ├── PointRend/
    │   │   ├── PointSup/
    │   │   ├── Rethinking-BatchNorm/
    │   │   ├── TensorMask/
    │   │   └── TridentNet/
    │   ├── setup.py
    ├── grit/
    │   ├── config.py
    │   ├── custom_solver.py
    │   ├── evaluation/
    │   │   └── eval.py
    │   ├── modeling/
    │   │   ├── backbone/
    │   │   ├── meta_arch/
    │   │   ├── roi_heads/
    │   │   ├── soft_nms.py
    │   │   └── text/
    │   └── predictor.py
    ├── ks_demo_predictions/
    │   ├── 0046.txt
    │   ├── 0046_gpt-4_out.txt
    │   ├── 0117.txt
    │   ├── 0117_gpt-4_out.txt
    │   ├── 0158.txt
    │   ├── 0158_gpt-4_out.txt
    │   ├── 0226.txt
    │   ├── 0226_gpt-4_out.txt
    │   ├── 0309.txt
    │   ├── 0309_gpt-4_out.txt
    │   ├── 0334.txt
    │   ├── 0334_gpt-4_out.txt
    │   ├── 0422.txt
    │   ├── 0422_gpt-4_out.txt
    │   ├── 0518.txt
    │   ├── 0518_gpt-4_out.txt
    │   ├── 0623.txt
    │   ├── 0623_gpt-4_out.txt
    │   ├── 0638.txt
    │   ├── 0638_gpt-4_out.txt
    │   ├── 0803.txt
    │   ├── 0803_gpt-4_out.txt
    │   ├── 0831.txt
    │   └── 0831_gpt-4_out.txt
    ├── ks_demo_safety/
    ├── ks_demo_visualizations/
    ├── lauch_deepspeed.py
    ├── node_modules/
    │   └── yaml/
    │       ├── browser/
    │       ├── package.json
    │       └── util.js
    ├── predictions/
    │   ├── 1.txt
    │   ├── 1_gpt-4_out.txt
    │   ├── 2.txt
    │   ├── 2_gpt-4_out.txt
    │   ├── 85building.txt
    │   ├── 85building_gpt-4_out.txt
    │   ├── KS000455.txt
    │   ├── KS000455_gpt-4_out.txt
    │   ├── KS002508.txt
    │   ├── KS002508_gpt-4_out.txt
    │   ├── KS003229.txt
    │   ├── KS003229_gpt-4_out.txt
    │   ├── Kaohsiung_highschool_N.txt
    │   ├── Kaohsiung_highschool_N_gpt-4_out.txt
    │   ├── Kaohsiung_port.txt
    │   ├── Kaohsiung_port_gpt-4_out.txt
    │   ├── beef.txt
    │   ├── beef_gpt-4_out.txt
    │   ├── black_hole.txt
    │   ├── black_hole_gpt-4_out.txt
    │   ├── crowded traffic.txt
    │   ├── crowded traffic_gpt-4_out.txt
    │   ├── dock.txt
    │   ├── dock_gpt-4_out.txt
    │   ├── dog.txt
    │   ├── dog_gpt-4_out.txt
    │   ├── door.txt
    │   ├── door_gpt-4_out.txt
    │   ├── game.txt
    │   ├── game_gpt-4_out.txt
    │   ├── giant_brick.txt
    │   ├── giant_brick_gpt-4_out.txt
    │   ├── kaohsiung_engaging.txt
    │   ├── kaohsiung_engaging_gpt-4_out.txt
    │   ├── new_scence.txt
    │   ├── new_scence_gpt-4_out.txt
    │   ├── ntu_door.txt
    │   ├── ntu_door_gpt-4_out.txt
    │   ├── police car.txt
    │   ├── police car_gpt-4_out.txt
    │   ├── police car_man.txt
    │   ├── police car_man_gpt-4_out.txt
    │   ├── police.txt
    │   ├── police_gpt-4_out.txt
    │   ├── rocket.txt
    │   ├── rocket1.txt
    │   ├── rocket1_gpt-4_out.txt
    │   ├── rocket_gpt-4_out.txt
    │   ├── science_building.txt
    │   ├── science_building2.txt
    │   ├── science_building2_gpt-4_out.txt
    │   ├── science_building3.txt
    │   ├── science_building3_gpt-4_out.txt
    │   ├── science_building_gpt-4_out.txt
    │   ├── space1.txt
    │   ├── space1_gpt-4_out.txt
    │   ├── stone.txt
    │   ├── stone_gpt-4_out.txt
    │   ├── strange_window_building.txt
    │   ├── strange_window_building2.txt
    │   ├── strange_window_building2_gpt-4_out.txt
    │   ├── strange_window_building_gpt-4_out.txt
    │   ├── student.txt
    │   ├── student_gpt-4_out.txt
    │   ├── temple.txt
    │   ├── temple2.txt
    │   ├── temple2_gpt-4_out.txt
    │   ├── temple_gpt-4_out.txt
    │   ├── test.txt
    │   ├── test2.txt
    │   ├── test2_gpt-4_out.txt
    │   ├── test3.txt
    │   ├── test3_gpt-4_out.txt
    │   ├── test_gpt-4_out.txt
    │   ├── trip.txt
    │   ├── trip_gpt-4_out.txt
    │   ├── zoo.txt
    │   └── zoo_gpt-4_out.txt
    ├── requirements.txt
    ├── third_party/
    │   └── CenterNet2/
    │       ├── .circleci/
    │       ├── .clang-format
    │       ├── .github/
    │       ├── configs/
    │       ├── datasets/
    │       ├── demo/
    │       ├── detectron2/
    │       ├── dev/
    │       ├── docker/
    │       ├── projects/
    │       ├── setup.py
    ├── train_deepspeed.py
    ├── train_net.py
    ├── visualizations/
    └── 指令.txt

```

</details>

---

## 🚀 Getting Started

### ⚙️ Installation

1. Clone the GritAndGPT4 repository:
```sh
git clone https://github.com/kennysuper007/GritAndGPT4
```

2. Change to the project directory:
```sh
cd GritAndGPT4
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🤖 Running GritAndGPT4
Use the following command to run GritAndGPT4:
```sh
python demo.py --test-task DenseCap --config-file configs/GRiT_B_DenseCap_ObjectDet.yaml --input demo_images --output visualizations --output_preds predictions --LLM gpt-4 --opts MODEL.WEIGHTS models/grit_b_densecap_objectdet.pth
```
---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/kennysuper007/GritAndGPT4/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/kennysuper007/GritAndGPT4/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/kennysuper007/GritAndGPT4/issues)**: Submit bugs found or log feature requests for GritAndGPT4.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone <your-forked-repo-url>
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear and concise message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

## 📄 License


This project is protected under the MIT License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 👏 Acknowledgments

- I modified the GPT-4 prompt to improve the quality and safety of the suggestions based on Kinan's code.
  
---
