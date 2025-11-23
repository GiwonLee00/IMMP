# Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning [ICCV 2025]

**Official PyTorch implementation of [*Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning*](https://arxiv.org/abs/2507.04790) [ICCV 2025].**

🏆🎉 **Highlight Paper at ICCV 2025!** 🎉🏆  
> Our paper has been **selected as a Highlight at the International Conference on Computer Vision (ICCV 2025)**

**Giwon Lee\***<sup>1</sup>, **Wooseong Jeong\***<sup>1</sup>, Daehee Park<sup>2</sup>, Jaewoo Jeong<sup>1</sup>, and Kuk-Jin Yoon<sup>1</sup>  
<sup>1</sup>Visual Intelligence Lab., KAIST, Korea  
<sup>2</sup>Intelligent Systems and Learning Lab., DGIST, Korea  

Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose **Interaction-Merged Motion Planning (IMMP)**, a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches.

---

## 1. Preparing Conda Environment

```bash
conda create -n IMMP python=3.9 -y
conda activate IMMP

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python tqdm scipy matplotlib trimesh pyrender pandas peft moviepy decord easydict einops pytorch_lightning torch_geometric dotdict tensorboard
```

---

2. Download Dataset for CrowdNav, ETH_UCY, SIT, and THOR and construct the data folder as shown below
```bash
-------------------------------------------
IMMP/data
--- demonstration (This is for CrowdNav)
------ data_imit.py
--- eth_ucy
------ all
------ eth
------ hotel
------ univ
------ zara1
------ zara2
--- SIT_dataset
------ sit_new
--------- Cafe_street_1-002
--------- Cafe_street_2-001
--------- Cafeteria_1-006
...
--- THOR
------ Exp_2_run_1.mat
------ Exp_2_run_2.mat
------ Exp_2_run_3.mat
------ Exp_2_run_4.mat
------ Exp_2_run_5.mat
-------------------------------------------
```

3. Using the preprocess_script folder to preprocess all raw data
    A. CrowdNav Preprocessing
    ```bash
        PYTHONPATH=. python preprocess_script/CROWDNAV_preprocess.py
        # it will save the processed folder at data_preprocessed/CROWDNAV_processed
    ```

    B. ETH-UCY Preprocessing
    ```bash
        PYTHONPATH=. python preprocess_script/ETH_UCY_preprocess.py
        # it will save the processed folder at data_preprocessed/ETH_UCY_processed
    ```

    C. SIT Preprocessing (It has 3 steps)
    ```bash
        PYTHONPATH=. python preprocess_script/SIT_preprocess_1st.py   # saves to SIT_processed_1st
        PYTHONPATH=. python preprocess_script/SIT_preprocess_2nd.py   # saves to SIT_processed_2nd
        PYTHONPATH=. python preprocess_script/SIT_preprocess_3rd.py   # saves to SIT_processed_3rd
    ```

    D. THOR Preprocessing
    ```bash
        PYTHONPATH=. python preprocess_script/THOR_preprocess.py
        # it will save the processed folder at data_preprocessed/THOR_processed
    ```

4. After preprocessing, the data_preprocessed folder should be constructed as below
```bash
------------------------------------
IMMP/data_preprocessed
--- CROWDNAV_processed
------ train
------ valid
--- ETH_UCY_processed
------ eth
--------- train
--------- val
------ hotel
--------- train
--------- val
------ univ
------ zara1
--------- train
--------- val
------ zara2
--------- train
--------- val
--- SIT_processed
------ SIT_APTP_processed
--------- Cafe_street_1-002_agents_0_to_200
--------- Cafe_street_2-001_agents_0_to_200
...
--- THOR_processed
------ train
------ valid
------------------------------------
```

5. Preparing trained parameters in the source domain of the Game-Theoretic model
```bash
PYTHONPATH=. python tools_baseline/Game_train/train_forecaster_Game.py
PYTHONPATH=. python tools_baseline/Game_train/train_planner_Game.py
```

6. Merging using our IMMP method
```bash
python merger_250206_woo.py
python merged_planner_finetune.py
```

7. Test code
```bash
python planner_test.py
```

---
## Contact

Giwon Lee: dlrldnjs@kaist.ac.kr

Wooseong Jeong: stk14570@kaist.ac.kr

---

# Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{lee2025interaction,
  title={Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning},
  author={Lee, Giwon and Jeong, Wooseong and Park, Daehee and Jeong, Jaewoo and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={28610--28621},
  year={2025}
}
```
