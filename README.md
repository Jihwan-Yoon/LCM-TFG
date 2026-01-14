# LCM-TFG:
# Training-Free Guidance for Latent Consistency Models


This repository implements **Training-Free Guidance (TFG)** methods on top of **Latent Consistency Models (LCM)**.


The goal of this project is to explore how TFG strategies can be applied to few-step inference models like LCM to achieve controllable generation (e.g., color control, symmetry) without fine-tuning the model.


## 📝 Implementation Details


This project partially implements the framework proposed in the **TFG (Unified Training-Free Guidance)** paper, adapted for the LCM scheduler.


### Implemented Features
* **Mean Guidance:** Modifies the latent trajectory using gradients from an external loss function (e.g., RGB loss, Symmetry loss).
* **Recurrent Strategy (Time-Travel):** Applies the guidance iteratively at each timestep to strengthen the effect. 
    * Instead of VE, this implementation utilizes a **Variance Preserving (VP)** re-noising strategy (DDPM Ancestral Sampling) to inject noise during the recurrence loop.
* **Constant $\rho_t$ Schedule:** The guidance scale ($\rho_t$ in the TFG paper) is implemented as a **constant scalar**, fixed by the user input (hyperparameter), rather than a dynamic or adaptive schedule.


### Omitted Features
The following components from the original TFG paper are **NOT** implemented in this repository:
* **Variance Guidance:** Only the mean of the latent distribution is guided.
* **Implicit Dynamics:** The solver follows the standard LCM sampling trajectory with VP re-noising, rather than implicit differentiation strategies.



## 📂 Project Structure


The repository is organized to facilitate experimental tracking and integration with Notion databases.

```text
LCM-TFG-Project/
├── main.py                 # Entry point for running experiments (Multi-GPU support)
├── src/
│   ├── pipeline.py         # Core LCM-TFG generation logic (Custom Pipeline)
│   ├── models.py           # Model loader & Worker class
│   └── losses.py           # Loss functions (Red, Symmetry, Center, etc.)
├── results/                # Output images organized by Loss Name
└── requirements.txt        # Python dependencies
```



## 🚀 Usage


### Installation

```Bash
git clone https://github.com/YOUR_USERNAME_HERE/LCM-TFG.git 
cd LCM-TFG 
pip install -r requirements.txt
```


Running Experiments You can run the experiment using ```main.py```. This script supports **Multi-GPU processing** and automatically generates an **Adaptive Grid** of results (varying scales per recurrence level).


**Basic Run (Uses all available GPUs by default):** 
```Bash
python main.py
```


**Custom Settings (Prompt, Seed, Specific GPUs):** 
```Bash
python main.py --prompt "A cyberpunk city with neon lights" --seed 42 --gpus 0,1
```


### Output Format 
Results are saved in a structure optimized for bulk-uploading to Notion databases or easy filtering: ```results/{Loss_Name}/{LOSS}{RECUR}{SCALE}.png```

Example: ```results/red/RED_R2_S0300.png```



## 🔗 References

Latent Consistency Models: Luo, Simian, et al. "Latent consistency models: Synthesizing high-resolution images with few-step inference." arXiv preprint arXiv:2310.04378 (2023).

Training-Free Guidance: Ye, Haotian, et al. "TFG: Unified Training-Free Guidance for Diffusion Models." arXiv preprint arXiv:2409.15761 (2024).