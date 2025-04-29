# Sym-Q

![overview](Overview.png)
Inspired by how human experts refine and adapt expressions, we introduce Symbolic Qnetwork (Sym-Q), a novel reinforcement learningbased model that redefines symbolic regression as a sequential decision-making task. Sym-Q leverages supervised demonstrations and refines expressions based on reward signals indicating the quality of fitting precision. Its distinctive ability to manage the complexity of expression trees and perform precise step-wise updates significantly enhances flexibility and efficiency. Our results demonstrate that Sym-Q excels not only in recovering underlying mathematical structures but also uniquely learns to efficiently refine the output expression based on reward signals, thereby discovering underlying expressions. Sym-Q paves the way for more intuitive and impactful discoveries in physical science, marking a substantial advancement in the field of symbolic regression. This repository contains the official Pytorch implementation for the paper ["Interactive Symbolic Regression through Offline Reinforcement Learning"](https://www.nature.com/articles/s41467-025-59288-y.pdf) and ["Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making"](https://arxiv.org/abs/2402.05306).

## Workspace Setup

To create the symbolic regression workspace with the necessary repositories, follow these steps:

```cmd
mkdir symbolic_ws
cd symbolic_ws
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git
git clone https://github.com/AILWQ/Joint_Supervised_Learning_for_SR.git
git clone https://github.com/Wenqi-Z/Sym-Q.git
cp -R NeuralSymbolicRegressionThatScales/src/nesymres Joint_Supervised_Learning_for_SR/src
mkdir -p Joint_Supervised_Learning_for_SR/Dataset/2_var/5000000/Train
mkdir -p Joint_Supervised_Learning_for_SR/Dataset/2_var/5000000/SSDNC
```

After running the above commands, your workspace should have the following structure:

```Diff
symbolic_ws
│
└───NeuralSymbolicRegressionThatScales
│
└───Joint_Supervised_Learning_for_SR
│ 
└───Sym-Q
```

## Environment Setup

Create and activate a Conda environment for Sym-Q:

```cmd
conda create -n SymQ python=3.9
conda activate SymQ
pip install -e NeuralSymbolicRegressionThatScales/src/
pip install -r Sym-Q/requirements.txt
cd Joint_Supervised_Learning_for_SR
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Dataset Creation

Generate the dataset for training:

```cmd
cd Joint_Supervised_Learning_for_SR/data_creation
python add_points_to_json.py
python gen_SSDNC_benchmark.py
```

## Quick Start

Train the Sym-Q model:

```cmd
cd Sym-Q
python train.py
```

This generates a folder `/model/GENERATION_TIME` with the model weights and evaluation data. Generate beamsearch results for evaluation:

```
python beam_search.py --weights_path PATH_TO_WEIGHTS --target ssdnc_r2
```

## V2

To enhance flexibility and scalability, we have modularized the pipeline, allowing the model to utilize various encoders and handle inputs with different dimensions. For more detailed information, please refer to `Sym-Q/symq_v2`.

## Acknowledgment

Our implementation is mainly refers to the following codebases. We gratefully thank the authors for their wonderful works.

[SymbolicGPT](https://github.com/mojivalipour/symbolicgpt), [DSO](https://github.com/brendenpetersen/deep-symbolic-optimization), [NeSymReS](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales), [T-JSL](https://github.com/AILWQ/Joint_Supervised_Learning_for_SR)

## Citing this work

If you found our work useful and used code, please use the following citation:
```
@article{tian2025interactive,
  title={Interactive symbolic regression with co-design mechanism through offline reinforcement learning},
  author={Tian, Yuan and Zhou, Wenqi and Viscione, Michele and Dong, Hao and Kammer, David S and Fink, Olga},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={3930},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

```
@article{tian2024sym,
  title={Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making},
  author={Tian, Yuan and Zhou, Wenqi and Dong, Hao and Kammer, David S and Fink, Olga},
  journal={arXiv preprint arXiv:2402.05306},
  year={2024}
}
```
## Supplementary material
[PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-59288-y/MediaObjects/41467_2025_59288_MOESM1_ESM.pdf)
