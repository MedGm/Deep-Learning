# Deep Learning – Labs & Practical Work

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" height="90" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" height="90" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" height="90" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" height="90" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" height="90" />
</p>

This repo groups my **labs** done for the **Deep Learning** module (Pr. Lotfi ELAACHAK).
Each lab is an independent project on a specific topic (MLP, CNN, RNN, ...), with its notebooks, figures and a short report in its own `README.md`.

---
## Goals of this repo
- Keep all my Deep Learning labs in one place
- Make it easy to clone and run the notebooks
- Show the progression : from simple MLPs to more advanced architectures
- Have a base I can reuse for future experiments

---
## Labs

| Lab | Folder | Topic | Main content |
|-----|--------|-------|--------------|
| **Lab 1 – DNN / MLP with PyTorch** | [`Lab1_DNN_PyTorch/`](Lab1_DNN_PyTorch/) | Regression and multi-class classification | NYSE stock prices (regression) and predictive maintenance (5 failure types). EDA, SMOTE, GridSearch with skorch, Loss/Accuracy curves, metrics, regularization (Dropout, L2, BatchNorm, Early stopping). |

> Each folder has its own `README.md` with the goal, the structure, the results and how to run it.

---
## Common tools
| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| ML tools | scikit-learn, skorch, imbalanced-learn |
| Environment | Jupyter Notebook / VS Code / Google Colab |

---
## Installation
```bash
git clone https://github.com/MedGm/Deep-Learning.git
cd Deep-Learning

# virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```
PyTorch CPU is enough for the labs. For a GPU version see [pytorch.org](https://pytorch.org/get-started/locally/).

---
## Running a lab
```bash
cd Lab1_DNN_PyTorch/notebooks
jupyter notebook  # open and run the cells
```
Dataset paths are relative (`../dataset/...`). Some datasets are too big for GitHub, the lab README explains how to download them from Kaggle.

---
## Roadmap
- [x] Lab 1 – DNN / MLP with PyTorch (regression + classification)
- [ ] Next labs (CNN, RNN, ...) will be added during the module

---
## Author
Made by **EL GORRIM Mohamed**
Cycle Ingénieur LSI – Université Abdelmalek Essaâdi, FST Tanger

---
## License
For educational and personal use. Please mention the source if you reuse it publicly.
