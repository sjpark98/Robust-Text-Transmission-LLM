# Robust Transmission of Punctured Text with Large Language Model-based Recovery

This repository contains the official implementation of the paper **"Robust Transmission of Punctured Text with Large Language Model-based Recovery"**.

## Official Paper Link:
https://ieeexplore.ieee.org/document/11112524

## 📄 Abstract

This project proposes a novel text transmission model that selects and transmits only a few critical characters and recovers the missing characters at the receiver using a Large Language Model (LLM). We introduce a novel **Importance Character Extractor (ICE)**, which strategically selects transmitted characters to maximize the recovery performance of the LLM.

Key features:

  - **ICE (Importance Character Extractor):** Selects characters based on importance scores to minimize ambiguity.
  - **LLM-based Recovery:** Uses GPT-3.5 Turbo to reconstruct the full text from punctured inputs.
  - **Robustness:** Demonstrates robust performance across different datasets and tasks compared to traditional and semantic communication models.

## 🛠️ Environment Setup (Recommended)

This project relies on specific library versions for reproducibility, especially **Sionna v0.19.2**. We highly recommend using the provided Conda environment file.

### Option 1: Using Conda (Preferred)

You can easily set up the environment using the provided `environment.yml` file.

```bash
# 1. Create the environment from the file
conda env create -f environment.yml

# 2. Activate the environment
conda activate punctured_text
```

**Note:** The `environment.yml` file includes `sionna==0.19.2` in the pip dependencies section, ensuring the correct version is installed.

### Option 2: Manual Installation (Pip)

If you are not using Conda, please ensure you install the specific version of Sionna to avoid compatibility issues.

```bash
# Crucial: Install specific Sionna version
pip install sionna==0.19.2

# Install other dependencies
pip install tensorflow torch transformers numpy pandas openai nltk symspellpy
```

## 🚀 Usage

### 1. API Key Configuration

To use the LLM-based recovery (GPT-3.5 Turbo), you need a valid OpenAI API key.
Open `config.py` and set your API key:

```python
# config.py
def def_param():
    param = dict()
    # Replace with your actual OpenAI API Key
    param['api_key'] = "YOUR_OPENAI_API_KEY_HERE" 
    ...
```

### 2. Running the Simulation

You can run the main simulation script to evaluate the proposed model's performance.

**Note:** The simulation is currently configured to perform a comparative experiment between the **Proposed Filter** and a **Random Filter** baseline. This allows you to directly observe the performance gains achieved by the importance score-based selection strategy.

```bash
python main.py
```

This script will:

1.  Load the test dataset (`dataset/sentences.csv`).
2.  Perform character puncturing using the ICE algorithm (**Proposed**) and Random selection (**Random**).
3.  Simulate transmission over a physical channel.
4.  Recover the sentences using the GPT solver.
5.  Calculate and print Similarity and BLEU scores for both methods to compare their performance.

## 📂 Project Structure

  - **`main.py`**: Entry point for the simulation. Handles data loading, model execution, and result evaluation.
  - **`model.py`**: Contains the implementation of the `Proposed` model (ICE, puncturing, marking) and the `Traditional` communication model.
  - **`solver.py`**: Implements the `GptSolver` class to interact with the OpenAI API for sentence restoration.
  - **`tools.py`**: Utility functions for:
      - ICE algorithm (importance score calculation).
      - Bit encoding/decoding.
      - Channel simulation (using Sionna).
      - Evaluation metrics (BLEU, BERT similarity).
  - **`config.py`**: Configuration file for simulation parameters (SNR, code rate, block length, etc.).
  - **`environment.yml`**: Conda environment file containing all dependencies.


## 🔗 Citation

If you use this code for your research, please cite our paper:

```bibtex
@ARTICLE{11112524,
  author={Park, Sojeong and Noh, Hyeonho and Yang, Hyun Jong},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Robust Transmission of Punctured Text With Large Language Model-Based Recovery}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Feature extraction;Training;Semantic communication;Symbols;Indexes;Decoding;Data models;Receivers;Information filters;Signal to noise ratio;Large language model (LLM);text transmission;data-independent;robust transmission;semantic communication},
  doi={10.1109/TVT.2025.3595593}}

```
