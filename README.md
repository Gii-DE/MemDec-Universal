# `MemDec-Universal` <em>(SLM optimized for Local AI)</em> <div style="text-align: right;"><a href="https://github.com/Gii-DE/MemDec-Universal" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Gii--DE-github.com%2FGii--DE%2FMemDec--Universal-darkred?labelColor=black&logo=github&style=plastic&logoColor=white&v=G-7PZXQPZPGX" title="Gii-DE · MemDec-Universal (GitHub)"></a> <a id="contact" href="https://g-dea.com/#contact" target="_blank" rel="noopener noreferrer" style="scroll-margin-top: 80px;"><img src="https://img.shields.io/badge/📧_G--DEA.COM-darkred?style=plastic&v=G-7PZXQPZPGX" alt="g-dea.com" title="Contact Me (Website)"></a></div>

This project implements the [*Memory Decoder (MemDec)*](https://arxiv.org/pdf/2508.09874) approach to extend a base *LLM* (Large Language Model) with domain-specific knowledge through a modular use-case system under [<kbd>MemDec-Universal/`config`</kbd>](config/HowTo.md#config-files), where each domain is defined by its own config files — demonstrated on the German legal domain using the open-source [*Open Legal Data*](https://de.openlegaldata.io/pages/api/) (a non-profit platform providing free API access to German court decisions, laws, and legal texts). The pipeline enables *plug-and-play* domain adaptation across lightweight, instruction-tuned *SLMs* (Small Language Models), including [*Gemma 3-270m*](https://huggingface.co/unsloth/gemma-3-270m-it), [*Qwen 3.5-0.8B*](https://huggingface.co/unsloth/Qwen3.5-0.8B), and [*SmolLM3-3B*](https://huggingface.co/unsloth/SmolLM3-3B), with the active use-case selected via the environment variable `MEMDEC_USECASE` and all pipeline behavior controlled by config files alone. To maximize efficiency, this repository integrates official [*Unsloth models*](https://unsloth.ai/docs/get-started/unsloth-model-catalog). By replacing standard PyTorch layers with custom-written NVIDIA Triton kernels, it achieves 2x to 3x faster training speeds and significantly reduces memory overhead.

## <em>Approach</em>: `MemDec vs. RAG`
MemDec offers a parametric alternative to *Retrieval-Augmented Generation* (RAG) for domain adaptation in LLMs. While RAG retrieves relevant documents from an external vector database at inference time to augment the prompt, it introduces substantial runtime overhead, increased latency, and scaling costs as the datastore grows. In contrast, the MemDec plug-and-play approach eliminates external retrieval by internalizing domain knowledge directly within a lightweight, pretrained transformer decoder. Once trained on a target domain, a single MemDec can be flexibly integrated into any base LLM sharing the same tokenizer with low latency and improved accuracy, as shown in the [paper's results](https://arxiv.org/pdf/2508.09874#page=9) by the reduced perplexity.

## <em>Architecture</em>
### <em>Pipeline</em>
<img src="./assets/project_architecture-pipeline.png?v=G-7PZXQPZPGX" width="50%">

### <em>Data Flow</em>
<img src="./assets/project_architecture-dataFlow.png?v=G-7PZXQPZPGX" width="100%">

---
<a id="main-components"></a>
## <em>Main Components (Features)</em>
<img src="./assets/project_features.png?v=G-7PZXQPZPGX" width="100%">

> [!IMPORTANT]
> All CLI flags listed above are optional overrides: arguments not passed on the command line resolve to their configured values in [<kbd>pipeline_config.yaml</kbd>](config/legal_usecase/pipeline_config.yaml) and [<kbd>dataset_config.yaml</kbd>](config/legal_usecase/dataset_config.yaml), so each step can equally be invoked as `python -m src.<step>` without flags. The complete YAML schemas, parameter descriptions and instructions for defining custom domains are documented in [<kbd>config/`HowTo.md`</kbd>](config/HowTo.md).

---
<a id="project-structure"></a>
## <em>Project Structure <small>(sorted)</small></em>
```bash
MemDec-Universal/
├── test/                              # Test Scripts (Unit Tests)
├── src/                               # Source Code
│   ├── utils.py                             # Utility functions
│   ├── step0_dataset.py                     # Data Preparation & Cleaning
│   ├── step0_data_management.py             # Data Management
│   ├── step1_cleaning.py                    # HF-Dataset Preparation & Cleaning
│   ├── step2_tokenization.py                # Data Tokenization
│   ├── step3_pretraining.py                 # MemDec's Knowledge Base Creation
│   ├── step4_training.py                    # MemDec-Model Training
│   ├── step5_testing.py                     # MemDec-Model Testing
│   └── step5_evaluation.py                  # MemDec-Tests Evaluation
├── MemoryDecoder/                     # MemDec's Training Scripts (used & modified)
│   ├── knn_utils/saveEmbedMulti.py          # Step 3
│   ├── utils/cal_loss.py                    # Step 4
│   ├── train_memdec.py                      # Step 4
│   ├── train_base.py                        # Step 4
│   └── demo/memDec.py                       # Step 5 (Testing)      
├── config/                            # Config Files
│   ├── custom_usecase/                      # Custom Usecase (Template)
│   ├── legal_usecase/                       # German Legal Usecase (Project Example)
│   ├── model_config.json                    # LLM Metadata
│   └── HowTo.md                             # Config-Guide
├── dataset/                           # Datasets (cleaned & tokenized)
├── knowledge_base/                    # KNN Datastore & FAISS Index
├── outputs/                           # Output Files
│   ├── logs/                                # Script Logs (while running "stepX".py)
│   └── step_5000/                           # Trained Model Weights
├── assets/                            # README Media
├── .env.example                       # Environment Variables (Template)
├── .gitattributes                     # Git Attributes (LFS for large files)
├── .gitignore                         # Git Ignore (files to exclude from commit)
├── requirements.txt                   # Project Base Dependencies
├── THIRD-PARTY-NOTICES.txt            # 3rd-Party Software & Dataset Notices
├── LICENSE                            # Project License
└── README.md                          # This file
```

## <em>Initial Setup (Instructions)</em>
The pipeline runs on `Python 3.11` with `PyTorch (CUDA 12.8)` and `Unsloth` for efficient model loading. All pinned dependencies are listed in [<kbd>requirements.txt</kbd>](requirements.txt).

0. <kbd>***Set Environment Variables***</kbd> → [`.env.example`](./.env.example)

1. <kbd>***Create & Activate Virtual Environment***</kbd>
   a) <u>*Using Python*</u>:
   ```powershell
   # Python v3.11 must be pre-installed: https://www.python.org/downloads/
   py -3.11 -m venv memdec-env
   .\memdec-env\Scripts\Activate.ps1
   ```
   b) <u>*Using Conda*</u>:
   ```powershell
   # Anaconda or Miniconda (lightweight ver.) must be pre-installed: 
   # https://www.anaconda.com/docs/getting-started/installation
   conda create -n memdec-env python=3.11
   conda activate memdec-env
   ```

2. <kbd>***Install Project Dependencies***</kbd>
   ```powershell
   # 1) Project base packages
   pip install -r requirements.txt --no-cache-dir
   # 2) CUDA 12.8 toolkit libraries required by PyTorch
   pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 --no-cache-dir
   # For Conda: conda install nvidia::cuda-nvcc nvidia::cuda-runtime -c nvidia
   # 3) PyTorch with CUDA 12.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
   # 4) Unsloth last, so it patches the final PyTorch build
   pip install unsloth
   ```

> [!CAUTION]
> On Windows, the install order is critical: **pip** re-resolves shared dependencies, so installing *PyTorch* or *Unsloth* first can silently upgrade/downgrade the pinned packages from [<kbd><u>requirements.txt</u></kbd>](requirements.txt), resulting in **CUDA DLL** conflicts and memory-isolation crashes (**Hypervisor Errors**).

3. <kbd>***Authenticate with Hugging Face***</kbd>
   <u>*Setup Steps*</u>:
   1. Login to [*Hugging Face*](https://huggingface.co/) to **request access** on the *Open Legal Data* datasets which are used in this project: 
   [<kbd>openlegaldata/court-decisions-germany</kbd>](https://huggingface.co/datasets/openlegaldata/court-decisions-germany) and
   [<kbd>openlegaldata/laws-germany</kbd>](https://huggingface.co/datasets/openlegaldata/laws-germany).
   2. Create an **User Access Token** at [<kbd>huggingface.co/settings/`tokens`</kbd>](https://huggingface.co/settings/tokens) to get your `HF_TOKEN`.
   3. Run the following command in the terminal to authenticate:
   
   ```bash
   huggingface-cli login   # Paste your Access Token when prompted
   ```

> [!IMPORTANT]
> Authentication via `huggingface-cli login` is required to access the gated **Open Legal Data** dataset. However, as this pipeline uses **Unsloth**-optimized model weights (including Gemma 3), there is no need to accept separate model license agreements on HuggingFace.

## <em>Testing</em>
To run **all** test scripts in the [<kbd>MemDec-Universal/`test`</kbd>](./test) directory:

```bash
python -m unittest discover -s test -v
```

> [!IMPORTANT]
> To run **individual** test scripts, refer to the **Terminal Command** column in the [<kbd><u>Main Components</u></kbd>](#main-components) table above.

---
## <em>Contributing & Contact</em>
This project is a config-driven prototype for domain adaptation. While ongoing maintenance is limited, contributions via Pull Requests are highly appreciated! 🤝

Key areas for contribution include:
- <kbd>***New Use-Cases***</kbd>: Add ready-made `<your_domain>_usecase` folders with config files, data sources, and test scenarios for new domains or languages.
- <kbd>***Config Enhancements***</kbd>: Extend `dataset_config.yaml` field mappings or `pipeline_config.yaml` options to cover more source types and pipeline behaviors.
- <kbd>***Framework Robustness***</kbd>: Fix bugs, refine pipeline steps, and expand test coverage or evaluation benchmarks.

For inquiries, technical feedback, or potential collaborations, feel free to reach out via my website's [<kbd><u>contact form</u></kbd>](#contact).

## <em>Credits / Acknowledgements</em>
This is an open-source reimplementation of the research paper 
> **Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models by Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, and Zhouhan Lin (2025)** [<img src="/assets/icon-arxiv.png?v=G-7PZXQPZPGX" alt="MemoryDecoder arxiv Research Paper" width="20" style="margin-left: 4px;">](https://arxiv.org/pdf/2508.09874)[<img src="/assets/icon-pdf.png?v=G-7PZXQPZPGX" alt="MemoryDecoder Slides" width="28">](https://neurips.cc/media/neurips-2025/Slides/119458.pdf) [<img src="/assets/icon-github.png?v=G-7PZXQPZPGX" alt="MemoryDecoder GitHub Repo" width="20">](https://github.com/LUMIA-Group/MemoryDecoder)

trained on the [*Open Legal Data*](https://openlegaldata.io/) [<img src="/assets/icon-arxiv.png?v=G-7PZXQPZPGX" alt="Open Legal Data arxiv Research Paper" width="20" style="margin-left: 4px;">](https://arxiv.org/pdf/2005.13342) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="Open Legal Data Hugging Face Datasets" width="20" style="margin-right: 4px;">](https://huggingface.co/openlegaldata/datasets) datasets for research, testing, and educational purposes. It is built on the open-weights foundation models [*Gemma 3*](https://deepmind.google/models/gemma/gemma-3/) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="Gemma 3 Hugging Face Model" width="20" style="margin-right: 4px;">](https://huggingface.co/google/gemma-3-270m-it), [*Qwen 3.5*](https://qwen.ai/blog?id=qwen3.5) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="Qwen 3.5 Hugging Face Model" width="20" style="margin-right: 4px;">](https://huggingface.co/Qwen/Qwen3.5-0.8B), and [*SmolLM3*](https://smollm3.org/) [<img src="/assets/icon-huggingface.png?v=G-7PZXQPZPGX" alt="SmolLM3 Hugging Face Model" width="20" style="margin-right: 4px;">](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). This repository does not host or redistribute model weights or datasets. The base models provided via [*Unsloth*](https://huggingface.co/unsloth/models) and the gated Open Legal Data datasets must be accessed directly from Hugging Face, subject to their respective licenses and terms of use.

## <em>License</em>
This project is licensed under the <a href="LICENSE"><img src="https://img.shields.io/badge/Apache_2.0-darkred?style=plastic&v=G-7PZXQPZPGX" alt="Apache 2.0 (License)" title="Apache 2.0 (License)" style="position: relative; top: 2px;"></a>.

### <em>[`Third-Party Code`](THIRD-PARTY-NOTICES.txt)</em>
This repository includes and modifies code from the official Memory Decoder (MemDec) repository, vendored under [<kbd>MemDec-Universal/`MemoryDecoder`</kbd>](/MemoryDecoder/) and licensed under the <a href="MemoryDecoder/LICENSE"><img src="https://img.shields.io/badge/Apache_2.0-black?style=plastic&v=G-7PZXQPZPGX" alt="Apache 2.0 (License)" title="Apache 2.0 (License)" style="position: relative; top: 2px;"></a>. All modified files are listed in the [<kbd><u>Project Structure</u></kbd>](#project-structure) tree above, and each one carries a header notice documenting the changes made.

<footer style="margin-top: 40px;">
  <small>
    © 2026
    <a href="https://g-dea.com" target="_blank" rel="noopener noreferrer" title="Gii-DE · Personal Website">
      <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6664f3de18f429c92609f06b/CLe4qN0b0GqJdbG1Ic38a.jpeg?v=G-7PZXQPZPGX"
           alt="Own Logo" width="30" height="30" align="absmiddle"
           style="border-radius:50%; object-fit:cover; margin-left:5px;">
    </a>
  </small>
</footer>
