# `MemDec-Universal` <em>Config-Guide</em>
This guide explains how to configure the MemDec domain adapter for a custom knowledge domain. Following a plugin approach, each domain lives in its own "use-case" folder under <kbd>MemDec-Universal/`config`</kbd> and is picked up by the pipeline automatically, so the config files alone control its behavior without touching the code. For pipeline internals and initial setup instructions, see <kbd>[`README.md`](../README.md)</kbd>.

## <em>Directory Structure</em>

```bash
MemDec-Universal\config\
├── custom_usecase/           # Use-Case Template
│   ├── data/                   # Drop local dataset files here (.json/.jsonl/.csv/.tsv/.gz)
│   ├── dataset_config.yaml     # Define the raw data sources (step 0-1)
│   ├── pipeline_config.yaml    # Define how each pipeline step runs (step 0-5)
│   ├── sources.json            # Reference texts for source-fidelity scoring (step 5)
│   └── test_cases.yaml         # Prompt scenarios for qualitative testing (step 5)
├── legal_usecase/            # German Legal domain (project example)
│   ├── data/ 
│   ├── dataset_config.yaml
│   ├── pipeline_config.yaml
│   ├── sources.json
│   └── test_cases.yaml
├── model_config.json         # Global model registry (shared across all use-cases)
└── HowTo.md                  # This file
```

## <em>Own Use-Case</em>
1. **Copy the template folder** <kbd>config/`custom_usecase`</kbd> and rename it after your domain (e.g., `<your_domain>_usecase` = `medical_usecase`):
    ```bash
    # Windows
    xcopy config\custom_usecase config\<your_domain>_usecase /E /I
    # Linux/Mac
    cp -r config/custom_usecase config/<your_domain>_usecase
    ```

2. **Edit the config files** in <kbd>config/`<your_domain>_usecase`</kbd> (details for each file are explained below):
   - <u>Required:</u> `dataset_config.yaml` and `pipeline_config.yaml`. 
   - <u>Optional:</u> `sources.json` / `test_cases.yaml` (only needed for `step5_testing`, which `step5_evaluation` depends on).

3. **Activate the use-case** in `.env` so that the pipeline loads your use-case folder instead of the default:
   ```bash
   MEMDEC_USECASE='<your_domain>_usecase'    # default: legal_usecase
   ```
> [!TIP]
> **EXAMPLE**<br>
> Check <kbd>config/legal_usecase</kbd> folder for inspiration on how a complete use-case setup looks like.

## <em>System Config-Files</em>
### <strong>[`dataset_config.yaml`](/config/custom_usecase/dataset_config.yaml)</strong>
Declares the raw data sources for your domain, grouped into 3 optional types that can be combined freely. Each source gets a name of your choice, and `pipeline_config.yaml` then lists those names under `steps.step0_data_management.datasets` to select which ones get merged into the corpus:
1. <kbd>local_files</kbd>: Dataset files placed in the <kbd><your_domain>_usecase/`data`</kbd> folder and processed in `step0_dataset`.
2. <kbd>datasets</kbd>: Remote dataset files downloaded from a URL and processed in `step0_dataset`.
3. <kbd>huggingface</kbd>: HuggingFace datasets available under `huggingface.datasets` to be downloaded and processed in `step1`.

#### <em>Field Mapping</em>
Sources name their fields differently, so each dataset entry can define a `field_mapping` that assigns source fields to the universal fields below. Each value lists the source fields whose contents are combined for that universal field, for example `text: ["title", "markdown_content"]` joins the title and the content with a blank line in between. A dot in a field name means the field lives inside a parent object, so `id: ["meta.id"]` reads the **id** entry inside the **meta** section of each record. `step1` stores every cleaned entry with these four fields no matter where the data came from, so the mapping is only needed when a source does not already use these names.

- <kbd>text</kbd>: The document content the model trains on.
- <kbd>meta_category</kbd>: What kind of entry it is, such as a **law**, **court decision**, or **question**, used for grouping and filtering.
- <kbd>meta_date</kbd>: When the entry is from, used for temporal sorting and analysis.
- <kbd>id</kbd>: A unique identifier per entry, which falls back to the row index if not mapped.

> [!TIP]
> **EXAMPLE**<br>
> Check <kbd>[legal_usecase/dataset_config.yaml](/config/legal_usecase/dataset_config.yaml)</kbd> for inspiration on how to create your own dataset collection.

---
### <strong>[`pipeline_config.yaml`](/config/custom_usecase/pipeline_config.yaml)</strong>
The file is split into 2 sections:
- <kbd>pipeline</kbd>: Settings around the corpus being built, like `corpus_name` (the name the merged corpus will carry), `sources_path` (the reference sources the trained MemDec model is checked against), and `models.default_model` (model keyword in `model_config.json` for the base LLM).
- <kbd>steps</kbd>: Per-step values for steps 0-5 that replace the step's default, like `tokenized_data`, `batch_size`, or trained `checkpoint`.

> [!CAUTION]
> - **Unset parameters** fall back to their built-in defaults, e.g. the commented-out <kbd>[`#min_text_length:`](custom_usecase/pipeline_config.yaml)</kbd> resolves to *60*.
> - Each step auto-selects `base_model` by evaluating the sources below in order of priority as a **hierarchy system**, using the first value found:
>
> <kbd>CLI `--model` → `steps.<step>.base_model` → auto-derived from `tokenized_data` & `checkpoint` → `models.default_model` → .env `MEMDEC_MODEL` → system default `gemma3`</kbd>
>
> The CLI `--model` argument takes precedence, followed by `base_model` set under **steps**, then the model auto-derived from the loaded `tokenized_data` or `checkpoint`, `models.default_model`, the `.env` variable `MEMDEC_MODEL`, and `gemma3` as the built-in fallback
> - Parameters marked **[FIXED]** in the CLI cannot be changed, such as `knowledge_base_path`, `output_dir`, and `results_dir`; changing those paths requires editing the source code directly!

> [!TIP]
> **EXAMPLE**<br>
> Check <kbd>[legal_usecase/pipeline_config.yaml](/config/legal_usecase/pipeline_config.yaml)</kbd> for inspiration on how to create your own pipeline orchestration.

---
### <strong>[`sources.json`](/config/custom_usecase/sources.json)</strong>
Optional file that stores reference texts per scenario so `step5_testing` can check whether generated answers stay faithful to your domain material. The **fuzzy score** measures how closely the response wording matches the closest reference text and the **overlap score** measures the share of key terms appearing in both, which supports the human review of the outputs. Its scenario keys must match the scenarios defined in [`test_cases.yaml`](#test_casesyaml), since `step5_testing` loads both files together.

> [!TIP]
> **EXAMPLE**<br>
> Check <kbd>[legal_usecase/sources.json](/config/legal_usecase/sources.json)</kbd> for inspiration on how to create your own reference sources to check model outputs against.

---
### <strong>[`test_cases.yaml`](/config/custom_usecase/test_cases.yaml)</strong>
Defines the **test scenarios** for `step5_testing`, where each scenario has a `title`, a `task` type like `mixed` (all prompt types), `qa` (question answering), `tc` (text completion), `ts` (text summarization), or `cc` (content creation), and a list of **prompt instructions** on which the base model and the MemDec model are compared. 

> [!TIP]
> **EXAMPLE**<br>
> Check <kbd>[legal_usecase/test_cases.yaml](/config/legal_usecase/test_cases.yaml)</kbd> for inspiration on how to create your own test scenarios.

---
### <strong>[`model_config.json`](/config/model_config.json)</strong>
The global model registry shared across all use-cases. Each entry maps a keyword like `gemma3`/`qwen3.5`/`smollm3` to a HuggingFace **name**, a **description**, and optional **dstore**/**index** filenames; entries are only added when registering a model. These files are only required by the model that builds the KNN knowledge base in `step3` and trains the MemDec module in `step4` (typically the smallest family member, e.g. `gemma3`'s 270m variant). Larger members like `gemma3-1b` keep `null` and serve only as comparison baselines in `step5_testing` and `step5_evaluation`.

---
<details>
<summary><kbd><strong>START THE SYSTEM!</strong></kbd></summary>
Once configured, all steps run without extra arguments because parameters are automatically resolved from the <strong>config files</strong>:

<pre><code>python -m src.step0_dataset
python -m src.step0_data_management
python -m src.step1_cleaning
python -m src.step2_tokenization
python -m src.step3_pretraining
python -m src.step4_training
python -m src.step5_testing
python -m src.step5_evaluation
</code></pre>

> 🛑 All steps resolve their parameters from <kbd>[`pipeline_config.yaml`](/config/custom_usecase/pipeline_config.yaml)</kbd>, where commented-out lines show the system defaults. To change one, uncomment and adjust it in the file or override it via CLI arguments. Run any script with `--help` to list all options.
</details>