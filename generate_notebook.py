#!/usr/bin/env python3
"""
Generate training_nids_xgboost_multiclass.ipynb from the original notebook.
Applies typo fixes, removes redundant imports, excludes Streamlit app code,
and restructures cells with proper markdown headers.
"""

import json
import copy

ORIGINAL_NB = "kode-skripsi-nf-unsw-nb15-v3 (11).ipynb"
OUTPUT_NB = "training_nids_xgboost_multiclass.ipynb"

# Load original notebook
with open(ORIGINAL_NB, "r", encoding="utf-8") as f:
    orig = json.load(f)

def get_source(cell_idx):
    """Get the full source string of a cell from the original notebook."""
    return "".join(orig["cells"][cell_idx]["source"])

def source_to_list(text):
    """Convert a source string to a list of lines (each ending with \\n except the last)."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    # Remove trailing empty string if the text ended with \n
    if result and result[-1] == "":
        result.pop()
        if result:
            # The previous line already has \n, that's fine
            pass
    return result

def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_to_list(text)
    }

def make_code_cell(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_to_list(text),
        "outputs": [],
        "execution_count": None
    }

def remove_import_lines(src, lines_to_remove):
    """Remove specific import lines from source. Each entry in lines_to_remove
    should be the exact line content (without trailing newline)."""
    src_lines = src.split("\n")
    result = []
    i = 0
    while i < len(src_lines):
        line = src_lines[i]
        stripped = line.strip()
        if stripped in lines_to_remove:
            # Also skip a blank line after the removed import if the next line is also an import or blank
            i += 1
            continue
        result.append(line)
        i += 1

    # Clean up: remove consecutive blank lines that may result from removals
    # but keep the overall structure intact
    text = "\n".join(result)
    # Remove blocks of imports that became empty (just blank lines between comments)
    return text

def remove_redundant_imports_block(src, imports_to_remove):
    """Remove specific import lines from the top portion of a cell (before main code).
    Only removes lines in the import section (before first non-import, non-comment, non-blank line
    that isn't in imports_to_remove)."""
    lines = src.split("\n")
    result = []
    removed_indices = set()

    # First pass: mark lines to remove
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in imports_to_remove:
            removed_indices.add(i)

    # Second pass: build result, cleaning up blank lines around removed imports
    i = 0
    while i < len(lines):
        if i in removed_indices:
            i += 1
            continue
        result.append(lines[i])
        i += 1

    # Clean up excessive blank lines (more than 2 consecutive)
    cleaned = []
    blank_count = 0
    for line in result:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned)


# =========================================================================
# BUILD CELLS
# =========================================================================
cells = []

# --- Cell 0: Markdown title ---
cells.append(make_markdown_cell(
    "# Training NIDS XGBoost Multi-Class — Multi-Objective HPO\n"
    "\n"
    "> **Skripsi**: Network Intrusion Detection System (NIDS) menggunakan XGBoost  \n"
    "> **Dataset**: NF-UNSW-NB15-v3  \n"
    "> **Optimasi**: Multi-Objective Hyperparameter Optimization (TPE, NSGA-II, Random)  \n"
    "> **Objectives**: Maximize Macro F1-Score & Minimize Inference Latency\n"
    "\n"
    "Notebook ini berisi pipeline lengkap mulai dari preprocessing, optimasi hyperparameter,\n"
    "evaluasi model, hingga ekspor artefak untuk deployment."
))

# --- Cell 1: Markdown ---
cells.append(make_markdown_cell("## 1. Setup & Import Libraries"))

# --- Cell 2: Code - Cell 1 with typo fixes ---
cell2_src = get_source(1)
cell2_src = cell2_src.replace("plotly. express", "plotly.express")
cell2_src = cell2_src.replace("scipy. stats", "scipy.stats")
cell2_src = cell2_src.replace("sys.version. split()", "sys.version.split()")
cell2_src = cell2_src.replace("test_model. fit", "test_model.fit")
cells.append(make_code_cell(cell2_src))

# --- Cell 3: Markdown ---
cells.append(make_markdown_cell("## 2. Load Dataset"))

# --- Cell 4: Code - Cell 3 without redundant imports ---
cell4_src = get_source(3)
cell4_imports_to_remove = {
    "import pandas as pd",
    "import numpy as np",
    "import os",
    "import gc",
    "from sklearn.model_selection import train_test_split",
}
cell4_src = remove_redundant_imports_block(cell4_src, cell4_imports_to_remove)
cells.append(make_code_cell(cell4_src))

# --- Cell 5: Markdown ---
cells.append(make_markdown_cell("## 3. Preprocessing (Cleaning, Encoding, Weighting)"))

# --- Cell 6: Code - Cell 5 without redundant imports ---
cell6_src = get_source(5)
cell6_imports_to_remove = {
    "import pandas as pd",
    "import numpy as np",
    "import xgboost as xgb",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.utils.class_weight import compute_sample_weight",
}
cell6_src = remove_redundant_imports_block(cell6_src, cell6_imports_to_remove)
cells.append(make_code_cell(cell6_src))

# --- Cell 7: Markdown ---
cells.append(make_markdown_cell("## 4. Visualisasi Distribusi & Bobot Kelas"))

# --- Cell 8: Code - Cell 7 without redundant imports ---
cell8_src = get_source(7)
cell8_imports_to_remove = {
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import pandas as pd",
    "import numpy as np",
    "from sklearn.utils.class_weight import compute_sample_weight",
}
cell8_src = remove_redundant_imports_block(cell8_src, cell8_imports_to_remove)
cells.append(make_code_cell(cell8_src))

# --- Cell 9: Markdown ---
cells.append(make_markdown_cell("## 5. Definisi Objective Function"))

# --- Cell 10: Code - Cell 9 without redundant imports ---
cell10_src = get_source(9)
cell10_imports_to_remove = {
    "import time",
    "import gc",
    "import xgboost as xgb",
    "import optuna",
    "from sklearn.metrics import f1_score, accuracy_score",
    "import numpy as np",
}
cell10_src = remove_redundant_imports_block(cell10_src, cell10_imports_to_remove)
cells.append(make_code_cell(cell10_src))

# --- Cell 11: Markdown ---
cells.append(make_markdown_cell("## 6. Optimasi Hyperparameter (TPE, NSGA-II, Random)"))

# --- Cell 12: Code - Cell 11 without redundant imports ---
cell12_src = get_source(11)
cell12_imports_to_remove = {
    "import optuna",
    "from optuna.samplers import TPESampler, NSGAIISampler, RandomSampler",
    "import time",
    "import pandas as pd",
    "import warnings",
}
cell12_src = remove_redundant_imports_block(cell12_src, cell12_imports_to_remove)
cells.append(make_code_cell(cell12_src))

# --- Cell 13: Code - Cell 12 (optimization_results container) ---
cells.append(make_code_cell(get_source(12)))

# --- Cell 14: Code - Cell 13 (TPE) ---
cells.append(make_code_cell(get_source(13)))

# --- Cell 15: Code - Cell 14 (NSGA-II) ---
cells.append(make_code_cell(get_source(14)))

# --- Cell 16: Code - Cell 15 (Random) ---
cells.append(make_code_cell(get_source(15)))

# --- Cell 17: Markdown ---
cells.append(make_markdown_cell("## 7. Ekstraksi Parameter Pareto Optimal"))

# --- Cell 18: Code - Cell 16 without redundant import ---
cell18_src = get_source(16)
cell18_imports_to_remove = {
    "import pandas as pd",
}
cell18_src = remove_redundant_imports_block(cell18_src, cell18_imports_to_remove)
cells.append(make_code_cell(cell18_src))

# --- Cell 19: Code - Cell 17 (comparison summary) ---
cells.append(make_code_cell(get_source(17)))

# --- Cell 20: Markdown ---
cells.append(make_markdown_cell("## 8. Evaluasi Model Final"))

# --- Cell 21: Code - Cell 19 without redundant imports ---
cell21_src = get_source(19)
cell21_imports_to_remove = {
    "import pandas as pd",
    "import numpy as np",
    "import time",
    "import xgboost as xgb",
    "import gc",
    "from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix",
    "from sklearn.utils.class_weight import compute_sample_weight",
}
cell21_src = remove_redundant_imports_block(cell21_src, cell21_imports_to_remove)
cells.append(make_code_cell(cell21_src))

# --- Cell 22: Markdown ---
cells.append(make_markdown_cell(
    "## 9. Visualisasi Hasil\n"
    "### 9A. Pareto Front Gabungan"
))

# --- Cell 23: Code - Cell 21 without redundant imports ---
cell23_src = get_source(21)
cell23_imports_to_remove = {
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import optuna",
    "import pandas as pd",
    "import numpy as np",
}
cell23_src = remove_redundant_imports_block(cell23_src, cell23_imports_to_remove)
cells.append(make_code_cell(cell23_src))

# --- Cell 24: Markdown ---
cells.append(make_markdown_cell("### 9B. Confusion Matrix (Raw & Normalized)"))

# --- Cell 25: Code - Cell 23 without redundant imports ---
cell25_src = get_source(23)
cell25_imports_to_remove = {
    "import seaborn as sns",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "from sklearn.metrics import confusion_matrix",
}
cell25_src = remove_redundant_imports_block(cell25_src, cell25_imports_to_remove)
cells.append(make_code_cell(cell25_src))

# --- Cell 26: Markdown ---
cells.append(make_markdown_cell("### 9C. Analisis Statistik (Cohen's Kappa & Error Breakdown)"))

# --- Cell 27: Code - Cell 24 without redundant imports ---
cell27_src = get_source(24)
cell27_imports_to_remove = {
    "import pandas as pd",
    "import numpy as np",
    "from sklearn.metrics import cohen_kappa_score, confusion_matrix",
    "import matplotlib.pyplot as plt",
}
cell27_src = remove_redundant_imports_block(cell27_src, cell27_imports_to_remove)
cells.append(make_code_cell(cell27_src))

# --- Cell 28: Markdown ---
cells.append(make_markdown_cell("### 9D. Optimization Convergence"))

# --- Cell 29: Code - Cell 28 without redundant imports ---
cell29_src = get_source(28)
cell29_imports_to_remove = {
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import numpy as np",
    "import pandas as pd",
    "import optuna",
}
cell29_src = remove_redundant_imports_block(cell29_src, cell29_imports_to_remove)
cells.append(make_code_cell(cell29_src))

# --- Cell 30: Markdown ---
cells.append(make_markdown_cell("## 10. Hyperparameter Importance Analysis"))

# --- Cell 31: Code - Cell 26 without redundant imports (keep RandomForestRegressor) ---
cell31_src = get_source(26)
cell31_imports_to_remove = {
    "import pandas as pd",
    "import numpy as np",
    "import optuna",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
}
cell31_src = remove_redundant_imports_block(cell31_src, cell31_imports_to_remove)
cells.append(make_code_cell(cell31_src))

# --- Cell 32: Markdown ---
cells.append(make_markdown_cell("## 11. Feature Importance Analysis"))

# --- Cell 33: Code - Cell 30 without redundant imports ---
cell33_src = get_source(30)
cell33_imports_to_remove = {
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import pandas as pd",
    "import numpy as np",
    "import xgboost as xgb",
}
cell33_src = remove_redundant_imports_block(cell33_src, cell33_imports_to_remove)
cells.append(make_code_cell(cell33_src))

# --- Cell 34: Markdown ---
cells.append(make_markdown_cell("### Detailed Performance Metrics per Class"))

# --- Cell 35: Code - Cell 32 without redundant imports (keep precision_recall_fscore_support) ---
cell35_src = get_source(32)
cell35_imports_to_remove = {
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import pandas as pd",
    "import numpy as np",
}
cell35_src = remove_redundant_imports_block(cell35_src, cell35_imports_to_remove)
cells.append(make_code_cell(cell35_src))

# --- Cell 36: Markdown ---
cells.append(make_markdown_cell("## 12. Statistical Validation (Cross-Validation)"))

# --- Cell 37: Code - Cell 34 without redundant imports ---
cell37_src = get_source(34)
cell37_imports_to_remove = {
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import time",
    "from sklearn.model_selection import StratifiedKFold, train_test_split",
    "from sklearn.metrics import f1_score",
    "from scipy.stats import kruskal",
    "import xgboost as xgb",
}
cell37_src = remove_redundant_imports_block(cell37_src, cell37_imports_to_remove)
cells.append(make_code_cell(cell37_src))

# --- Cell 38: Markdown ---
cells.append(make_markdown_cell("## 13. Final Summary & Recommendations"))

# --- Cell 39: Code - Cell 36 without redundant imports ---
cell39_src = get_source(36)
cell39_imports_to_remove = {
    "import pandas as pd",
    "import numpy as np",
    "import json",
    "from datetime import datetime",
}
cell39_src = remove_redundant_imports_block(cell39_src, cell39_imports_to_remove)
cells.append(make_code_cell(cell39_src))

# --- Cell 40: Markdown ---
cells.append(make_markdown_cell("## 14. Export Model & Artifacts"))

# --- Cell 41: Code - Cell 37 export logic only (no Streamlit app) ---
cell41_full = get_source(37)
# Find where the Streamlit section starts
streamlit_marker = "\n# ==============================================================================\n# E. BUAT STREAMLIT APP"
streamlit_idx = cell41_full.find(streamlit_marker)
if streamlit_idx == -1:
    raise ValueError("Could not find Streamlit section marker in cell 37")

# Take everything before the Streamlit section
cell41_src = cell41_full[:streamlit_idx].rstrip() + "\n"

# Remove redundant imports (keep only pickle, joblib, shutil which are new)
cell41_imports_to_remove = {
    "import json",
    "import os",
    "import pandas as pd",
    "import numpy as np",
    "import xgboost as xgb",
}
cell41_src = remove_redundant_imports_block(cell41_src, cell41_imports_to_remove)
cells.append(make_code_cell(cell41_src))

# --- Cell 42: Code - Cell 38 (backup/archive) without redundant imports ---
cell42_src = get_source(38)
cell42_imports_to_remove = {
    "import os",
    "import shutil",
    "import glob",
    "from datetime import datetime",
}
cell42_src = remove_redundant_imports_block(cell42_src, cell42_imports_to_remove)
cells.append(make_code_cell(cell42_src))

# =========================================================================
# BUILD NOTEBOOK
# =========================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

# Write output
with open(OUTPUT_NB, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Generated {OUTPUT_NB} with {len(cells)} cells")
