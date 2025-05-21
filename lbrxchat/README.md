# LBRXCHAT - LIBRAXIS Advanced Chat Framework

![LBRXCHAT Logo](https://via.placeholder.com/800x200/0a0a0a/ffffff?text=LBRXCHAT)

> "Coding that takes months, we do in days, and weeks in hours" - Vibecoding Philosophy

## 🔥 Overview

LBRXCHAT is an advanced chatbot framework based on RAG (Retrieval-Augmented Generation) with a TUI interface, optimized for MLX models on Apple Silicon. The system combines a modern approach to LLM interaction with local processing capabilities.

### Key Features:

- **TUI Interface** - beautiful, responsive terminal interface built with Textual
- **Native MLX** - full support for MLX models on Apple Silicon
- **LM Studio Integration** - official integration with LM Studio (API and REST fallback)
- **RAG System** - built-in vector engine for knowledge retrieval
- **JIT Loading** - models loaded on demand with TTL for resource efficiency
- **Dual API** - support for both native API and REST

### Screenshots

![LBRXCHAT TUI](https://via.placeholder.com/800x500/1a1a1a/ffffff?text=LBRXCHAT+TUI+Screen)

## 🚀 Installation

### Requirements

- Python 3.10+ (3.12 recommended)
- macOS with Apple Silicon (M1/M2/M3)
- LM Studio with MLX models
- 16GB+ RAM

### Quick Installation

```bash
# Create virtual environment
uv venv -p 3.12

# Activate environment
source .venv/bin/activate

# Install from project
pip install -e .

# Or install dependencies
uv pip install -r requirements.txt
```

## 📊 Usage

### Basic Launch

```bash
python -m lbrxchat.tui
```

### Custom Corpus Configuration

1. Prepare documents in JSONL format
2. Place them in the `corpus/` directory
3. Build the index:

```bash
python -m lbrxchat.tools.build_index --corpus_dir=corpus --output_dir=indexes
```

### LLM Models

LBRXCHAT works with LM Studio and supports the following MLX models:

- Qwen3 (8B/14B/32B)
- Llama3 (8B/70B)
- Mixtral (8x7B)
- Mistral (7B)
- Phi-3 (3.8B/14B)

## 🛠️ Architecture

```
lbrxchat/
├── core/               # Core components
│   ├── rag.py          # RAG system
│   ├── embedding.py    # Embedding handling
│   ├── models.py       # Model management
│   └── config.py       # Configuration
├── ui/                 # User interface
│   ├── tui.py          # Main TUI interface
│   ├── components/     # UI components
│   └── styles.css      # CSS styles
├── data/               # Data management
│   ├── corpus.py       # Corpus handling
│   ├── index.py        # Index management
│   └── vector_store.py # Vector database
└── tools/              # Helper tools
    ├── build_index.py  # Index building
    ├── convert.py      # Data conversion
    └── benchmark.py    # Performance tests
```

## 🔄 VIBECODING Workflow

LBRXCHAT was created according to the VIBECODING philosophy:

1. **Months to days**: What normally takes a month, we do in a day
2. **Weeks to hours**: What normally takes a week, we do in hours
3. **AI-driven**: Programming through prompts instead of manual coding
4. **Iteration through prompts**: Quick fixes and direction changes through new prompts
5. **Focus on creation**: We focus on the creative part, AI handles the details

## 🧩 Extensions

LBRXCHAT is designed modularly for easy extension:

### Adding New Models

```python
from lbrxchat.core.models import register_model

register_model(
    name="my-model-mlx",
    model_type="mlx",
    context_length=8192,
    system_prompt="You are a helpful assistant."
)
```

### Custom Data Sources

```python
from lbrxchat.data.corpus import Corpus

class MyCorpus(Corpus):
    def load(self):
        # Custom implementation of data loading
        pass
```

## 📝 License

LBRXCHAT is available under the MIT license.

## 👥 Team

Project developed by the LIBRAXIS Team.

---

"Programming is an art, and we are artists!" - Team LIBRAXIS 