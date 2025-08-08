# Pre-trained LLMs Translation Project

This repository contains Python scripts for machine translation using pre-trained Large Language Models (LLMs) from Hugging Face Transformers.

## Files

- `translator.py` - English to German translation using Helsinki-NLP model
- `translator_alternative.py` - Alternative translation using Facebook's NLLB model
- `text_generation.py` - Text generation script
- `summarize.py` - Text summarization script
- `guiding_text_gen.py` - Guided text generation script

## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd LLMs
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Translation Example

Run the translator script:
```bash
python "Pre-trained LLMs/translator.py"
```

Or use the alternative model:
```bash
python "Pre-trained LLMs/translator_alternative.py"
```

## Requirements

- Python 3.8+
- PyTorch 2.6+
- Transformers library
- SentencePiece (for tokenization)

## Notes

- The first run will download the pre-trained models (this may take a few minutes)
- Models are cached locally after the first download
- Make sure you have sufficient disk space for the models

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.
