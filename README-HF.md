---
title: Medical LLM Assistant
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Medical LLM Assistant

A fine-tuned TinyLlama model specialized for answering medical questions. This assistant provides evidence-based medical information using a lightweight language model trained on medical flashcard data.

## Important Disclaimer

**This is an educational project and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of qualified healthcare professionals for any medical concerns.

## Features

- **Medical Knowledge**: Fine-tuned on Medical Meadow Medical Flashcards dataset
- **Interactive Interface**: User-friendly Gradio interface with customizable parameters
- **Adjustable Parameters**: 
  - Temperature control for response creativity
  - Maximum length control for response detail
- **Example Questions**: Pre-loaded medical questions to get started

## Technical Details

### Model Architecture
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Parameters**: 1.1 billion
- **Training Dataset**: Medical Meadow Medical Flashcards
- **Framework**: Transformers, PEFT, PyTorch

### Training Configuration
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Training Steps**: 975 checkpoints
- **Optimization**: AdamW with cosine learning rate schedule

## Performance

The model has been evaluated on medical question-answering tasks and shows improvement over the base model in:
- Medical terminology accuracy
- Evidence-based responses
- Domain-specific knowledge

## Usage

### Basic Usage

1. Enter your medical question in the text box
2. Adjust temperature and max length if desired
3. Click "Ask Question"
4. View the generated response

### Parameter Guide

- **Temperature (0.1 - 1.0)**: 
  - Lower values (0.1-0.4): More conservative, factual responses
  - Higher values (0.7-1.0): More creative, varied responses
  
- **Max Response Length (50-300 tokens)**:
  - Shorter (50-100): Concise answers
  - Longer (200-300): Detailed explanations

## Local Development

### Setup

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/medical-llm-assistant
cd medical-llm-assistant

# Install dependencies
pip install -r requirements-hf.txt

# Run the app
python app.py
```

### Model Files

The fine-tuned LoRA adapters are stored in the `medical_llm_final` directory. If deploying to Hugging Face Spaces:

1. Upload the adapter files to the Space repository
2. Or, push the adapters to Hugging Face Model Hub and modify `app.py` to load from there

## Project Structure

```
.
├── app.py                    # Gradio interface
├── requirements-hf.txt       # Python dependencies
├── README.md                 # This file
└── medical_llm_final/        # LoRA adapter files (optional)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

## Example Questions

- What are the symptoms of diabetes?
- How does the immune system work?
- What is the difference between bacteria and viruses?
- What are the risk factors for heart disease?
- How do vaccines provide immunity?
- What is the function of hemoglobin?

## Training Details

This model was fine-tuned using:
- **Dataset**: Medical Meadow Medical Flashcards (~33k Q&A pairs)
- **Training Time**: ~2-3 hours on GPU
- **Hardware**: NVIDIA GPU with 16GB+ VRAM
- **Technique**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Medical Meadow Dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- [PEFT Library](https://github.com/huggingface/peft)
- [Gradio Documentation](https://gradio.app/docs/)

## Author

Created as an educational project to demonstrate medical domain fine-tuning of large language models.

---

**Note**: This model is for educational and research purposes only. It should not replace professional medical advice.
