import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Model configuration
MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
ADAPTER_PATH = './notebooks/medical_llm_final'  # LoRA adapter path

# Check if we're running on Hugging Face Spaces
IS_SPACES = os.getenv("SPACE_ID") is not None

print("Initializing Medical LLM Assistant...")
print("=" * 80)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading model...")
try:
    # Try loading the fine-tuned model with LoRA adapters
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading fine-tuned model from {ADAPTER_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map='auto',
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
        print("Fine-tuned model loaded successfully!")
    else:
        print(f"WARNING: Adapter path {ADAPTER_PATH} not found.")
        print("Loading base model instead...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map='auto',
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Base model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='auto',
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

model.eval()
print("=" * 80)
print("Model initialization complete!")


def generate_medical_response(question, temperature=0.7, max_length=150):
    """
    Generate medical response using the fine-tuned model.
    
    Args:
        question: Medical question from user
        temperature: Controls randomness (0.1-1.0)
        max_length: Maximum response length
    
    Returns:
        Generated medical response
    """
    if not question.strip():
        return "Please enter a medical question."
    
    # Format the prompt
    prompt = f"""<|system|>
You are a helpful medical assistant. Provide accurate, evidence-based medical information.
<|user|>
{question}
<|assistant|>
"""
    
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate response with optimized settings for speed
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1  # Use greedy decoding (faster than beam search)
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Create minimalist black and white theme
custom_theme = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="gray",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#000000",
    button_primary_background_fill_hover="#333333",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f5f5f5",
    button_secondary_border_color="#000000",
    button_secondary_text_color="#000000",
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
)

# Create Gradio interface
with gr.Blocks(
    title="Medical LLM Assistant"
) as demo:
    custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif;
            background: #ffffff !important;
        }
        .contain {
            max-width: 1200px;
            margin: auto;
            background: #ffffff !important;
        }
        * {
            color: #000000 !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div, li {
            color: #000000 !important;
        }
        .prose, .prose * {
            color: #000000 !important;
        }
        .markdown-body, .markdown-body * {
            color: #000000 !important;
        }
        textarea, input {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #e5e5e5 !important;
        }
        .svelte-*, [class*="svelte-"] {
            color: #000000 !important;
        }
        .input-container, .output-container {
            background: transparent !important;
        }
        .block, .form, .wrap, .label-wrap {
            background: transparent !important;
        }
        .wrap *, .label-wrap *, .info {
            color: #000000 !important;
        }
        /* Keep primary button text white */
        .btn-primary, button.primary {
            color: #ffffff !important;
        }
    """
    gr.Markdown("""
    # Medical LLM Assistant
    
    This is a fine-tuned TinyLlama model specialized for medical question answering.
    
    **DISCLAIMER:** This is an educational project. Always consult qualified healthcare professionals for medical advice.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Medical Question",
                placeholder="Enter your medical question here...",
                lines=3
            )
            
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (Creativity)",
                    info="Lower = more conservative, Higher = more creative"
                )
                
                max_length_slider = gr.Slider(
                    minimum=50,
                    maximum=300,
                    value=100,  # Reduced default from 150 for faster responses
                    step=50,
                    label="Max Response Length",
                    info="Maximum tokens to generate"
                )
            
            with gr.Row():
                submit_btn = gr.Button(
                    "Ask Question",
                    variant="primary",
                    size="lg"
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    size="lg"
                )
        
        with gr.Column(scale=2):
            response_output = gr.Textbox(
                label="Medical Assistant Response",
                lines=10
            )
    
    # Example questions
    gr.Examples(
        examples=[
            ["What are the symptoms of diabetes?"],
            ["How does the immune system work?"],
            ["What is the difference between bacteria and viruses?"],
            ["What are the risk factors for heart disease?"],
            ["How do vaccines provide immunity?"],
            ["What is the function of hemoglobin?"],
            ["What causes high blood pressure?"],
            ["Explain the role of insulin in the body."]
        ],
        inputs=question_input,
        label="Example Medical Questions"
    )
    
    # Connect the buttons to functions
    submit_btn.click(
        fn=generate_medical_response,
        inputs=[question_input, temperature_slider, max_length_slider],
        outputs=response_output
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[question_input, response_output]
    )
    
    gr.Markdown("""
    ### Model Information
    - **Base Model:** TinyLlama-1.1B-Chat-v1.0
    - **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
    - **Dataset:** Medical Meadow Medical Flashcards
    - **Training Domain:** General medical knowledge and terminology

    """)

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=not IS_SPACES,  # Only create share link when running locally, not on Spaces
        theme=custom_theme,
        css=custom_css
    )
