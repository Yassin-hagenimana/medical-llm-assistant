"""
Gradio deployment interface for the fine-tuned medical LLM assistant.
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalAssistant:
    """
    Medical LLM Assistant wrapper for inference.
    """
    
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_adapter_path: str = "models/final/medical-llm-lora",
        device: str = "auto"
    ):
        """
        Initialize the medical assistant.
        
        Args:
            base_model_name: Name of the base model
            lora_adapter_path: Path to LoRA adapter weights
            device: Device to run model on
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch.float16,
            load_in_8bit=True
        )
        
        # Load LoRA adapter
        try:
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            logger.info("LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter: {e}")
            logger.warning("Using base model only")
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate_response(
        self,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Generate response to a query.
        
        Args:
            query: User query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response
        """
        # Format input
        prompt = f"### Instruction:\n{query}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response
        
        return response


def create_interface(assistant: MedicalAssistant) -> gr.Blocks:
    """
    Create Gradio interface for the medical assistant.
    
    Args:
        assistant: MedicalAssistant instance
        
    Returns:
        Gradio Blocks interface
    """
    
    def chat(
        query: str,
        history: List[Tuple[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Chat function for Gradio interface.
        """
        if not query.strip():
            return history, ""
        
        response = assistant.generate_response(
            query,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        history.append((query, response))
        return history, ""
    
    # Create interface
    with gr.Blocks(title="Medical LLM Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Medical LLM Assistant
            
            A fine-tuned AI assistant for medical question-answering.
            
            **Disclaimer:** This is an educational project. Always consult healthcare 
            professionals for medical advice.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a medical question...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Conversation")
                
                gr.Examples(
                    examples=[
                        "What are the symptoms of diabetes?",
                        "How is hypertension treated?",
                        "What are the side effects of aspirin?",
                        "Explain what an MRI scan is",
                        "What is the difference between type 1 and type 2 diabetes?"
                    ],
                    inputs=query_input,
                    label="Example Questions"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Generation Parameters")
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=512,
                    value=256,
                    step=50,
                    label="Max Tokens",
                    info="Maximum length of response"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling parameter"
                )
                
                gr.Markdown(
                    """
                    ### About
                    
                    This assistant is fine-tuned using:
                    - LoRA (Parameter-Efficient Fine-Tuning)
                    - Medical domain dataset
                    - TinyLlama base model
                    
                    **Limitations:**
                    - Educational purpose only
                    - Limited to training data knowledge
                    - Cannot provide personalized medical advice
                    """
                )
        
        # Event handlers
        submit_btn.click(
            fn=chat,
            inputs=[query_input, chatbot, max_tokens, temperature, top_p],
            outputs=[chatbot, query_input]
        )
        
        query_input.submit(
            fn=chat,
            inputs=[query_input, chatbot, max_tokens, temperature, top_p],
            outputs=[chatbot, query_input]
        )
        
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, query_input]
        )
    
    return demo


def main():
    """
    Main function to launch the application.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical LLM Assistant")
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name"
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default="models/final/medical-llm-lora",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = MedicalAssistant(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_adapter
    )
    
    # Create and launch interface
    demo = create_interface(assistant)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
