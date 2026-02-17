#!/bin/bash
# Deployment script for Hugging Face Spaces

echo "==================================="
echo "Medical LLM Assistant Deployment"
echo "==================================="
echo ""

# Check if Hugging Face CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  Hugging Face CLI not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# Check if logged in
echo "Checking Hugging Face authentication..."
huggingface-cli whoami

if [ $? -ne 0 ]; then
    echo ""
    echo " Not logged in to Hugging Face."
    echo "Please run: huggingface-cli login"
    exit 1
fi

echo ""
echo "Creating deployment directory..."

# Create a temporary deployment directory
DEPLOY_DIR="deploy_medical_llm"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy necessary files
echo "Copying files..."
cp app.py $DEPLOY_DIR/
cp requirements-hf.txt $DEPLOY_DIR/requirements.txt
cp README-HF.md $DEPLOY_DIR/README.md

# Copy model files if they exist
if [ -d "notebooks/medical_llm_final" ]; then
    echo "Copying model adapter files..."
    cp -r notebooks/medical_llm_final $DEPLOY_DIR/
elif [ -d "medical_llm_final" ]; then
    echo "Copying model adapter files..."
    cp -r medical_llm_final $DEPLOY_DIR/
else
    echo "⚠️  Model adapters not found. The app will use the base model."
    echo "   To use the fine-tuned model, copy 'medical_llm_final' folder to the Space."
fi

echo ""
echo "✅ Deployment files prepared in $DEPLOY_DIR/"
echo ""
echo "Next steps:"
echo "1. Create a new Space on Hugging Face:"
echo "   https://huggingface.co/new-space"
echo ""
echo "2. Choose these settings:"
echo "   - SDK: Gradio"
echo "   - Space hardware: CPU Basic (or GPU for faster inference)"
echo ""
echo "3. Clone your new Space:"
echo "   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME"
echo ""
echo "4. Copy files to the Space:"
echo "   cp -r $DEPLOY_DIR/* YOUR_SPACE_NAME/"
echo ""
echo "5. Push to Hugging Face:"
echo "   cd YOUR_SPACE_NAME"
echo "   git add ."
echo "   git commit -m 'Initial deployment'"
echo "   git push"
echo ""
echo "Or use Gradio CLI to deploy:"
echo "   cd $DEPLOY_DIR"
echo "   gradio deploy"
echo ""
