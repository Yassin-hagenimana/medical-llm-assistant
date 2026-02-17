"""
Hugging Face Spaces Deployment Script
Deploy Medical LLM Assistant to Hugging Face Spaces
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    import gradio as gr
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Installing huggingface_hub and gradio...")
    os.system("pip install -U huggingface_hub[cli] gradio")
    print("\nSUCCESS: Packages installed. Please run this script again.")
    sys.exit(0)

def check_login():
    """Check if user is logged into Hugging Face"""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"SUCCESS: Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print("ERROR: Not logged in to Hugging Face.")
        print("\nPlease login using ONE of these methods:")
        print("\n1. Using Python:")
        print("   >>> from huggingface_hub import login")
        print("   >>> login()")
        print("\n2. Using command line:")
        print("   $ python -m huggingface_hub.commands.huggingface_cli login")
        print("\n3. Or set environment variable:")
        print("   $ export HUGGING_FACE_HUB_TOKEN='your_token_here'")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        return False

def prepare_deployment():
    """Prepare files for deployment"""
    print("\n" + "="*60)
    print("Medical LLM Assistant - Hugging Face Deployment")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    # Check required files
    required_files = {
        'app.py': project_root / 'app.py',
        'requirements.txt': project_root / 'requirements-hf.txt',
        'README.md': project_root / 'README-HF.md'
    }
    
    print("\nChecking required files...")
    for name, path in required_files.items():
        if path.exists():
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name} not found at {path}")
            return False
    
    # Check for model files
    model_paths = [
        project_root / 'notebooks' / 'medical_llm_final',
        project_root / 'medical_llm_final'
    ]
    
    model_dir = None
    for path in model_paths:
        if path.exists() and path.is_dir():
            model_dir = path
            print(f"\nSUCCESS: Found model adapters at: {model_dir}")
            break
    
    if not model_dir:
        print("\nWARNING: Model adapter files not found.")
        print("   The app will use the base TinyLlama model without fine-tuning.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True

def deploy_to_hf_spaces(space_name=None, private=False):
    """
    Deploy to Hugging Face Spaces
    
    Args:
        space_name: Name of the space (e.g., 'username/medical-llm-assistant')
        private: Whether to make the space private
    """
    if not check_login():
        return
    
    if not prepare_deployment():
        print("\nERROR: Deployment preparation failed.")
        return
    
    if not space_name:
        print("\nEnter your Space name (format: username/space-name)")
        print("   Example: john/medical-llm-assistant")
        space_name = input("   Space name: ").strip()
    
    if not space_name or '/' not in space_name:
        print("ERROR: Invalid space name. Format should be: username/space-name")
        return
    
    try:
        api = HfApi()
        project_root = Path(__file__).parent
        
        print(f"\nCreating/updating Space: {space_name}")
        
        # Create the space repository
        try:
            repo_url = create_repo(
                repo_id=space_name,
                repo_type="space",
                space_sdk="gradio",
                private=private,
                exist_ok=True
            )
            print(f"SUCCESS: Space created/updated: {repo_url}")
        except Exception as e:
            print(f"WARNING: {e}")
        
        # Create deployment directory
        deploy_dir = project_root / 'deploy_temp'
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy files
        print("\nPreparing files for upload...")
        import shutil
        
        shutil.copy(project_root / 'app.py', deploy_dir / 'app.py')
        shutil.copy(project_root / 'requirements-hf.txt', deploy_dir / 'requirements.txt')
        shutil.copy(project_root / 'README-HF.md', deploy_dir / 'README.md')
        
        # Copy model files if they exist
        model_paths = [
            project_root / 'notebooks' / 'medical_llm_final',
            project_root / 'medical_llm_final'
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                dest = deploy_dir / 'medical_llm_final'
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(model_path, dest)
                print(f"  [OK] Copied model adapters")
                break
        
        # Upload to HF Spaces
        print(f"\nUploading to {space_name}...")
        upload_folder(
            folder_path=str(deploy_dir),
            repo_id=space_name,
            repo_type="space",
            commit_message="Deploy Medical LLM Assistant"
        )
        
        print("\n" + "="*60)
        print("SUCCESS: Deployment successful!")
        print("="*60)
        print(f"\nYour Space: https://huggingface.co/spaces/{space_name}")
        print("\nThe Space may take a few minutes to build and start.")
        print("   You can monitor the build process on the Space page.")
        
        # Cleanup
        shutil.rmtree(deploy_dir)
        
    except Exception as e:
        print(f"\nERROR: Deployment failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main deployment function"""
    print("\nHugging Face Spaces Deployment Options:")
    print("\n1. Deploy with manual Space name")
    print("2. Test deployment preparation only")
    print("3. Check login status")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        deploy_to_hf_spaces()
    elif choice == '2':
        prepare_deployment()
        print("\nSUCCESS: Deployment preparation check complete.")
    elif choice == '3':
        check_login()
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
