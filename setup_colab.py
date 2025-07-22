#!/usr/bin/env python3
"""
Setup script for AI Recruiter Assistant in Google Colab
This script helps prepare the environment and verify all components are ready.
"""

import os
import sys
import pandas as pd
from pathlib import Path

def check_environment():
    """Check if all required packages are installed"""
    print("🔍 Checking environment...")
    
    required_packages = [
        'transformers', 'torch', 'peft', 'bitsandbytes', 'accelerate',
        'langchain', 'langchain_community', 'faiss', 'sentence_transformers',
        'gradio', 'pandas', 'numpy', 'sklearn', 'tqdm', 'datasets'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!")
        return True

def check_data_files():
    """Check if all required data files exist"""
    print("\n📁 Checking data files...")
    
    required_files = [
        'RAG/cv.md',
        'RAG/job_expectations.md',
        'data/linkedin_messages.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All data files found!")
        return True

def analyze_linkedin_data():
    """Analyze LinkedIn messages data for fine-tuning"""
    print("\n📊 Analyzing LinkedIn messages...")
    
    try:
        df = pd.read_csv('data/linkedin_messages.csv')
        
        print(f"Total messages: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for conversation patterns
        if 'CONTENT' in df.columns:
            content_lengths = df['CONTENT'].str.len()
            print(f"Average message length: {content_lengths.mean():.1f} characters")
            print(f"Min message length: {content_lengths.min()} characters")
            print(f"Max message length: {content_lengths.max()} characters")
        
        # Check for different conversation types
        if 'FOLDER' in df.columns:
            folder_counts = df['FOLDER'].value_counts()
            print(f"\nMessage folders:")
            for folder, count in folder_counts.items():
                print(f"  {folder}: {count:,} messages")
        
        return True
    except Exception as e:
        print(f"❌ Error analyzing LinkedIn data: {e}")
        return False

def create_colab_instructions():
    """Create instructions for Colab setup"""
    print("\n📋 Colab Setup Instructions:")
    print("=" * 50)
    print("""
 1. Upload these files to your Colab environment:
    - RAG/cv.md
    - RAG/job_expectations.md
    - data/linkedin_messages.csv
    - ai_recruiter_assistant.ipynb

2. Install dependencies:
   !pip install transformers>=4.36.0 torch>=2.0.0 peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.24.0
   !pip install langchain>=0.1.0 langchain-community>=0.0.10 faiss-cpu>=1.7.4 sentence-transformers>=2.2.0
   !pip install gradio>=4.0.0 pandas>=2.0.0 numpy>=1.24.0 tqdm>=4.65.0 datasets>=2.14.0

3. Run the notebook cells in order:
   - Cell 1: Imports and setup
   - Cell 2: Data loading
   - Cell 3: RAG pipeline setup
   - Cell 4: State management
   - Cell 5: Intent detection
   - Cell 6: LLM setup (uncomment when ready)
   - Cell 7: LinkedIn data processing
   - Cell 8: Response generation
   - Cell 9: Main processing function

4. For fine-tuning (Days 3-4):
   - Process LinkedIn messages
   - Load base LLM (Mistral-7B-Instruct)
   - Implement LoRA fine-tuning
   - Train and validate model

5. For web interface (Day 7):
   - Create Gradio interface
   - Test end-to-end functionality
   - Deploy for production use
    """)

def main():
    """Main setup function"""
    print("🚀 AI Recruiter Assistant - Environment Setup")
    print("=" * 50)
    
    # Check environment
    env_ok = check_environment()
    
    # Check data files
    data_ok = check_data_files()
    
    # Analyze LinkedIn data
    if data_ok:
        linkedin_ok = analyze_linkedin_data()
    else:
        linkedin_ok = False
    
    # Create Colab instructions
    create_colab_instructions()
    
    # Summary
    print("\n📊 Setup Summary:")
    print("=" * 30)
    print(f"Environment: {'✅ Ready' if env_ok else '❌ Issues'}")
    print(f"Data files: {'✅ Ready' if data_ok else '❌ Issues'}")
    print(f"LinkedIn data: {'✅ Ready' if linkedin_ok else '❌ Issues'}")
    
    if env_ok and data_ok and linkedin_ok:
        print("\n🎉 Everything is ready! You can now:")
        print("1. Open ai_recruiter_assistant.ipynb in Google Colab")
        print("2. Upload your files to Colab")
        print("3. Start developing your AI Recruiter Assistant")
    else:
        print("\n⚠️ Please fix the issues above before proceeding")
    
    return env_ok and data_ok and linkedin_ok

if __name__ == "__main__":
    main() 