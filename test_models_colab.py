# Test Models Response Format - Google Colab Version
# Copy and paste this code into a Google Colab cell

# Install required packages
!pip install transformers>=4.36.0 torch>=2.0.0 bitsandbytes>=0.41.0 accelerate>=0.24.0

# Core imports
import os, json, torch, time
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("✅ All dependencies loaded!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set up project paths for caching
project_path = "/content/drive/MyDrive/Colab Notebooks/KEEPCODING/PROJECT/AI_Recruiter_Assistant"
cache_path = f"{project_path}/huggingface_cache"

# Create directories
os.makedirs(cache_path, exist_ok=True)

print(f"✅ Google Drive mounted")
print(f"📁 Project path: {project_path}")
print(f"🗂️ Cache path: {cache_path}")

# Model loading functions
def check_cached_models(cache_path):
    """Check for cached models"""
    cached_models = []
    if not os.path.exists(cache_path):
        return cached_models

    try:
        items = os.listdir(cache_path)
        for item in items:
            item_path = os.path.join(cache_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                try:
                    contents = os.listdir(item_path)
                    has_models_folder = any(f.startswith('models--') for f in contents if os.path.isdir(os.path.join(item_path, f)))
                    if has_models_folder:
                        cached_models.append(item)
                except Exception:
                    continue
    except Exception:
        pass

    return cached_models

def create_model_cache_dir(model_name: str, cache_path: str) -> str:
    """Create clean cache directory for a model"""
    model_folder = model_name.replace('/', ' ')
    model_cache_dir = os.path.join(cache_path, model_folder)
    os.makedirs(model_cache_dir, exist_ok=True)
    return model_cache_dir

def load_model_with_quantization(model_name: str, cache_path: str):
    """Load model with cache detection and quantization"""
    cached_models = check_cached_models(cache_path)
    model_folder_space = model_name.replace('/', ' ')

    if model_folder_space in cached_models:
        print(f"⚡ Loading {model_name} from cache...")
        model_cache_dir = os.path.join(cache_path, model_folder_space)
    else:
        print(f"📥 Downloading {model_name}...")
        model_cache_dir = create_model_cache_dir(model_name, cache_path)

    # Quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=model_cache_dir
        )

        print(f"✅ {model_name} loaded successfully!")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        raise

print("✅ Model loading functions ready!")

# Test response generation function
def test_model_response(model, tokenizer, model_name: str, prompt: str, max_new_tokens: int = 150):
    """Test model response and show detailed structure"""
    print(f"\n🧪 TESTING MODEL: {model_name}")
    print("=" * 60)
    print(f"📝 INPUT PROMPT:")
    print(f'"{prompt}"')
    print("-" * 60)
    
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        print(f"📊 TOKENIZATION INFO:")
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Input tokens count: {inputs['input_ids'].shape[1]}")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"   Device: {device}")
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        generation_time = time.time() - start_time
        
        print(f"\n⚡ GENERATION INFO:")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Output type: {type(outputs)}")
        print(f"   Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'No keys (tensor)'}")
        
        # Extract sequences
        if hasattr(outputs, 'sequences'):
            sequences = outputs.sequences
            print(f"   Sequences shape: {sequences.shape}")
        else:
            sequences = outputs
            print(f"   Sequences shape: {sequences.shape}")
        
        # Decode full response
        full_response = tokenizer.decode(sequences[0], skip_special_tokens=True)
        
        # Extract only new tokens (response without prompt)
        generated_response = full_response.replace(prompt, "").strip()
        
        print(f"\n💬 RESPONSE ANALYSIS:")
        print(f"   Full response length: {len(full_response)} chars")
        print(f"   Generated response length: {len(generated_response)} chars")
        print(f"   New tokens generated: {sequences.shape[1] - inputs['input_ids'].shape[1]}")
        
        print(f"\n📄 FULL RESPONSE:")
        print("<<START_FULL>>")
        print(full_response)
        print("<<END_FULL>>")
        
        print(f"\n✨ GENERATED ONLY:")
        print("<<START_GENERATED>>")
        print(generated_response)
        print("<<END_GENERATED>>")
        
        # Show token structure for debugging
        print(f"\n🔍 TOKEN ANALYSIS:")
        input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        output_tokens = tokenizer.convert_ids_to_tokens(sequences[0])
        new_tokens = output_tokens[len(input_tokens):]
        
        print(f"   Input tokens: {input_tokens[:5]}...{input_tokens[-5:]}")
        print(f"   New tokens: {new_tokens[:10]}")
        
        # Return structured result
        return {
            "model_name": model_name,
            "prompt": prompt,
            "full_response": full_response,
            "generated_response": generated_response,
            "generation_time": generation_time,
            "input_tokens_count": inputs['input_ids'].shape[1],
            "output_tokens_count": sequences.shape[1],
            "new_tokens_count": sequences.shape[1] - inputs['input_ids'].shape[1],
            "device": str(device)
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return {"error": str(e)}

print("✅ Response testing function ready!")

# Define test prompts
test_prompts = [
    {
        "name": "Simple Validation",
        "prompt": """You are an expert at validating professional email responses.

Check if this response is written in first person:
"The candidate's technical skills match the job requirements very well."

Respond with ONLY:
VALIDATION: [PASS or FAIL]
ISSUES: [List problems or "None"]"""
    },
    {
        "name": "Simple Correction", 
        "prompt": """Fix this response to use first person:

"The candidate's experience in Python and data engineering is excellent."

Write the corrected version:"""
    },
    {
        "name": "Basic Question",
        "prompt": "What is 2+2? Answer briefly."
    }
]

print("✅ Test prompts defined!")
print(f"📊 Total prompts: {len(test_prompts)}")
for i, prompt in enumerate(test_prompts, 1):
    print(f"   {i}. {prompt['name']}")

# Test Llama 3 Model
llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"🚀 TESTING LLAMA 3 MODEL")
print("=" * 80)

try:
    # Load Llama 3 model
    llama_model, llama_tokenizer = load_model_with_quantization(llama_model_name, cache_path)
    
    print(f"\n📋 MODEL INFO:")
    print(f"   Model: {llama_model_name}")
    print(f"   Tokenizer vocab size: {llama_tokenizer.vocab_size}")
    print(f"   EOS token: {llama_tokenizer.eos_token}")
    print(f"   PAD token: {llama_tokenizer.pad_token}")
    
    # Test all prompts
    llama_results = []
    
    for prompt_info in test_prompts:
        result = test_model_response(
            llama_model, 
            llama_tokenizer, 
            "Llama-3-8B", 
            prompt_info["prompt"],
            max_new_tokens=100
        )
        result["prompt_name"] = prompt_info["name"]
        llama_results.append(result)
        
        print("\n" + "="*80)
    
    print(f"\n✅ LLAMA 3 TESTING COMPLETE!")
    print(f"📊 Tested {len(llama_results)} prompts")
    
except Exception as e:
    print(f"❌ Failed to test Llama 3: {str(e)}")
    llama_results = []

# Test Gemma Model
gemma_model_name = "google/gemma-3-4b-it"

print(f"🚀 TESTING GEMMA MODEL")
print("=" * 80)

try:
    # Load Gemma model
    gemma_model, gemma_tokenizer = load_model_with_quantization(gemma_model_name, cache_path)
    
    print(f"\n📋 MODEL INFO:")
    print(f"   Model: {gemma_model_name}")
    print(f"   Tokenizer vocab size: {gemma_tokenizer.vocab_size}")
    print(f"   EOS token: {gemma_tokenizer.eos_token}")
    print(f"   PAD token: {gemma_tokenizer.pad_token}")
    
    # Test all prompts
    gemma_results = []
    
    for prompt_info in test_prompts:
        result = test_model_response(
            gemma_model, 
            gemma_tokenizer, 
            "Gemma-3-4B", 
            prompt_info["prompt"],
            max_new_tokens=100
        )
        result["prompt_name"] = prompt_info["name"]
        gemma_results.append(result)
        
        print("\n" + "="*80)
    
    print(f"\n✅ GEMMA TESTING COMPLETE!")
    print(f"📊 Tested {len(gemma_results)} prompts")
    
except Exception as e:
    print(f"❌ Failed to test Gemma: {str(e)}")
    gemma_results = []

# Compare responses from both models
print("📊 MODEL COMPARISON SUMMARY")
print("=" * 80)

if llama_results and gemma_results:
    for i, prompt_info in enumerate(test_prompts):
        print(f"\n🔍 PROMPT: {prompt_info['name']}")
        print("-" * 50)
        
        if i < len(llama_results) and i < len(gemma_results):
            llama_result = llama_results[i]
            gemma_result = gemma_results[i]
            
            print(f"🦙 LLAMA 3 ({llama_result.get('generation_time', 0):.2f}s):")
            print(f"   Response: {llama_result.get('generated_response', 'ERROR')[:200]}...")
            
            print(f"\n💎 GEMMA ({gemma_result.get('generation_time', 0):.2f}s):")
            print(f"   Response: {gemma_result.get('generated_response', 'ERROR')[:200]}...")
            
            # Performance comparison
            if 'generation_time' in llama_result and 'generation_time' in gemma_result:
                faster = "Llama 3" if llama_result['generation_time'] < gemma_result['generation_time'] else "Gemma"
                print(f"\n⚡ Faster: {faster}")
        
        print("\n" + "="*50)
    
    # Overall performance summary
    print(f"\n🏆 OVERALL PERFORMANCE:")
    
    if llama_results:
        avg_llama_time = sum(r.get('generation_time', 0) for r in llama_results) / len(llama_results)
        print(f"   🦙 Llama 3 avg time: {avg_llama_time:.2f}s")
    
    if gemma_results:
        avg_gemma_time = sum(r.get('generation_time', 0) for r in gemma_results) / len(gemma_results)
        print(f"   💎 Gemma avg time: {avg_gemma_time:.2f}s")

else:
    print("❌ Could not compare - one or both models failed to load/test")

print(f"\n✅ TESTING COMPLETE!")

# Save detailed results to file for analysis
results_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "test_info": {
        "purpose": "Model response format testing",
        "models_tested": [llama_model_name, gemma_model_name],
        "prompts_count": len(test_prompts)
    },
    "llama_results": llama_results,
    "gemma_results": gemma_results,
    "test_prompts": test_prompts
}

# Save to file
results_file = f"{project_path}/model_response_test_results.json"
try:
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to: {results_file}")
except Exception as e:
    print(f"❌ Could not save results: {str(e)}")

print(f"\n🎯 ANALYSIS COMPLETE!")
print(f"📊 Models tested: {len([r for r in [llama_results, gemma_results] if r])}")
print(f"📝 Total responses generated: {len(llama_results) + len(gemma_results)}")
print(f"\n💡 Use this data to improve the output guardrail implementation!") 