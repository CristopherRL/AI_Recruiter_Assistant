# AI Recruiter Assistant - Prompt Organization

## 📋 Overview

This directory contains all prompts used by the AI Recruiter Assistant, organized following the complete process flow from recruiter message input to final response output.

## 🏗️ Directory Structure

```
prompts/app/
├── 01_input_guardrail/
│   ├── classification_prompt.txt          # Classifies messages as generic/concrete
│   └── generic_response_template.txt      # Template for generic message responses
├── 02_main_generator/
│   ├── match_scoring_prompt.txt           # Calculates job match score
│   └── response_templates/
│       ├── passed_template.txt            # Template for high match jobs (>80%)
│       ├── stand_by_template.txt          # Template for medium match jobs (60-80%)
│       └── finished_template.txt          # Template for low match jobs (<60%)
├── 03_output_guardrail/
│   ├── validation_prompt.txt              # Validates response naturalness
│   └── correction_prompt.txt              # Corrects non-natural responses
├── prompt_loader.py                       # Utility to load all prompts
└── README.md                              # This documentation
```

## 🔄 Process Flow

The prompts are organized to follow the complete AI assistant process:

1. **📨 Recruiter Message Input** 
2. **🛡️ Input Guardrail** (`01_input_guardrail/`)
   - Classify message type (generic vs concrete offer)
   - Generate response for generic messages
3. **🧠 Main Generator** (`02_main_generator/`)
   - Calculate match score for concrete offers
   - Generate appropriate response based on score
4. **🛡️ Output Guardrail** (`03_output_guardrail/`)
   - Validate response naturalness
   - Correct issues if found
5. **💬 Final Response Output**

## 🔧 Usage

The prompts are loaded automatically by the AI assistant classes using the `PromptLoader` utility:

```python
# In InputGuardrail
from prompt_loader import PromptLoader
prompt_loader = PromptLoader(f"{project_path}/prompts/app")
input_prompts = prompt_loader.load_input_guardrail_prompts()

# In OutputGuardrail  
output_prompts = prompt_loader.load_output_guardrail_prompts()

# In AIRecruiterAssistant
main_generator_prompts = prompt_loader.load_main_generator_prompts()
```

## 📝 Editing Prompts

To modify prompts:

1. Edit the relevant `.txt` file directly
2. Save the file
3. Restart the AI assistant to load the updated prompts
4. No code changes needed!

## 🧪 Testing

Test prompts are kept in the main notebook code for development purposes:
- `benchmark.system_prompt` - Used for model benchmarking
- `benchmark.test_prompts` - Used for testing scenarios

## 💡 Benefits

- **🎯 Clear Organization**: Prompts follow the logical process flow
- **📝 Easy Editing**: Modify prompts without touching code
- **🔄 Version Control**: Track prompt changes separately from code
- **🚀 Faster Iteration**: Quick prompt engineering improvements
- **🧹 Clean Code**: Reduced clutter in main notebook 