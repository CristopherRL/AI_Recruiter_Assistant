"""
Prompt Loader Utility for AI Recruiter Assistant
Loads prompts from organized file structure in prompts/app/
"""

import os
from typing import Dict, Any
from pathlib import Path

class PromptLoader:
    """Utility class to load prompts from the organized file structure"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            # Default to current directory structure
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)
    
    def load_text_file(self, file_path: str) -> str:
        """Load content from a text file"""
        full_path = self.base_path / file_path
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt file {full_path}: {str(e)}")
    
    def load_input_guardrail_prompts(self) -> Dict[str, str]:
        """Load all input guardrail prompts"""
        return {
            "classification_prompt": self.load_text_file("01_input_guardrail/classification_prompt.txt"),
            "generic_response_template": self.load_text_file("01_input_guardrail/generic_response_template.txt"),
            "brief_response_template": self.load_text_file("01_input_guardrail/brief_response_template.txt"),
            "language_templates": self.load_text_file("01_input_guardrail/language_templates.txt"),
            "basic_intro_prompt": self.load_text_file("01_input_guardrail/basic_intro_prompt.txt"),
            "opportunity_inquiry_prompt": self.load_text_file("01_input_guardrail/opportunity_inquiry_prompt.txt")
        }
    
    def load_main_generator_prompts(self) -> Dict[str, Any]:
        """Load all main generator prompts"""
        response_templates = {
            "passed": self.load_text_file("02_main_generator/response_templates/passed_template.txt"),
            "stand_by": self.load_text_file("02_main_generator/response_templates/stand_by_template.txt"),
            "finished": self.load_text_file("02_main_generator/response_templates/finished_template.txt")
        }
        
        return {
            "match_scoring_prompt": self.load_text_file("02_main_generator/match_scoring_prompt.txt"),
            "language_instructions": self.load_text_file("02_main_generator/language_instructions.txt"),
            "natural_response_prompt": self.load_text_file("02_main_generator/natural_response_prompt.txt"),
            "response_templates": response_templates
        }
    
    def load_output_guardrail_prompts(self) -> Dict[str, str]:
        """Load all output guardrail prompts"""
        return {
            "validation_prompt": self.load_text_file("03_output_guardrail/validation_prompt.txt"),
            "correction_prompt": self.load_text_file("03_output_guardrail/correction_prompt.txt")
        }
    
    def load_all_prompts(self) -> Dict[str, Any]:
        """Load all prompts from the organized structure"""
        return {
            "input_guardrail": self.load_input_guardrail_prompts(),
            "main_generator": self.load_main_generator_prompts(),
            "output_guardrail": self.load_output_guardrail_prompts()
        }
    
    def validate_structure(self) -> bool:
        """Validate that all expected prompt files exist"""
        expected_files = [
            "01_input_guardrail/classification_prompt.txt",
            "01_input_guardrail/generic_response_template.txt",
            "01_input_guardrail/brief_response_template.txt",
            "01_input_guardrail/language_templates.txt",
            "01_input_guardrail/basic_intro_prompt.txt",
            "01_input_guardrail/opportunity_inquiry_prompt.txt",
            "02_main_generator/match_scoring_prompt.txt",
            "02_main_generator/language_instructions.txt",
            "02_main_generator/natural_response_prompt.txt",
            "02_main_generator/response_templates/passed_template.txt",
            "02_main_generator/response_templates/stand_by_template.txt",
            "02_main_generator/response_templates/finished_template.txt",
            "03_output_guardrail/validation_prompt.txt",
            "03_output_guardrail/correction_prompt.txt"
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = self.base_path / file_path
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            print(f"❌ Missing prompt files:")
            for file in missing_files:
                print(f"   • {file}")
            return False
        
        print("✅ All prompt files found!")
        return True

# Convenience function for easy usage
def load_prompts(base_path: str = None) -> Dict[str, Any]:
    """Load all prompts from the organized structure"""
    loader = PromptLoader(base_path)
    loader.validate_structure()
    return loader.load_all_prompts()

if __name__ == "__main__":
    # Test the loader
    print("🔍 Testing Prompt Loader...")
    try:
        prompts = load_prompts()
        print("✅ All prompts loaded successfully!")
        
        # Show structure
        print("\n📋 Loaded prompt structure:")
        for category, content in prompts.items():
            print(f"   📂 {category}:")
            if isinstance(content, dict):
                for key in content.keys():
                    if key == "response_templates":
                        templates = content[key]
                        print(f"      📄 {key}:")
                        for template_name in templates.keys():
                            print(f"         📄 {template_name}")
                    else:
                        print(f"      📄 {key}")
            else:
                print(f"      📄 {content}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}") 