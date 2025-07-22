#!/usr/bin/env python3
"""
Test script for AI Recruiter Assistant basic functionality
"""

import sys
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_document_loading():
    """Test loading of CV and job expectations"""
    print("🧪 Testing document loading...")
    
    try:
        # Test CV loading
        with open('RAG/cv.md', 'r', encoding='utf-8') as f:
            cv_content = f.read()
        print(f"✅ CV loaded: {len(cv_content)} characters")
        
        # Test job expectations loading
        with open('RAG/job_expectations.md', 'r', encoding='utf-8') as f:
            expectations_content = f.read()
        print(f"✅ Job expectations loaded: {len(expectations_content)} characters")
        
        return True
    except FileNotFoundError as e:
        print(f"❌ Error loading documents: {e}")
        return False

def test_linkedin_data():
    """Test LinkedIn messages data loading"""
    print("\n🧪 Testing LinkedIn data loading...")
    
    try:
        df = pd.read_csv('data/linkedin_messages.csv')
        print(f"✅ LinkedIn messages loaded: {len(df)} messages")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Sample data:")
        print(df.head(3))
        return True
    except Exception as e:
        print(f"❌ Error loading LinkedIn data: {e}")
        return False

def test_intent_detection():
    """Test intent detection functionality"""
    print("\n🧪 Testing intent detection...")
    
    # Simple keyword-based detection (from notebook)
    def detect_intent(message: str) -> str:
        job_keywords = [
            'position', 'role', 'job', 'opportunity', 'vacancy', 'opening',
            'salary', 'remote', 'hybrid', 'onsite', 'tech stack', 'requirements',
            'experience', 'years', 'skills', 'technologies', 'framework'
        ]
        
        generic_keywords = [
            'hi', 'hello', 'interested', 'opportunities', 'open', 'available',
            'connect', 'network', 'profile', 'background'
        ]
        
        message_lower = message.lower()
        
        job_score = sum(1 for keyword in job_keywords if keyword in message_lower)
        generic_score = sum(1 for keyword in generic_keywords if keyword in message_lower)
        
        if job_score > generic_score and job_score >= 2:
            return "job_offer"
        else:
            return "generic_message"
    
    # Test cases
    test_messages = [
        "Hi, are you open to new opportunities?",
        "We have a Data Engineer position with Python and cloud technologies",
        "Looking for a Senior AI Engineer with 5+ years experience",
        "Hello, I'd like to connect with you"
    ]
    
    for message in test_messages:
        intent = detect_intent(message)
        print(f"Message: '{message[:50]}...' -> Intent: {intent}")
    
    return True

def test_state_management():
    """Test conversation state management"""
    print("\n🧪 Testing state management...")
    
    class ConversationState(Enum):
        PENDING_DETAILS = "pending_details"
        PASSED = "passed"
        STAND_BY = "stand_by"
        FINISHED = "finished"
    
    @dataclass
    class ConversationContext:
        state: ConversationState
        match_score: Optional[float] = None
        job_details: Optional[str] = None
        conversation_history: List[Dict] = None
        
        def __post_init__(self):
            if self.conversation_history is None:
                self.conversation_history = []
    
    # Test state transitions
    context = ConversationContext(state=ConversationState.PENDING_DETAILS)
    print(f"✅ Initial state: {context.state.value}")
    
    # Test state update
    context.state = ConversationState.PASSED
    context.match_score = 85.0
    print(f"✅ Updated state: {context.state.value}, Match score: {context.match_score}")
    
    return True

def test_response_generation():
    """Test response generation based on states"""
    print("\n🧪 Testing response generation...")
    
    class ConversationState(Enum):
        PENDING_DETAILS = "pending_details"
        PASSED = "passed"
        STAND_BY = "stand_by"
        FINISHED = "finished"
    
    def generate_response(message: str, state: ConversationState, match_score: Optional[float] = None) -> str:
        if state == ConversationState.PENDING_DETAILS:
            return "Thank you for reaching out! I'm interested in hearing more about this opportunity."
        elif state == ConversationState.PASSED:
            return f"Excellent! This opportunity looks like a great fit with a {match_score:.1f}% match."
        elif state == ConversationState.STAND_BY:
            return f"Thank you for sharing this opportunity! It seems interesting with a {match_score:.1f}% match."
        elif state == ConversationState.FINISHED:
            return f"Thank you for considering me! After reviewing, I don't think this is the right fit (match score: {match_score:.1f}%)."
        else:
            return "I'm processing your message. Please wait..."
    
    # Test responses
    test_cases = [
        (ConversationState.PENDING_DETAILS, None),
        (ConversationState.PASSED, 85.0),
        (ConversationState.STAND_BY, 70.0),
        (ConversationState.FINISHED, 45.0)
    ]
    
    for state, score in test_cases:
        response = generate_response("test message", state, score)
        print(f"State: {state.value}, Score: {score} -> Response: {response[:50]}...")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting AI Recruiter Assistant Tests\n")
    
    tests = [
        test_document_loading,
        test_linkedin_data,
        test_intent_detection,
        test_state_management,
        test_response_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to proceed with LLM integration.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 