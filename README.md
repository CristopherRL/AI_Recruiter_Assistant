# AI Recruiter Assistant 🤖

An intelligent conversational chatbot designed to pre-screen job offers from recruiters using advanced AI techniques including RAG (Retrieval-Augmented Generation) and fine-tuned language models.

## 🎯 Project Overview

This AI assistant automates the initial screening of job offers by:
1. **Intent Detection**: Analyzing if messages are job offers or generic inquiries
2. **RAG Analysis**: Comparing job descriptions against your CV and preferences
3. **Match Scoring**: Calculating compatibility percentages
4. **Smart Responses**: Generating appropriate responses based on match scores

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  State Manager  │    │   RAG Pipeline  │
│   (Gradio)      │◄──►│  (Conversation  │◄──►│  (Vector DB +   │
│                 │    │   States)       │    │   Retrieval)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Fine-tuned LLM │
                       │  (LoRA + Open   │
                       │   Source Model) │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data
- Ensure `cv.md` contains your CV/resume
- Ensure `job_expectations.md` contains your job preferences
- Place LinkedIn messages in `data/linkedin_messages.csv`

### 3. Run the Notebook
```bash
# Open the Jupyter notebook
jupyter notebook ai_recruiter_assistant.ipynb
```

## 📊 Conversation Flow

### State Management
- **`pending_details`**: Waiting for job details
- **`passed`**: High match (>80%) - ready to schedule call
- **`stand_by`**: Medium match (60-80%) - manual review needed
- **`finished`**: Low match (<60%) - politely decline

### Match Scoring Logic
- **>80%**: Positive response, request call scheduling
- **60-80%**: Cordial response, manual review needed
- **<60%**: Polite decline with explanation

## 🛠️ Technical Stack

### Core Technologies
- **LLM**: Mistral-7B-Instruct-v0.2 (open-source)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **RAG**: FAISS vector database + sentence transformers
- **Framework**: LangChain for orchestration
- **Interface**: Gradio web application

### Key Libraries
- `transformers`: Hugging Face transformers
- `peft`: Parameter-Efficient Fine-Tuning
- `langchain`: LLM orchestration
- `faiss-cpu`: Vector similarity search
- `gradio`: Web interface

## 📁 Project Structure

```
AI_Recruiter_Assistant/
├── ai_recruiter_assistant.ipynb    # Main notebook
├── requirements.txt                 # Dependencies
├── README.md                       # This file
├── RAG/
│   ├── cv.md                      # Your CV/resume
│   └── job_expectations.md        # Job preferences
└── data/
    └── linkedin_messages.csv      # Training data
```

## 🔧 Development Plan (1 Week)

### Day 1-2: Foundation
- [x] Environment setup
- [x] Data loading and processing
- [x] RAG pipeline implementation
- [x] Vector database creation

### Day 3-4: Fine-tuning
- [ ] LinkedIn messages processing
- [ ] LoRA fine-tuning implementation
- [ ] Model training and validation

### Day 5-6: Core Logic
- [ ] Intent detection system
- [ ] Match scoring algorithm
- [ ] State management
- [ ] Response generation

### Day 7: Integration
- [ ] Gradio interface
- [ ] End-to-end testing
- [ ] Deployment and documentation

## 🎯 Features

### ✅ Implemented
- Document loading and processing
- RAG pipeline with vector database
- State management system
- Intent detection (basic)
- Match scoring algorithm
- Response generation templates

### 🚧 In Progress
- LLM loading and fine-tuning
- Advanced intent detection
- Gradio web interface

### 🔮 Future Enhancements
- Google Calendar integration
- Advanced conversation memory
- Multi-language support
- Analytics dashboard

## 🤝 Usage

1. **Open the notebook** in Google Colab
2. **Run all cells** to set up the environment
3. **Load your data** (CV, expectations, LinkedIn messages)
4. **Fine-tune the model** with your conversational style
5. **Launch the Gradio interface** for testing
6. **Deploy** for production use

## 📈 Performance Metrics

- **Response Time**: < 5 seconds
- **Match Accuracy**: 85%+ (target)
- **Memory Usage**: Optimized for Colab free tier
- **Model Size**: ~4GB (quantized)

## 🔒 Privacy & Security

- All data processed locally
- No external API calls required
- Open-source model ensures transparency
- No data sent to third parties

## 📞 Support

For questions or issues:
1. Check the notebook comments
2. Review the conversation flow logic
3. Verify your data format
4. Test with sample messages

---

**Built with ❤️ using open-source AI technologies** 