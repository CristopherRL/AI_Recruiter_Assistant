# AI Recruiter Assistant 🤖

An intelligent conversational chatbot designed to pre-screen job offers from recruiters using Retrieval-Augmented Generation (RAG) with a powerful open-source Large Language Model.

## 🎯 Project Overview

This AI assistant automates the initial screening of job offers by:
1.  **Intent Detection**: Analyzing if messages are job offers or generic inquiries.
2.  **RAG Analysis**: Augmenting the LLM with real-time context from your CV and job preferences.
3.  **Match Scoring**: Calculating a compatibility score based on the provided context.
4.  **Smart Responses**: Generating contextual, human-like responses based on the match score.

## 🏗️ Architecture

The system follows a **RAG-first approach** to ensure contextual accuracy and optimal performance without the immediate need for fine-tuning. This strategy prioritizes context-aware prompt engineering over model specialization.

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Web Interface  │◄───►│  State Manager  │◄───►│   RAG Pipeline  │
│    (Gradio)     │      │ (Conversation)  │      │ (FAISS VectorDB)│
└─────────────────┘      └─────────────────┘      └─────────────────┘
         ▲                                                 │
         │                                                 ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  User/Recruiter │      │ Prompt Engineer │◄───►│  Retrieved Docs │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                         ▲                       ▲
         └─────────────────────────┼───────────────────────┘
                                   ▼
                          ┌─────────────────┐
                          │   Base LLM      │
                          │ (No Fine-tuning)│
                          └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
- Ensure `cv.md` contains your CV/resume
- Ensure `job_expectations.md` contains your job preferences
- Place LinkedIn messages in `data/linkedin_messages.csv`

### 3. Run the Notebook
```bash
# Open the Jupyter notebook
jupyter notebook ai_recruiter_assistant_colab.ipynb
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
- **LLM**: Multiple model options (Mistral-7B, Llama-3-8B, Phi-3-mini, Gemma-2-9B)
- **Strategy**: RAG-first approach with advanced prompt engineering
- **RAG**: FAISS vector database + sentence transformers
- **Framework**: LangChain for orchestration
- **Interface**: Gradio web application

### Model Selection
We benchmark **4 open-source models** to select the best performer for our RAG-based approach:

| Model | Size | Context | Multimodal | Description |
|-------|------|---------|------------|-------------|
| **Mistral-7B-Instruct-v0.2** | 7B | 32K | 📝 No | Efficient instruction-following |
| **Meta-Llama-3-8B-Instruct** | 8B | 8K | 📝 No | Strong reasoning capabilities |
| **Microsoft/Phi-3-mini-4k** | 3.8B | 4K | 📝 No | Lightweight, fast inference |
| **Google/Gemma-2-9B-IT** | 9B | 8K | 📝 No | Google's efficient instruction-tuned |

### Key Libraries
- `transformers`: Hugging Face transformers
- `langchain`: LLM orchestration
- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Embedding generation
- `gradio`: Web interface

## 📁 Project Structure

```
AI_Recruiter_Assistant/
├── ai_recruiter_assistant_colab.ipynb # Main notebook
├── requirements.txt                    # Dependencies
├── README.md                          # This file
├── PROJECT_SUMMARY.md                 # Development status
├── RAG/
│   ├── cv.md                         # Your CV/resume
│   └── job_expectations.md           # Job preferences
├── data/
│   └── linkedin_messages.csv         # Historical data
└── notebooks/                        # Development iterations
    └── *.ipynb                       # Version history
```

## 🔧 Development Plan (RAG-First Approach)

### Day 1-2: Foundation ✅ COMPLETED
- [x] Environment setup
- [x] Data loading and processing
- [x] RAG pipeline implementation
- [x] Vector database creation

### Day 3-4: Model Selection & Optimization 🔄 CURRENT
- [x] LLM model benchmarking
- [x] Performance comparison
- [x] Model selection and optimization
- [x] Cache management setup

### Day 5-6: Prompt Engineering & Logic 📋 NEXT
- [x] Context-aware prompt design
- [ ] Intent detection enhancement
- [ ] Match scoring refinement
- [ ] Response generation optimization

### Day 7: Integration & Testing 🚀 FINAL
- [ ] Gradio interface
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Deployment and documentation

## 🎯 Features

### ✅ Implemented
- Document loading and processing
- RAG pipeline with FAISS vector database
- State management system
- Basic intent detection
- Match scoring algorithm
- Response generation templates
- Model benchmarking infrastructure

### 🚧 In Progress
- Model selection and optimization
- Advanced prompt engineering
- Context-aware retrieval

### 🔮 Future Enhancements (Post-RAG Optimization)
- Fine-tuning with QLoRA (when performance plateau is reached)
- Google Calendar integration
- Advanced conversation memory
- Multi-language support
- Analytics dashboard

## 🤝 Usage

1. **Open the notebook** in Google Colab
2. **Run all cells** to set up the environment
3. **Load your data** (CV, expectations, LinkedIn messages)
4. **Select optimal model** based on benchmark results
5. **Test prompt engineering** with RAG retrieval
6. **Launch the Gradio interface** for testing
7. **Deploy** for production use

## 📈 Performance Metrics

- **Response Time**: < 5 seconds
- **Match Accuracy**: 85%+ (target)
- **Memory Usage**: Optimized for Colab free tier
- **Model Size**: ~4GB (quantized)
- **RAG Retrieval**: < 1 second

## 🔒 Privacy & Security

- All data processed locally
- No external API calls required
- Open-source model ensures transparency
- No data sent to third parties

## 🎓 Development Methodology

This project follows a **4-stage Generative AI lifecycle**:

1. **Define the Scope** ✅ - Problem identification and requirements
2. **Select Models** 🔄 - Benchmark and choose optimal LLM
3. **Adapt & Align** 📋 - RAG optimization and prompt engineering
4. **Application Integration** 🚀 - Deploy and test complete system

**Current Strategy**: RAG-first approach with fine-tuning postponed to future iterations, focusing on maximizing base model potential through advanced context retrieval and prompt engineering.

## 📞 Support

For questions or issues:
1. Check the notebook comments
2. Review the conversation flow logic
3. Verify your data format
4. Test with sample messages

---

**Built with ❤️ using open-source AI technologies** 