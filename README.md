# 🧠 RAG-based Generative AI System

A **Retrieval-Augmented Generation (RAG)** application that combines the power of **Large Language Models (LLMs)** with **vector-based document retrieval** to deliver accurate, context-aware, and verifiable responses.

This project enables users to query their **private or domain-specific data** using natural language — making it ideal for **chatbots, document assistants, and knowledge-driven systems**.

---

## 🚀 Features

- 🔍 **Context Retrieval:** Fetches the most relevant chunks from documents using embeddings.  
- 🧩 **Generative Response:** Uses an LLM (Llama) to generate grounded answers.  
- 💾 **Vector Database Integration:** Supports FAISS / Chroma / Pinecone for similarity search.  
- 🗂️ **Multi-format Document Support:** Handles PDFs, text files, and more.  
- 🧠 **Memory & Chat History:** Retains conversation context.  
- ⚙️ **Purely local: doesnt run on any API using llama cpp python model to generate answers.

---

## 🏗️ System Architecture

```
             ┌────────────────────────┐
             │      User Query         │
             └────────────┬────────────┘
                          │
                Natural Language Input
                          │
            ┌─────────────▼─────────────┐
            │     Embedding Model       │
            └─────────────┬─────────────┘
                          │
                 Similarity Search in
                ┌─────────────────────┐
                │  Vector Database    │
                └────────┬────────────┘
                         │ Retrieved context
            ┌────────────▼────────────┐
            │     LLM (Generator)     │
            └────────────┬────────────┘
                         │
                 ┌───────▼────────┐
                 │  Final Answer  │
                 └────────────────┘
```

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend (optional)** | React.js / Streamlit |
| **Backend** | FastAPI / Flask |
| **LLM** | cpp Llama 3 |
| **Embeddings** | OpenAI Embeddings / SentenceTransformers |
| **Vector Store** | FAISS / Chroma / Pinecone |
| **Storage** | Local Files / Cloud Bucket |
| **Deployment** | Docker / AWS / Vercel / Streamlit Cloud |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/rag-genai-project.git
cd rag-genai-project
```




```

### 5️⃣ Run the Application
```bash
docker run --rm -it -p 5001:5000 -v ./data:/app/data cyberdocai
```
will run on localhost 5000



---

## 📚 Example Usage

**Query:**
> “Summarize the main findings of the company’s 2024 annual report.”

**RAG Pipeline Output:**
> “According to the 2024 Annual Report, the company saw a 23% growth in digital sales and a 12% increase in overall revenue, primarily driven by e-commerce and subscription-based models.”

---



---



---

## 🤝 Contributing

Contributions are welcome!  
To contribute:
1. Fork this repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Added new feature"`)  
4. Push to your branch (`git push origin feature-name`)  
5. Open a pull request 🎉  

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

