# RAG Chatbot with PDF + Website Support (Offline via Ollama)

A fully offline Retrieval-Augmented Generation (RAG) chatbot that:
-  Answers questions based on website URLs or uploaded PDFs
-  Supports multilingual queries with translation
- Remembers chat history for context
- Runs **offline** using Ollama (e.g. `llama3`, `mistral`)
- Has a web UI via **Gradio**

---

## Features

- Web Scraping: Extracts clean HTML content from JS-based websites
- PDF Parsing: Reads and chunks PDF files
- RAG Pipeline: Uses FAISS vectorstore + LangChain retriever
- Multilingual: Auto-translates input/output using Deep Translator
- Memory: Retains chat history across turns
- Ollama LLMs: Offline language model (`llama3`)

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Demo (Run Locally)

```bash
git clone https://github.com/Fhareed/RAG-Chatbot-Offline-with-Ollama-.git
cd RAG-Chatbot-Offline-with-Ollama-
pip install -r requirements.txt

# Make sure Ollama is installed and a model like llama3 is pulled:
ollama run llama3

# Run the app
python rag_chatbot.py