import os
import gradio as gr
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from deep_translator import GoogleTranslator
from langchain_ollama import OllamaLLM
from playwright.sync_api import sync_playwright

# 1. Scrape JS websites using Playwright
def scrape_website(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state('networkidle')
        content = page.inner_text("body")
        browser.close()
        return content

# 2. Read PDF content
def read_pdf(file):
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# 3. Convert text to vector store
def process_text_to_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# 4. Create chatbot engine using Ollama (offline)
chat_chain = None

def create_chatbot(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = OllamaLLM(model="llama3")
    return ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)

# 5. Load website or PDF sources
def load_sources(website_url, pdf_file):
    combined_text = ""

    if website_url:
        combined_text += scrape_website(website_url)

    if pdf_file:
        combined_text += read_pdf(pdf_file)

    if not combined_text.strip():
        return "No content found.", None

    vectorstore = process_text_to_vectorstore(combined_text)
    global chat_chain
    chat_chain = create_chatbot(vectorstore)
    return "Sources loaded successfully!", None

# 6. Chat interface
def chatbot_interface(user_input, lang='en'):
    global chat_chain
    if not chat_chain:
        return "Please load content first."

    input_translated = GoogleTranslator(source=lang, target='en').translate(user_input)
    response = chat_chain.invoke(input_translated)

    if isinstance(response, dict) and 'answer' in response:
        response_text = response['answer']
    else:
        response_text = str(response)

    if len(response_text) > 5000:
        response_text = response_text[:4999]

    response_translated = GoogleTranslator(source='en', target=lang).translate(response_text)
    return response_translated

# 7. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Website + PDF Chatbot (Multilingual, with Memory, Offline via Ollama)")

    with gr.Row():
        website_input = gr.Textbox(label="Website URL", placeholder="https://example.com")
        pdf_input = gr.File(label="Upload PDF", file_types=['.pdf'])

    load_btn = gr.Button("Load Content")
    status = gr.Textbox(label="Status")

    with gr.Row():
        lang_selector = gr.Dropdown(choices=['en', 'fi', 'fr','sv', 'de', 'es'], value='en', label="Response Language")
        user_query = gr.Textbox(label="Ask your question:")
        response_output = gr.Textbox(label="Bot Response")

    load_btn.click(fn=load_sources, inputs=[website_input, pdf_input], outputs=[status, pdf_input])
    user_query.submit(fn=chatbot_interface, inputs=[user_query, lang_selector], outputs=response_output)

if __name__ == "__main__":
    demo.launch()

