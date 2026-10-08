import os
import shutil
import traceback
import sqlite3
import hashlib
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from pypdf import PdfReader

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONSTANTS ---
DATA_DIR = 'data'
UPLOAD_FOLDER_BASE = os.path.join(DATA_DIR, 'uploads')
VECTORSTORE_BASE = os.path.join(DATA_DIR, 'vectorstores')
DATABASE_PATH = os.path.join(DATA_DIR, 'users.db')
MODEL_PATH = "/app/models/qwen2-1_5b-instruct-q4_k_m.gguf"

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.urandom(24)
os.makedirs(UPLOAD_FOLDER_BASE, exist_ok=True)
os.makedirs(VECTORSTORE_BASE, exist_ok=True)

# --- GLOBAL RAG COMPONENTS ---
embeddings = None
llm = None
qa_chain = None

# --- AI MODEL INITIALIZATION ---
def initialize_ai_components():
    global embeddings, llm, qa_chain
    print("Initializing embedding model (local)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model initialized.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize embeddings. Error: {e}")
        return

    print("Initializing local LLM (Qwen2 1.5B) with LlamaCpp...")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"CRITICAL: Model file not found at {MODEL_PATH}.")
            return

        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=0,
            n_ctx=4096,
            temperature=0.2,
            max_tokens=1024,
            repetition_penalty=1.15,
            verbose=False,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        print("Local LLM (Qwen2) initialized successfully.")

        prompt_template = """<|im_start|>system
You are a helpful cybersecurity assistant. Answer the user's question based only on the provided context. If the context does not contain the answer, say "The provided documents do not contain sufficient information on this topic."<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{input}<|im_end|>
<|im_start|>assistant
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
        qa_chain = create_stuff_documents_chain(llm, PROMPT)

    except Exception as e:
        print(f"Could not initialize AI components: {e}")
        llm = None
        qa_chain = None

# --- DATABASE & HELPER FUNCTIONS ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn
def init_db():
    with app.app_context():
        conn = get_db_connection()
        conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, filename TEXT NOT NULL, FOREIGN KEY (user_id) REFERENCES users (id))')
        conn.commit()
        conn.close()
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
def get_user_folders(user_id):
    user_upload_folder = os.path.join(UPLOAD_FOLDER_BASE, f"user_{user_id}")
    user_vectorstore_path = os.path.join(VECTORSTORE_BASE, f"user_{user_id}")
    os.makedirs(user_upload_folder, exist_ok=True)
    return user_upload_folder, user_vectorstore_path
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        try:
            pdf_reader = PdfReader(pdf_path, strict=False)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    clean_text = page_text.encode('ascii', 'ignore').decode('ascii')
                    text += clean_text
        except Exception as e:
            print(f"Error reading PDF {os.path.basename(pdf_path)}: {e}")
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, length_function=len)
    return text_splitter.split_text(text)
def create_and_save_vectorstore(text_chunks, embeddings_model, vectorstore_path):
    if not text_chunks: return None
    try:
        new_vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings_model)
        if os.path.exists(vectorstore_path): shutil.rmtree(vectorstore_path)
        new_vector_store.save_local(vectorstore_path)
        return new_vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None
def rebuild_user_vectorstore(user_id):
    if not embeddings: return False
    user_upload_folder, user_vectorstore_path = get_user_folders(user_id)
    conn = get_db_connection()
    docs_cursor = conn.execute('SELECT filename FROM documents WHERE user_id = ?', (user_id,))
    user_docs = [row['filename'] for row in docs_cursor.fetchall()]
    conn.close()
    doc_paths = [os.path.join(user_upload_folder, f) for f in user_docs]
    if not doc_paths:
        if os.path.exists(user_vectorstore_path): shutil.rmtree(user_vectorstore_path)
        return True
    try:
        raw_text = get_pdf_text(doc_paths)
        text_chunks = get_text_chunks(raw_text)
        create_and_save_vectorstore(text_chunks, embeddings, user_vectorstore_path)
        return True
    except Exception:
        return False

# --- ROUTES ---
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json; username = data.get('username'); email = data.get('email'); password = data.get('password')
    if not all([username, email, password]): return jsonify({"message": "Missing required fields"}), 400
    password_hash = hash_password(password)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", (username, email, password_hash))
        conn.commit(); user_id = cursor.lastrowid; session['user_id'] = user_id
        return jsonify({"message": "Registration successful", "user": {"id": user_id, "username": username, "email": email}}), 201
    except sqlite3.IntegrityError: return jsonify({"message": "Username or email already exists"}), 409
    finally: conn.close()
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json; email_or_username = data.get('email'); password = data.get('password')
    password_hash = hash_password(password)
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE (email = ? OR username = ?) AND password_hash = ?', (email_or_username, email_or_username, password_hash)).fetchone()
    conn.close()
    if user: session['user_id'] = user['id']; return jsonify({"message": "Login successful", "user": {"id": user['id'], "username": user['username'], "email": user['email']}})
    return jsonify({"message": "Invalid credentials"}), 401
@app.route('/api/logout', methods=['GET'])
def logout():
    session.pop('user_id', None)
    return jsonify({"message": "Logged out"})
@app.route('/api/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        conn = get_db_connection()
        user = conn.execute('SELECT id, username, email FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        if user: return jsonify({"logged_in": True, "user": dict(user)})
    return jsonify({"logged_in": False})
@app.before_request
def require_login():
    public_endpoints = ['root', 'static', 'login', 'register', 'check_session']
    if request.endpoint and request.endpoint not in public_endpoints and 'user_id' not in session:
        return jsonify({"message": "Authentication required"}), 401
@app.route('/api/upload', methods=['POST'])
def upload_files():
    user_id = session['user_id']; user_upload_folder, _ = get_user_folders(user_id)
    files = request.files.getlist('files'); conn = get_db_connection()
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            if not conn.execute('SELECT id FROM documents WHERE user_id = ? AND filename = ?', (user_id, filename)).fetchone():
                file.save(os.path.join(user_upload_folder, filename))
                conn.execute('INSERT INTO documents (user_id, filename) VALUES (?, ?)', (user_id, filename))
    conn.commit(); conn.close()
    if not rebuild_user_vectorstore(user_id): return jsonify({'success': False, 'message': 'Failed to process documents.'}), 500
    return jsonify({'success': True, 'message': 'Files processed.'})
@app.route('/api/documents', methods=['GET'])
def get_documents():
    user_id = session['user_id']; conn = get_db_connection()
    docs_cursor = conn.execute('SELECT filename FROM documents WHERE user_id = ?', (user_id,))
    docs = [row['filename'] for row in docs_cursor.fetchall()]; conn.close()
    return jsonify({'documents': docs})
@app.route('/api/delete/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    user_id = session['user_id']; user_upload_folder, _ = get_user_folders(user_id)
    secure_name = secure_filename(filename); conn = get_db_connection()
    conn.execute('DELETE FROM documents WHERE user_id = ? AND filename = ?', (user_id, secure_name))
    conn.commit(); conn.close()
    filepath = os.path.join(user_upload_folder, secure_name)
    if os.path.exists(filepath): os.remove(filepath)
    rebuild_user_vectorstore(user_id)
    return jsonify({'success': True, 'message': f'File {filename} deleted.'})
@app.route('/api/chat', methods=['POST'])
def chat():
    user_id = session['user_id']; _, user_vectorstore_path = get_user_folders(user_id)
    user_message = request.json.get('message')
    if not qa_chain: return jsonify({'reply': 'AI components not available.'}), 503
    if not os.path.exists(user_vectorstore_path): return jsonify({'reply': 'Please upload documents first.'}), 400
    try:
        current_vector_store = FAISS.load_local(user_vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        # --- THE FIX IS HERE: Changed `user_.message` to the correct `user_message` variable ---
        docs = current_vector_store.similarity_search(user_message, k=3)
        if not docs: return jsonify({'reply': "Could not find relevant information in your documents."})
        response = qa_chain.invoke({"input": user_message, "context": docs})
        return jsonify({'reply': response.strip()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'reply': 'An error occurred during chat processing.'}), 500
@app.route('/api/profile', methods=['POST'])
def update_profile():
    return jsonify({"message": "Profile updated successfully (simulated)"})
@app.route('/api/delete_account', methods=['DELETE'])
def delete_account():
    return jsonify({"message": "Account deleted successfully (simulated)"})

if __name__ == '__main__':
    print("Starting Flask server...")
    with app.app_context():
        init_db()
    initialize_ai_components()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
