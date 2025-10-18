import os
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np

from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from werkzeug.utils import secure_filename

# --- LLM / Transcription config ---
# Read from env (don’t hardcode secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# OpenAI client (lazy import only if key set)
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("Warning: OpenAI client init failed:", e)
        client = None

# Try offline transcription via faster-whisper if available
FAST_WHISPER = None
try:
    from faster_whisper import WhisperModel  # pip install faster-whisper
    # Force CPU to avoid CUDA/cuDNN issues on Windows
    FAST_WHISPER = WhisperModel(
        "base",            # or "tiny" for faster CPU
        device="cpu",
        compute_type="int8"  # good speed/quality trade-off on CPU
    )
except Exception as e:
    print("faster-whisper unavailable:", e)
    FAST_WHISPER = None

# --- multimodal deps ---
import fitz  # PyMuPDF
import docx
import mammoth
from PIL import Image
import pytesseract

from sentence_transformers import SentenceTransformer
import faiss

# Optional: get audio duration
AUDIO_DURATION_ENABLED = True
try:
    from pydub import AudioSegment  # pip install pydub; requires ffmpeg on system
except Exception:
    AUDIO_DURATION_ENABLED = False

# ---------------- Config ----------------
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = str(BASE_DIR / 'rag_local.db')
MEDIA_DIR = str(BASE_DIR / 'ingested_media')
INDEX_DIR = str(BASE_DIR / 'index_store')
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

DOC_EXT = {'.pdf', '.docx', '.doc', '.txt'}
IMG_EXT = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
AUDIO_EXT = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.webm'}
ALLOWED_EXT = DOC_EXT | IMG_EXT | AUDIO_EXT

# ---------------- DB setup ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    file_type TEXT,
    file_hash TEXT UNIQUE,
    created_at TEXT,
    orig_path TEXT
)''')

c.execute('''CREATE TABLE IF NOT EXISTS text_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    chunk_text TEXT,
    chunk_meta TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id)
)''')

c.execute('''CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    image_path TEXT,
    page_num INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id)
)''')

# NEW: audio metadata table
c.execute('''CREATE TABLE IF NOT EXISTS audios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    audio_path TEXT,
    duration_sec REAL,
    sample_rate INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id)
)''')

conn.commit()

# ---------------- Utilities ----------------
def file_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def save_uploaded_file(file_storage, dest_dir=MEDIA_DIR) -> str:
    filename = secure_filename(file_storage.filename)
    dest = os.path.join(dest_dir, filename)
    count = 1
    base, ext = os.path.splitext(filename)
    while os.path.exists(dest):
        filename = f"{base}_{count}{ext}"
        dest = os.path.join(dest_dir, filename)
        count += 1
    file_storage.save(dest)
    return dest

# ---------------- Extraction: docs/images ----------------
def process_pdf(file_path: str) -> Tuple[str, List[Tuple[str,int]]]:
    doc = fitz.open(file_path)
    text_pages, images = [], []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_pages.append(page.get_text())
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_name = f'{Path(file_path).stem}_page{page_num}_{img_index}.png'
            img_path = os.path.join(MEDIA_DIR, img_name)
            if pix.n - pix.alpha < 4:
                pix.save(img_path)
            else:
                pix0 = fitz.Pixmap(fitz.csRGB, pix)
                pix0.save(img_path)
                pix0 = None
            pix = None
            images.append((img_path, page_num))
    return '\n'.join(text_pages), images

def process_docx(file_path: str) -> Tuple[str, List[Tuple[str,int]]]:
    d = docx.Document(file_path)
    text = '\n'.join([p.text for p in d.paragraphs])
    return text, []

def process_doc(file_path: str) -> Tuple[str, List[Tuple[str,int]]]:
    with open(file_path, 'rb') as f:
        result = mammoth.extract_raw_text(f)
        return result.value, []

def ocr_image(file_path: str) -> Tuple[str, List[Tuple[str,int]]]:
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text, [(file_path, 0)]
    except Exception as e:
        print("OCR error:", e)
        return "", [(file_path, 0)]

# ---------------- Extraction: audio ----------------
def _audio_duration(file_path: str) -> float:
    if not AUDIO_DURATION_ENABLED:
        return None
    try:
        seg = AudioSegment.from_file(file_path)
        return round(len(seg) / 1000.0, 3)  # seconds
    except Exception:
        return None

def _transcribe_offline_faster_whisper(file_path: str) -> Tuple[str, int]:
    """
    Returns (text, sample_rate) using faster-whisper if available.
    """
    if FAST_WHISPER is None:
        return "", None
    try:
        segments, info = FAST_WHISPER.transcribe(file_path, beam_size=1)
        text = " ".join([s.text.strip() for s in segments if s.text])
        sr = getattr(info, "sample_rate", None)
        return text.strip(), sr
    except Exception as e:
        print("faster-whisper failed:", e)
        return "", None

def _transcribe_openai(file_path: str) -> Tuple[str, int]:
    """
    Returns (text, sample_rate). OpenAI Whisper returns only text; sample_rate unknown.
    """
    if client is None:
        return "", None
    try:
        with open(file_path, "rb") as af:
            # Whisper-1 transcription
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=af
            )
        text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else "")
        return (text or "").strip(), None
    except Exception as e:
        print("OpenAI Whisper failed:", e)
        return "", None

def process_audio(file_path: str) -> Tuple[str, float, int]:
    """
    Returns (transcript_text, duration_sec, sample_rate)
    Tries faster-whisper (offline) first, then OpenAI Whisper. If both fail, returns empty text.
    """
    duration = _audio_duration(file_path)
    # Try offline first (if installed)
    text, sr = _transcribe_offline_faster_whisper(file_path)
    if not text:
        # Try OpenAI Whisper if key present
        t2, sr2 = _transcribe_openai(file_path)
        text = t2
        sr = sr2
    return text, duration, (sr or None)

# ---------------- Ingestion ----------------
def ingest_file_disk(file_path: str):
    ext = Path(file_path).suffix.lower()
    with open(file_path, 'rb') as f:
        b = f.read()
    fh = file_hash_bytes(b)
    c.execute('SELECT id FROM documents WHERE file_hash=?', (fh,))
    if c.fetchone():
        return {"status": "exists", "file_hash": fh}

    text, images = "", []
    audio_meta = None  # (duration_sec, sample_rate)

    try:
        if ext == '.pdf':
            text, images = process_pdf(file_path)
        elif ext == '.docx':
            text, images = process_docx(file_path)
        elif ext == '.doc':
            text, images = process_doc(file_path)
        elif ext in IMG_EXT:
            text, images = ocr_image(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext in AUDIO_EXT:
            # process audio: store file + (optional) transcript to text_chunks
            t, duration, sr = process_audio(file_path)
            text = t or ""  # may be empty if transcription unavailable
            audio_meta = (duration, sr)
        else:
            return {"status": "unsupported", "ext": ext}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    now = datetime.now(timezone.utc).isoformat()
    c.execute('INSERT INTO documents (file_name, file_type, file_hash, created_at, orig_path) VALUES (?, ?, ?, ?, ?)',
              (Path(file_path).name, ext, fh, now, file_path))
    doc_id = c.lastrowid
    conn.commit()

    # Save extracted text (including transcribed audio) if present
    if text and text.strip():
        meta = {"source": Path(file_path).name}
        if ext in AUDIO_EXT:
            meta["modality"] = "audio_transcript"
        c.execute('INSERT INTO text_chunks (document_id, chunk_text, chunk_meta) VALUES (?, ?, ?)',
                  (doc_id, text, json.dumps(meta)))
        conn.commit()

    # Save image references (from PDFs or images)
    for img_path, page_num in images:
        c.execute('INSERT INTO images (document_id, image_path, page_num) VALUES (?, ?, ?)',
                  (doc_id, img_path, page_num))
        conn.commit()

    # Save audio metadata if applicable
    if ext in AUDIO_EXT:
        dur, sr = (audio_meta or (None, None))
        c.execute('INSERT INTO audios (document_id, audio_path, duration_sec, sample_rate) VALUES (?, ?, ?, ?)',
                  (doc_id, file_path, dur, sr))
        conn.commit()

    return {"status": "ingested", "doc_id": doc_id, "file_hash": fh, "file_type": ext}

# ---------------- Embedding + FAISS ----------------
EMBED_MODEL = SentenceTransformer('intfloat/e5-small-v2')
INDEX_FILE = os.path.join(INDEX_DIR, 'faiss_e5_small.index')
ID_MAP_FILE = os.path.join(INDEX_DIR, 'id_mapping.jsonl')

def _chunk_text(text: str, max_chars=1000, overlap=200) -> List[str]:
    text = text.strip()
    if not text: return []
    chunks, start, L = [], 0, len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunks.append(text[start:end].strip())
        if end == L: break
        start = end - overlap
    return chunks

def build_faiss_index():
    c.execute('SELECT id, document_id, chunk_text, chunk_meta FROM text_chunks')
    rows = c.fetchall()
    pieces = []
    for row in rows:
        parent_id, doc_id, text, meta = row
        meta_dict = json.loads(meta) if meta else {}
        for piece in _chunk_text(text):
            pieces.append((parent_id, doc_id, piece, meta_dict))
    if not pieces: return {"status": "no_text"}

    texts = [p[2] for p in pieces]
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    mapping = []
    for i, p in enumerate(pieces):
        mapping.append({
            "faiss_idx": i,
            "parent_chunk_id": p[0],
            "document_id": p[1],
            "text": p[2],
            "meta": p[3]
        })
    with open(ID_MAP_FILE, 'w', encoding='utf-8') as f:
        for m in mapping:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    return {"status": "built", "num_pieces": len(pieces)}

def load_faiss_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(ID_MAP_FILE):
        index = faiss.read_index(INDEX_FILE)
        id_map = [json.loads(line) for line in open(ID_MAP_FILE, 'r', encoding='utf-8')]
        return index, id_map
    return None, None

def semantic_search(query: str, top_k=5):
    index, id_map = load_faiss_index()
    if index is None: return []
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        entry = id_map[idx]
        doc_id = entry['document_id']
        c.execute('SELECT file_name FROM documents WHERE id=?', (doc_id,))
        r = c.fetchone()
        fn = r[0] if r else None
        results.append({
            "faiss_idx": int(idx),
            "score": float(score),
            "document_id": int(doc_id),
            "file_name": fn,
            "text": entry['text'],
            "meta": entry.get('meta', {})
        })
    return results

# ---------------- Summarization (grounded) ----------------
def generate_grounded_summary(query: str, hits: List[dict], max_tokens=300) -> str:
    if OPENAI_API_KEY and client:
        prompt_snippets = []
        for i, h in enumerate(hits, start=1):
            src = h.get('file_name') or f"doc_{h['document_id']}"
            snippet = h.get('text', '').strip().replace('\n', ' ')
            if len(snippet) > 400: snippet = snippet[:400].rsplit(' ', 1)[0] + "…"
            prompt_snippets.append(f"[{i}] Source: {src}\nText: {snippet}")

        system = (
            "You are a helpful assistant that answers user queries using only the provided source snippets. "
            "Always include numbered citations like [1], [2]. If info not found, say so clearly."
        )
        user = f"User query: {query}\n\nSources:\n" + "\n\n".join(prompt_snippets)

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI summarization failed:", e)

    # fallback extractive
    lines = []
    for i, h in enumerate(hits, start=1):
        snippet = h.get('text', '').strip()
        if len(snippet) > 300: snippet = snippet[:300].rsplit(' ', 1)[0] + "…"
        lines.append(f"[{i}] {snippet}")
    return "Extractive snippets (top results):\n" + "\n\n".join(lines)

# ---------------- Flask App ----------------
app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "no selected file"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "unsupported file type", "ext": ext}), 400
    saved = save_uploaded_file(f, dest_dir=MEDIA_DIR)
    result = ingest_file_disk(saved)
    return jsonify(result)

@app.route('/api/build_index', methods=['POST'])
def api_build_index():
    res = build_faiss_index()
    return jsonify(res)

def json_safe(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"Object {type(o)} not serializable")

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json or {}
    q = (data.get('query') or "").strip()
    if not q:
        return jsonify({"error": "must provide 'query'"}), 400

    # fetch exactly one result
    hits = semantic_search(q, top_k=1)
    if not hits:
        # keep legacy shape so UI doesn't break
        return jsonify({"hits": [], "summary": "", "message": "no match"}), 200

    h = hits[0]
    doc_id = h['document_id']

    # ensure we always have a file_url
    c.execute('SELECT file_name FROM documents WHERE id=?', (doc_id,))
    r = c.fetchone()
    file_name = (r[0] if r else h.get('file_name')) or ""
    file_url = url_for('media', filename=file_name, _external=False) if file_name else None

    # return minimal, but compatible with UI
    minimal_hit = {
        "document_id": doc_id,
        "file_name": file_name,
        "file_url": file_url,
        "text": h.get("text", "")
        # (no score, no faiss_idx, no meta)
    }

    # if your UI also reads `summary`, reuse text there
    return jsonify({
        "hits": [minimal_hit],
        "summary": minimal_hit["text"]
    })

@app.route('/api/list_docs', methods=['GET'])
def api_list_docs():
    c.execute('SELECT id, file_name, file_type, created_at FROM documents ORDER BY id DESC')
    rows = c.fetchall()
    out = [{"id": r[0], "file_name": r[1], "file_type": r[2], "created_at": r[3],
            "url": url_for('media', filename=r[1], _external=False)} for r in rows]
    return jsonify(out)

# if __name__ == '__main__':
#     print("🚀 Starting Flask app — open http://127.0.0.1:5000")
#     app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", "7860"))
    print(f"🚀 Starting Flask app — open http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)




