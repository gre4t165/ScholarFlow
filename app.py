"""
ScholarFlow - AI Systematic Literature Review (SLR) Workflow
Multi-step application for processing large volumes of academic papers.
"""
import streamlit as st

# --- BAGIAN 1: CEK LOGIN ---
# Jika belum login, tampilkan input password
if not st.session_state.get("password_correct", False):
    pwd = st.text_input("Masukkan Password", type="password")
    
    # Cek saat tombol ditekan
    if st.button("Masuk"):
        if pwd == st.secrets["password_akses"]:  # Cek ke secrets
            st.session_state["password_correct"] = True
            st.rerun()  # Refresh halaman
        else:
            st.error("Password salah!")
    
    st.stop()  # ⛔ INI RAHASIANYA: Kode di bawah ini TIDAK akan jalan kalau belum login

# --- BAGIAN 2: APLIKASI UTAMA ---
# Tulis kode aplikasi Anda seperti biasa di sini (tidak perlu masuk ke dalam else)

st.title("✅ Dashboard Utama")
st.write("Selamat! Anda berhasil masuk.")
st.write("Ini adalah data rahasia...")

# Tombol Logout (Simpel)
if st.button("Logout"):
    st.session_state["password_correct"] = False
    st.rerun()


import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from llama_index.core import Document, VectorStoreIndex

# --- Constants ---
TEMP_PAPERS_DIR = Path("./data/temp_papers")
PROVIDERS = {
    "openai": "OpenAI (GPT-4o)",
    "gemini": "Google Gemini (3.0 Pro)",
    "anthropic": "Anthropic (Claude 3.5 Sonnet)",
    "ollama": "Ollama (Local)",
}


# --- LLM Initialization ---
def _get_llm_and_embed(provider: str, api_key: str, base_url: str) -> tuple[Any, Any]:
    """Initialize LLM and embed_model based on provider."""
    provider = provider.lower()

    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not api_key or not api_key.strip():
            raise ValueError("API Key OpenAI belum dimasukkan.")
        llm = OpenAI(model="gpt-4o", api_key=api_key.strip())
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key.strip())

    elif provider == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        if not api_key or not api_key.strip():
            raise ValueError("API Key Google belum dimasukkan.")
        llm = GoogleGenAI(model="models/gemini-3.0-pro", api_key=api_key.strip())
        embed_model = GoogleGenAIEmbedding(
            model_name="models/embedding-001",
            api_key=api_key.strip(),
        )

    elif provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        if not api_key or not api_key.strip():
            raise ValueError("API Key Anthropic belum dimasukkan.")
        llm = Anthropic(model="claude-3-5-sonnet-20241022", api_key=api_key.strip())
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5",
            trust_remote_code=True,
        )

    elif provider == "ollama":
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        url = (base_url or "http://localhost:11434").strip().rstrip("/")
        llm = Ollama(model="llama3", base_url=url, request_timeout=120.0)
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url=url,
        )

    else:
        raise ValueError(f"Provider tidak dikenal: {provider}")

    return llm, embed_model


def _extract_json_from_response(text: str) -> dict | None:
    """Parse JSON from LLM response (handle markdown code block)."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# --- Step 1: Upload & Criteria ---
def save_uploaded_files(uploaded_files: list) -> list[dict]:
    """Save uploaded files to local temp directory and return metadata."""
    TEMP_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            continue
        
        file_path = TEMP_PAPERS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        saved_files.append({
            "filename": uploaded_file.name,
            "path": str(file_path),
            "size_kb": round(len(uploaded_file.getvalue()) / 1024, 1),
        })
    
    return saved_files


# --- Step 2: AI Screening ---
_SCREENING_PROMPT = """Anda adalah asisten review literatur akademik. Tugas Anda adalah menilai apakah paper ini RELEVAN berdasarkan kriteria inklusi yang diberikan.

KRITERIA INKLUSI:
{criteria}

TEKS PAPER (Halaman Pertama - Abstrak/Judul):
---
{abstract_text}
---

Analisis dengan seksama apakah paper ini memenuhi kriteria inklusi.

Respond HANYA dengan JSON valid berikut (tanpa teks lain):
{{
  "title": "judul paper yang terdeteksi",
  "reason": "alasan singkat 1-2 kalimat mengapa INCLUDE atau EXCLUDE",
  "status": "INCLUDE" atau "EXCLUDE"
}}"""


def extract_first_page(file_path: str) -> str:
    """Extract text from the first page (abstract/title) of a PDF."""
    try:
        reader = PdfReader(file_path)
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text() or ""
            return text.strip()
        return ""
    except Exception as e:
        return f"Error reading PDF: {e}"


def screen_paper(file_path: str, criteria: str, llm) -> dict:
    """Screen a single paper using LLM based on inclusion criteria."""
    abstract_text = extract_first_page(file_path)
    
    if not abstract_text or abstract_text.startswith("Error"):
        return {
            "title": Path(file_path).name,
            "reason": abstract_text or "Tidak dapat membaca PDF",
            "status": "EXCLUDE",
        }
    
    prompt = _SCREENING_PROMPT.format(criteria=criteria, abstract_text=abstract_text[:4000])
    
    try:
        response = llm.complete(prompt)
        resp_text = getattr(response, "text", None) or str(response)
        data = _extract_json_from_response(resp_text)
        
        if data:
            return {
                "title": data.get("title", Path(file_path).name),
                "reason": data.get("reason", "N/A"),
                "status": data.get("status", "EXCLUDE").upper(),
            }
        else:
            return {
                "title": Path(file_path).name,
                "reason": "Gagal parse respons LLM",
                "status": "EXCLUDE",
            }
    except Exception as e:
        return {
            "title": Path(file_path).name,
            "reason": f"Error: {e}",
            "status": "EXCLUDE",
        }


def run_screening(files_info: list, criteria: str, provider: str, api_key: str, base_url: str) -> list[dict]:
    """Run AI screening on all uploaded papers."""
    try:
        llm, _ = _get_llm_and_embed(provider, api_key, base_url)
    except ValueError as e:
        return [{"title": "Error", "reason": str(e), "status": "EXCLUDE"}]
    
    results = []
    for file_info in files_info:
        result = screen_paper(file_info["path"], criteria, llm)
        result["filename"] = file_info["filename"]
        result["path"] = file_info["path"]
        results.append(result)
    
    return results


# --- Step 3: PICO Extraction ---
_PICO_PROMPT = """Anda adalah asisten ekstraksi data untuk Systematic Literature Review. Ekstrak informasi PICO dari paper berikut.

PICO Framework:
- Population: Siapa/apa yang diteliti (subjek, sampel, populasi target)
- Intervention: Intervensi/metode/treatment yang diterapkan
- Comparison: Pembanding (jika ada), kontrol, baseline
- Outcome: Hasil/temuan utama yang diukur

TEKS PAPER:
---
{full_text}
---

Respond HANYA dengan JSON valid berikut:
{{
  "title": "judul paper",
  "authors": "nama penulis",
  "year": "tahun publikasi atau N/A",
  "population": "deskripsi populasi/sampel",
  "intervention": "deskripsi intervensi/metode",
  "comparison": "deskripsi pembanding atau N/A jika tidak ada",
  "outcome": "hasil utama penelitian"
}}"""


def extract_full_text(file_path: str) -> str:
    """Extract full text from PDF."""
    try:
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"Error reading PDF: {e}"


def extract_pico(file_path: str, llm) -> dict:
    """Extract PICO data from a single paper."""
    full_text = extract_full_text(file_path)
    
    if not full_text or full_text.startswith("Error"):
        return {
            "title": Path(file_path).name,
            "authors": "N/A",
            "year": "N/A",
            "population": "Error membaca PDF",
            "intervention": "N/A",
            "comparison": "N/A",
            "outcome": "N/A",
        }
    
    # Limit text to avoid token overflow
    prompt = _PICO_PROMPT.format(full_text=full_text[:15000])
    
    try:
        response = llm.complete(prompt)
        resp_text = getattr(response, "text", None) or str(response)
        data = _extract_json_from_response(resp_text)
        
        if data:
            return {
                "title": data.get("title", Path(file_path).name),
                "authors": data.get("authors", "N/A"),
                "year": data.get("year", "N/A"),
                "population": data.get("population", "N/A"),
                "intervention": data.get("intervention", "N/A"),
                "comparison": data.get("comparison", "N/A"),
                "outcome": data.get("outcome", "N/A"),
            }
        else:
            return {
                "title": Path(file_path).name,
                "authors": "N/A",
                "year": "N/A",
                "population": "Gagal parse respons",
                "intervention": "N/A",
                "comparison": "N/A",
                "outcome": "N/A",
            }
    except Exception as e:
        return {
            "title": Path(file_path).name,
            "authors": "N/A",
            "year": "N/A",
            "population": f"Error: {e}",
            "intervention": "N/A",
            "comparison": "N/A",
            "outcome": "N/A",
        }


def run_pico_extraction(included_papers: list, provider: str, api_key: str, base_url: str) -> list[dict]:
    """Run PICO extraction on included papers."""
    try:
        llm, _ = _get_llm_and_embed(provider, api_key, base_url)
    except ValueError as e:
        return [{"title": "Error", "population": str(e)}]
    
    results = []
    for paper in included_papers:
        result = extract_pico(paper["path"], llm)
        result["filename"] = paper["filename"]
        results.append(result)
    
    return results


# --- Step 4: Report & Bibliography ---
_DRAFT_PROMPT = """Bertindaklah sebagai peneliti akademis. Tulis draf literature review berdasarkan data PICO dari paper-paper berikut.

DATA PICO (dari {num_papers} paper):
{pico_summary}

TOPIK: {topic}

Gunakan bahasa Indonesia baku yang akademis.
Sertakan sitasi (Author, Year) setiap kali mengambil fakta dari paper.
Panjang tulisan sekitar {target_words} kata.
{custom_instructions}

Tulis draf yang lengkap, terstruktur, dan mengikuti standar penulisan akademis."""


def generate_draft_from_pico(
    pico_data: list,
    topic: str,
    target_words: int,
    custom_instructions: str,
    provider: str,
    api_key: str,
    base_url: str,
) -> tuple[str, str]:
    """Generate draft from PICO data."""
    if not pico_data:
        return "", "Tidak ada data PICO untuk ditulis."
    
    try:
        llm, _ = _get_llm_and_embed(provider, api_key, base_url)
    except ValueError as e:
        return "", str(e)
    
    # Create PICO summary
    pico_summary = ""
    for i, p in enumerate(pico_data, 1):
        pico_summary += f"""
Paper {i}: {p.get('title', 'N/A')} ({p.get('authors', 'N/A')}, {p.get('year', 'N/A')})
- Population: {p.get('population', 'N/A')}
- Intervention: {p.get('intervention', 'N/A')}
- Comparison: {p.get('comparison', 'N/A')}
- Outcome: {p.get('outcome', 'N/A')}
"""
    
    prompt = _DRAFT_PROMPT.format(
        num_papers=len(pico_data),
        pico_summary=pico_summary,
        topic=topic,
        target_words=target_words,
        custom_instructions=f"Instruksi khusus: {custom_instructions}" if custom_instructions else "",
    )
    
    try:
        response = llm.complete(prompt)
        draft_text = getattr(response, "text", None) or str(response)
        return draft_text, ""
    except Exception as e:
        return "", f"Error: {e}"


def generate_bibliography_apa7(pico_data: list) -> str:
    """Generate APA 7th style bibliography from PICO data."""
    if not pico_data:
        return ""
    
    entries = []
    for p in pico_data:
        authors = p.get("authors", "Unknown Author")
        year = p.get("year", "n.d.")
        title = p.get("title", "Untitled")
        
        # Format: Author(s). (Year). Title.
        if year != "N/A" and year:
            entry = f"{authors}. ({year}). *{title}*."
        else:
            entry = f"{authors}. (n.d.). *{title}*."
        
        entries.append(entry)
    
    # Sort alphabetically by author
    entries.sort()
    
    return "\n\n".join(entries)


# --- UI Components ---
def render_sidebar():
    """Render sidebar configuration and return provider settings."""
    with st.sidebar:
        st.header("⚙️ Konfigurasi AI")
        st.divider()

        provider_label = st.selectbox(
            label="Pilih Provider AI",
            options=list(PROVIDERS.values()),
            index=1,
            help="Pilih model LLM untuk proses SLR",
        )
        provider = next(k for k, v in PROVIDERS.items() if v == provider_label)

        api_key = ""
        base_url = "http://localhost:11434"

        if provider == "openai":
            api_key_input = st.text_input(
                label="Masukkan OpenAI API Key",
                type="password",
                placeholder="sk-...",
            )
            api_key = (api_key_input or "").strip() or os.getenv("OPENAI_API_KEY", "")

        elif provider == "gemini":
            api_key_input = st.text_input(
                label="Masukkan Google API Key",
                type="password",
                placeholder="AIza...",
            )
            api_key = (api_key_input or "").strip() or os.getenv("GOOGLE_API_KEY", "")

        elif provider == "anthropic":
            api_key_input = st.text_input(
                label="Masukkan Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
            )
            api_key = (api_key_input or "").strip() or os.getenv("ANTHROPIC_API_KEY", "")

        elif provider == "ollama":
            base_url_input = st.text_input(
                label="Ollama Base URL",
                value="http://localhost:11434",
            )
            base_url = (base_url_input or "http://localhost:11434").strip().rstrip("/")

        # Credential status
        if provider == "ollama":
            has_creds = bool(base_url)
            st.success("Ollama Ready ✓") if has_creds else st.info("Isi Base URL Ollama.")
        else:
            has_creds = bool(api_key)
            st.success("API Key ✓") if has_creds else st.warning("Masukkan API Key.")

        st.divider()
        st.caption(f"📁 Temp folder: `{TEMP_PAPERS_DIR}`")
        
        # Show current step in sidebar
        current_step = st.session_state.get("current_step", 1)
        st.markdown(f"### 📍 Step {current_step} / 4")

    return provider, api_key, base_url, has_creds


def render_progress_bar(current_step: int):
    """Render step progress indicator."""
    steps = ["Upload & Criteria", "AI Screening", "PICO Extraction", "Report & Bibliography"]
    
    cols = st.columns(4)
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current_step:
                st.success(f"✅ Step {i}")
            elif i == current_step:
                st.info(f"🔵 Step {i}")
            else:
                st.markdown(f"⚪ Step {i}")
            st.caption(step_name)


def render_step_1(provider: str, api_key: str, base_url: str, has_creds: bool):
    """Step 1: Upload & Criteria"""
    st.header("📤 Step 1: Upload & Kriteria Inklusi")
    st.markdown("Upload file PDF paper Anda dan tentukan kriteria inklusi untuk screening.")
    
    st.divider()
    
    # File uploader
    uploaded_files = st.file_uploader(
        label="Upload Paper PDF (Multiple)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload hingga 100 file PDF sekaligus",
    )
    
    if uploaded_files:
        st.success(f"📎 {len(uploaded_files)} file dipilih")
        with st.expander("Lihat daftar file", expanded=False):
            for f in uploaded_files:
                st.caption(f"• {f.name} ({round(len(f.getvalue())/1024, 1)} KB)")
    
    st.divider()
    
    # Inclusion criteria
    criteria = st.text_area(
        label="Kriteria Inklusi",
        placeholder="Contoh: Paper harus membahas Turbin Savonius dan menggunakan metode CFD. Paper harus dalam bahasa Inggris atau Indonesia.",
        height=120,
        help="Jelaskan kriteria apa saja yang harus dipenuhi agar paper dimasukkan dalam review",
    )
    
    st.divider()
    
    # Start screening button
    col1, col2 = st.columns([3, 1])
    with col1:
        start_btn = st.button(
            "🚀 Mulai Screening",
            type="primary",
            use_container_width=True,
            disabled=not (uploaded_files and criteria and has_creds),
        )
    
    if not has_creds:
        st.warning("⚠️ Lengkapi konfigurasi API di sidebar sebelum melanjutkan.")
    elif not uploaded_files:
        st.info("👆 Upload file PDF untuk memulai.")
    elif not criteria:
        st.info("✏️ Isi kriteria inklusi untuk proses screening.")
    
    if start_btn and uploaded_files and criteria and has_creds:
        with st.spinner("Menyimpan file ke folder lokal..."):
            saved_files = save_uploaded_files(uploaded_files)
            st.session_state.uploaded_files_info = saved_files
            st.session_state.inclusion_criteria = criteria
        
        st.success(f"✅ {len(saved_files)} file disimpan ke `{TEMP_PAPERS_DIR}`")
        
        with st.spinner(f"Menjalankan AI Screening pada {len(saved_files)} paper..."):
            screening_results = run_screening(
                saved_files, criteria, provider, api_key, base_url
            )
            st.session_state.screening_results = screening_results
        
        st.session_state.current_step = 2
        st.rerun()


def render_step_2(provider: str, api_key: str, base_url: str, has_creds: bool):
    """Step 2: AI Screening Results"""
    st.header("🔍 Step 2: Hasil AI Screening")
    st.markdown("Review hasil screening AI. Anda dapat mengubah status INCLUDE/EXCLUDE secara manual.")
    
    # Back button
    if st.button("⬅️ Kembali ke Step 1"):
        st.session_state.current_step = 1
        st.rerun()
    
    st.divider()
    
    screening_results = st.session_state.get("screening_results", [])
    
    if not screening_results:
        st.warning("Tidak ada hasil screening. Kembali ke Step 1.")
        return
    
    # Display criteria
    criteria = st.session_state.get("inclusion_criteria", "")
    with st.expander("📋 Kriteria Inklusi yang Digunakan", expanded=False):
        st.info(criteria)
    
    # Summary stats
    include_count = sum(1 for r in screening_results if r.get("status") == "INCLUDE")
    exclude_count = len(screening_results) - include_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Paper", len(screening_results))
    col2.metric("INCLUDE", include_count, delta=None)
    col3.metric("EXCLUDE", exclude_count, delta=None)
    
    st.divider()
    
    # Editable table
    df = pd.DataFrame(screening_results)
    df = df[["filename", "title", "reason", "status"]]
    df.columns = ["File", "Judul", "Alasan", "Status"]
    
    edited_df = st.data_editor(
        df,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["INCLUDE", "EXCLUDE"],
                required=True,
            ),
            "Alasan": st.column_config.TextColumn("Alasan", width="large"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )
    
    # Update session state with edited values
    for i, row in edited_df.iterrows():
        screening_results[i]["status"] = row["Status"]
    st.session_state.screening_results = screening_results
    
    st.divider()
    
    # Count included after edits
    include_count = sum(1 for r in screening_results if r.get("status") == "INCLUDE")
    
    if include_count == 0:
        st.warning("⚠️ Tidak ada paper dengan status INCLUDE. Ubah status minimal 1 paper untuk melanjutkan.")
    
    # Next button
    next_btn = st.button(
        f"➡️ Lanjut ke Ekstraksi Data ({include_count} paper)",
        type="primary",
        use_container_width=True,
        disabled=include_count == 0,
    )
    
    if next_btn and include_count > 0:
        st.session_state.current_step = 3
        st.rerun()


def render_step_3(provider: str, api_key: str, base_url: str, has_creds: bool):
    """Step 3: PICO Extraction"""
    st.header("📊 Step 3: Ekstraksi Data PICO")
    st.markdown("Ekstraksi mendalam (full text) untuk menghasilkan Matriks PICO.")
    
    # Back button
    if st.button("⬅️ Kembali ke Step 2"):
        st.session_state.current_step = 2
        st.rerun()
    
    st.divider()
    
    # Get included papers
    screening_results = st.session_state.get("screening_results", [])
    included_papers = [r for r in screening_results if r.get("status") == "INCLUDE"]
    
    if not included_papers:
        st.warning("Tidak ada paper INCLUDE. Kembali ke Step 2.")
        return
    
    st.info(f"📚 {len(included_papers)} paper akan diekstraksi")
    
    # Check if PICO data already exists
    pico_data = st.session_state.get("pico_data", [])
    
    if not pico_data:
        extract_btn = st.button(
            "🔬 Mulai Ekstraksi PICO",
            type="primary",
            use_container_width=True,
        )
        
        if extract_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, paper in enumerate(included_papers):
                status_text.text(f"Mengekstrak: {paper['filename']} ({i+1}/{len(included_papers)})")
                progress_bar.progress((i + 1) / len(included_papers))
                
                result = extract_pico(paper["path"], _get_llm_and_embed(provider, api_key, base_url)[0])
                result["filename"] = paper["filename"]
                results.append(result)
            
            st.session_state.pico_data = results
            status_text.empty()
            progress_bar.empty()
            st.rerun()
    else:
        st.success(f"✅ Data PICO tersedia untuk {len(pico_data)} paper")
        
        # Display PICO table
        df = pd.DataFrame(pico_data)
        display_cols = ["title", "authors", "year", "population", "intervention", "comparison", "outcome"]
        df = df[[c for c in display_cols if c in df.columns]]
        df.columns = ["Judul", "Penulis", "Tahun", "Population", "Intervention", "Comparison", "Outcome"]
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Download PICO Matrix (CSV)",
            data=csv,
            file_name="pico_matrix.csv",
            mime="text/csv",
        )
        
        st.divider()
        
        # Re-extract button
        if st.button("🔄 Ekstraksi Ulang"):
            st.session_state.pop("pico_data", None)
            st.rerun()
        
        # Next button
        next_btn = st.button(
            "➡️ Lanjut ke Penulisan",
            type="primary",
            use_container_width=True,
        )
        
        if next_btn:
            st.session_state.current_step = 4
            st.rerun()


def render_step_4(provider: str, api_key: str, base_url: str, has_creds: bool):
    """Step 4: Report & Bibliography"""
    st.header("📝 Step 4: Penulisan & Daftar Pustaka")
    st.markdown("Generate draft literature review dan daftar pustaka APA 7th.")
    
    # Back button
    if st.button("⬅️ Kembali ke Step 3"):
        st.session_state.current_step = 3
        st.rerun()
    
    st.divider()
    
    pico_data = st.session_state.get("pico_data", [])
    
    if not pico_data:
        st.warning("Data PICO belum tersedia. Kembali ke Step 3.")
        return
    
    # Draft generation section
    st.subheader("📄 Generate Draft")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input(
            label="Topik / Judul Literature Review",
            placeholder="Contoh: Analisis Performa Turbin Savonius dengan Metode CFD",
        )
    with col2:
        target_words = st.slider(
            label="Target Jumlah Kata",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
        )
    
    custom_instructions = st.text_area(
        label="Instruksi Khusus (Opsional)",
        placeholder="Contoh: Fokus pada perbandingan hasil antar paper",
        height=80,
    )
    
    draft_btn = st.button(
        "✍️ Generate Draft",
        type="primary",
        use_container_width=True,
        disabled=not topic,
    )
    
    if draft_btn and topic:
        with st.spinner("Menulis draft..."):
            draft_text, err = generate_draft_from_pico(
                pico_data, topic, target_words, custom_instructions,
                provider, api_key, base_url
            )
        
        if err:
            st.error(err)
        else:
            st.session_state.draft_text = draft_text
            st.session_state.draft_topic = topic
    
    # Display draft
    draft_text = st.session_state.get("draft_text", "")
    if draft_text:
        st.divider()
        st.markdown("### 📝 Draft yang Dihasilkan")
        st.markdown(draft_text)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Draft (TXT)",
                data=draft_text.encode("utf-8"),
                file_name=f"{st.session_state.get('draft_topic', 'draft').replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            md_content = f"# {st.session_state.get('draft_topic', 'Draft')}\n\n{draft_text}"
            st.download_button(
                label="📥 Download Draft (MD)",
                data=md_content.encode("utf-8"),
                file_name=f"{st.session_state.get('draft_topic', 'draft').replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    
    st.divider()
    
    # Bibliography section
    st.subheader("📚 Daftar Pustaka (APA 7th)")
    
    bib_btn = st.button(
        "📚 Generate Bibliography",
        use_container_width=True,
    )
    
    if bib_btn:
        bibliography = generate_bibliography_apa7(pico_data)
        st.session_state.bibliography = bibliography
    
    bibliography = st.session_state.get("bibliography", "")
    if bibliography:
        st.markdown("---")
        st.markdown("### References")
        st.markdown(bibliography)
        
        st.download_button(
            label="📥 Download Bibliography",
            data=bibliography.encode("utf-8"),
            file_name="bibliography_apa7.txt",
            mime="text/plain",
            use_container_width=True,
        )
    
    st.divider()
    
    # Reset workflow
    st.subheader("🔄 Mulai Ulang")
    if st.button("🗑️ Reset Semua & Mulai Review Baru", type="secondary"):
        # Clear all session state
        for key in ["current_step", "uploaded_files_info", "inclusion_criteria",
                    "screening_results", "pico_data", "draft_text", "draft_topic", "bibliography"]:
            st.session_state.pop(key, None)
        st.rerun()


# --- Main Application ---
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="ScholarFlow SLR",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure temp directory exists
    TEMP_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize session state
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    
    # Render sidebar and get config
    provider, api_key, base_url, has_creds = render_sidebar()
    
    # Main content
    st.title("📚 ScholarFlow - Systematic Literature Review")
    st.caption("Workflow bertahap untuk review literatur volume besar (hingga 100 paper)")
    
    # Progress bar
    render_progress_bar(st.session_state.current_step)
    
    st.divider()
    
    # Render current step
    current_step = st.session_state.current_step
    
    if current_step == 1:
        render_step_1(provider, api_key, base_url, has_creds)
    elif current_step == 2:
        render_step_2(provider, api_key, base_url, has_creds)
    elif current_step == 3:
        render_step_3(provider, api_key, base_url, has_creds)
    elif current_step == 4:
        render_step_4(provider, api_key, base_url, has_creds)


if __name__ == "__main__":
    main()


