import streamlit as st
import requests
import base64

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="EchoVerse — AI Audiobook Creator",
    layout="centered",
    page_icon="🎧",
)

st.title("🎧 EchoVerse — AI Audiobook Creator")
st.markdown(
    """
    Upload a **PDF / Image / Text file**, or paste text directly.  
    Optionally, let the LLM rewrite the narration style before converting it into an audiobook.  
    You can also select the **output audio language**.
    """
)

# ----------------------------
# Sidebar (Backend Config)
# ----------------------------
st.sidebar.header("⚙️ Configuration")
BACKEND_URL = st.sidebar.text_input(
    "Backend URL", "http://localhost:8000", help="URL of the FastAPI backend"
)

# Fetch available models dynamically
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
    if health_resp.status_code == 200:
        available_models = health_resp.json().get("available_models", ["granite"])
    else:
        available_models = ["granite"]
except Exception:
    available_models = ["granite"]

model_choice = st.sidebar.selectbox("Choose LLM model", available_models)

# ----------------------------
# Language mapping
# ----------------------------
LANGUAGE_MAP = {
    "English": "en",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Telugu": "te",
    "Kannada": "kn",
}

# ----------------------------
# Helper: Download link
# ----------------------------
def download_link(file_url, filename):
    """Generate a download link for an audio file."""
    try:
        response = requests.get(file_url, timeout=60)
        if response.status_code == 200:
            b64 = base64.b64encode(response.content).decode()
            href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">⬇️ Download {filename}</a>'
            return href
    except Exception as e:
        st.warning(f"Download link failed: {e}")
    return None


# ----------------------------
# Input Form
# ----------------------------
with st.form("input_form"):
    uploaded_file = st.file_uploader(
        "📂 Upload PDF / Image / Text file",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
    )
    raw_text = st.text_area("✍️ Or paste text here (overrides file)", height=200)
    title = st.text_input("📖 Audiobook Title (optional)")

    use_llm = st.checkbox("✨ Use LLM for narration style?", value=False)
    llm_system_prompt = st.text_area(
        "📝 LLM system prompt (optional)",
        height=100,
        placeholder="E.g., Narrate in a calm storytelling style...",
    )

    # 🔥 NEW: Language selection
    language_choice = st.selectbox(
        "🌐 Select Audiobook Language",
        list(LANGUAGE_MAP.keys()),
        index=0,  # default = English
    )

    submitted = st.form_submit_button("🚀 Generate Audiobook")


# ----------------------------
# Handle Submit
# ----------------------------
if submitted:
    if not uploaded_file and not raw_text.strip():
        st.warning("⚠️ Please upload a file or paste some text.")
    else:
        with st.spinner("🎙️ Generating audiobook..."):
            try:
                files = None
                if uploaded_file:
                    files = {
                        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                    }

                data = {
                    "title": title,
                    "llm_system_prompt": llm_system_prompt if use_llm else "",
                    "raw_text": raw_text,
                    "model_choice": model_choice,
                    "language": LANGUAGE_MAP[language_choice],  # ✅ send language code
                }

                response = requests.post(
                    f"{BACKEND_URL}/generate", data=data, files=files, timeout=180
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"✅ Audiobook generated! (Model: {result['model_choice']}, "
                        f"Language: {language_choice}, Status: {result['llm_status']})"
                    )

                    audio_url = f"{BACKEND_URL}/download/{result['id']}"
                    st.audio(audio_url, format="audio/mp3")

                    link = download_link(audio_url, result["filename"])
                    if link:
                        st.markdown(link, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.error(f"🚨 Request failed: {e}")


# ----------------------------
# History Section
# ----------------------------
st.markdown("---")
st.subheader("📜 Audiobook History")

refresh = st.button("🔄 Refresh History")

if refresh or True:  # always load at start
    try:
        resp = requests.get(f"{BACKEND_URL}/history", timeout=30)
        if resp.status_code == 200:
            history = resp.json()
            if history:
                for item in reversed(history):  # latest first
                    with st.expander(f"📖 {item['title']} — {item['created_at']}"):
                        st.markdown(
                            f"**Model:** {item['metadata'].get('model_choice','?')}  \n"
                            f"**Language:** {item['metadata'].get('language','?')}  \n"
                            f"**LLM Status:** {item['metadata'].get('llm_status','?')}"
                        )
                        st.audio(f"{BACKEND_URL}/download/{item['id']}", format="audio/mp3")
            else:
                st.info("ℹ️ No audiobooks generated yet.")
        else:
            st.warning("⚠️ Could not fetch history.")
    except Exception as e:
        st.warning(f"⚠️ History fetch failed: {e}")
