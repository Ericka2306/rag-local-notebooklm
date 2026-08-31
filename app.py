"""
TP — Système RAG Local (Clone de NotebookLM)
=============================================
ÉTAPE 1 : interface graphique Streamlit (design inspiré de NotebookLM :
sidebar "Sources" à gauche, chat au centre, sources citées sous les réponses).
Icônes : Material Symbols (Google) — celles du vrai NotebookLM.

⚠️ Le backend (Étapes 2, 3, 4) n'est PAS encore implémenté : les trois
fonctions de la section 🚧 renvoient des données factices pour que
l'interface soit testable immédiatement.

Lancement :  streamlit run app.py
"""

import html
import time

import streamlit as st

# =============================================================================
# CONFIGURATION DE LA PAGE (doit être le tout premier appel Streamlit)
# =============================================================================
st.set_page_config(
    page_title="NotebookLM Local",
    page_icon=":material/auto_stories:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLE — polices Inter + Material Symbols, cartes arrondies façon NotebookLM
# =============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..600,0..1,-50..200&display=swap');

    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    /* IMPORTANT : la règle ci-dessus écrase la police de TOUS les éléments
       Streamlit, y compris les icônes Material natives (qui sont des
       ligatures : sans leur police, on voit "bolt" en texte). On la
       restaure ici, en priorité absolue. */
    [data-testid="stIconMaterial"], [class*="material-symbols"], .msr {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Icône Material utilisable dans notre HTML : <span class="msr">nom</span> */
    .msr {
        font-weight: normal; font-style: normal; line-height: 1;
        display: inline-block; vertical-align: -3px; font-size: 17px;
    }

    /* En-tête Streamlit transparent (on garde le bouton de la sidebar) */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* ---------- En-tête de l'application ---------- */
    .app-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.1rem;
    }
    .app-title {
        font-size: 1.7rem; font-weight: 700; color: #202124;
        display: flex; align-items: center; gap: 10px;
    }
    .app-title .msr { font-size: 30px; color: #1A73E8; vertical-align: -6px; }
    .app-sub { color: #5F6368; font-size: 0.88rem; margin-bottom: 1.2rem; }

    /* Pastille du mode actif */
    .pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; white-space: nowrap;
    }
    .pill .msr { font-size: 15px; }
    .pill-rag    { background: #E8F0FE; color: #1A73E8; }
    .pill-search { background: #E6F4EA; color: #188038; }

    /* ---------- Sidebar ---------- */
    .side-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #5F6368; margin: 0.4rem 0 0.5rem 0;
    }
    .file-card {
        display: flex; align-items: center; gap: 8px;
        background: #FFFFFF; border: 1px solid #E8EAED; border-radius: 10px;
        padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem; color: #202124;
    }
    .file-card .file-size { color: #5F6368; margin-left: auto; font-size: 0.78rem; }
    .stats-row { display: flex; gap: 8px; margin-top: 10px; }
    .stat-chip {
        flex: 1; display: flex; align-items: center; gap: 6px; justify-content: center;
        background: #FFFFFF; border: 1px solid #E8EAED; border-radius: 10px;
        padding: 7px 10px; font-size: 0.8rem; font-weight: 600; color: #202124;
    }
    .stat-chip .msr { color: #1A73E8; }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E8EAED;
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.4rem;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.06);
    }
    /* Messages de l'utilisateur légèrement teintés (si le navigateur sait :has) */
    [data-testid="stChatMessage"]:has([data-testid*="User"]),
    [data-testid="stChatMessage"]:has([data-testid*="user"]) {
        background: #F8F9FA;
    }

    /* Carte d'un extrait (chunk) retourné par la recherche */
    .chunk-card {
        background: #F8F9FA; border-left: 3px solid #1A73E8;
        border-radius: 8px; padding: 10px 14px; margin: 8px 0;
        font-size: 0.9rem; color: #202124;
    }
    .chunk-source {
        display: inline-flex; align-items: center; gap: 5px;
        color: #1A73E8; font-weight: 600; font-size: 0.78rem; margin-bottom: 4px;
    }

    /* ---------- Écran d'accueil ---------- */
    .hero { text-align: center; padding: 2.2rem 1rem 0.5rem 1rem; }
    .hero-badge {
        width: 84px; height: 84px; margin: 0 auto 1rem auto; border-radius: 50%;
        background: #E8F0FE; display: flex; align-items: center; justify-content: center;
    }
    .hero-badge .msr { font-size: 42px; color: #1A73E8; }
    .hero h2 { color: #202124; margin: 0 0 0.3rem 0; }
    .hero > p { color: #5F6368; margin-bottom: 1.6rem; }
    .steps { display: flex; gap: 10px; }
    .step {
        flex: 1; background: #FFFFFF; border: 1px solid #E8EAED;
        border-radius: 14px; padding: 14px 12px; text-align: center;
    }
    .step .msr { font-size: 26px; color: #1A73E8; }
    .step b { display: block; margin: 6px 0 2px 0; font-size: 0.86rem; color: #202124; }
    .step p { font-size: 0.78rem; color: #5F6368; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 🚧🚧🚧 BACKEND — À TOI DE JOUER (Étapes 2, 3 et 4 du sujet) 🚧🚧🚧
# =============================================================================
# Les trois fonctions ci-dessous sont des MAQUETTES : elles renvoient des
# données factices pour que l'interface soit testable dès maintenant.
# Ton travail : les remplacer par le vrai pipeline RAG (LangChain, ChromaDB,
# sentence-transformers, Ollama). L'interface, elle, n'aura pas à changer.
# =============================================================================

def index_documents(uploaded_files):
    """🚧 ÉTAPE 2 — Pipeline d'ingestion.

    À implémenter : extraction du texte (PyMuPDFLoader / TextLoader),
    découpage en chunks (RecursiveCharacterTextSplitter), vectorisation
    (HuggingFaceEmbeddings) et stockage dans ChromaDB, en conservant le nom
    du fichier dans les métadonnées. La base doit être gardée dans
    st.session_state pour survivre aux ré-exécutions du script.

    Doit retourner : le nombre de chunks indexés (int).
    """
    time.sleep(1.5)                      # simule le travail d'indexation
    return 42                            # ← nombre de chunks factice


def semantic_search(query):
    """🚧 ÉTAPE 3 — Recherche sémantique pure (AUCUN appel au LLM).

    À implémenter : interroger la base vectorielle (similarity_search) et
    retourner les k chunks les plus proches de la question.

    Doit retourner : une liste de dicts {"content": str, "source": str}.
    """
    return [                              # ← extraits factices
        {"content": f"Extrait factice n°{i} en lien avec « {query} ». "
                    "Remplace-moi par un vrai similarity_search !",
         "source": f"document_{i}.pdf"}
        for i in (1, 2, 3)
    ]


def rag_answer(query, chunks):
    """🚧 ÉTAPE 4 — Génération RAG (prompt strict + LLM local via Ollama).

    À implémenter : construire un PromptTemplate injectant {context} (les
    chunks) et {question}, puis appeler le modèle Ollama en streaming.

    Doit retourner : un GÉNÉRATEUR de morceaux de texte (pour l'affichage
    progressif avec st.write_stream).
    """
    fake = (f"Réponse factice à « {query} » générée à partir de "
            f"{len(chunks)} extraits. Branche-moi sur Ollama !")
    for word in fake.split(" "):          # simule le streaming token par token
        yield word + " "
        time.sleep(0.05)


# =============================================================================
# ÉTAT DE SESSION
# =============================================================================
# Streamlit ré-exécute tout le script à chaque interaction : ce qui doit
# survivre (historique, état d'indexation) vit dans st.session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []        # [{role, content, chunks?}, …]
if "indexed" not in st.session_state:
    st.session_state.indexed = False      # une indexation a-t-elle eu lieu ?
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0
if "source_names" not in st.session_state:
    st.session_state.source_names = []    # noms des fichiers indexés

# Icône Material + couleur selon le type de fichier (cartes de la sidebar).
FILE_ICONS = {
    "pdf": ("picture_as_pdf", "#D93025"),   # rouge Google
    "md":  ("markdown",       "#188038"),   # vert Google
    "txt": ("description",    "#5F6368"),   # gris
}

# Avatars du chat (icônes Material rendues par Streamlit).
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}


def render_chunks(chunks):
    """Affiche une liste d'extraits sous forme de cartes (source + contenu)."""
    for chunk in chunks:
        st.markdown(
            f"""<div class="chunk-card">
                <span class="chunk-source"><span class="msr">draft</span>
                {html.escape(chunk["source"])}</span><br>
                {html.escape(chunk["content"])}
            </div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# BARRE LATÉRALE — Sources, indexation, mode (façon panneau NotebookLM)
# =============================================================================
with st.sidebar:
    st.markdown('<div class="side-label">Sources</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Glissez vos fichiers ici",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Aperçu des fichiers sélectionnés (icône typée + nom + taille).
    for f in uploaded_files or []:
        icon, color = FILE_ICONS.get(f.name.rsplit(".", 1)[-1].lower(),
                                     FILE_ICONS["txt"])
        size_kb = len(f.getvalue()) / 1024
        st.markdown(
            f"""<div class="file-card">
                <span class="msr" style="color:{color}">{icon}</span>
                {html.escape(f.name)}
                <span class="file-size">{size_kb:,.0f} Ko</span>
            </div>""",
            unsafe_allow_html=True,
        )

    if st.button("Indexer les documents", icon=":material/bolt:",
                 type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        with st.spinner("Extraction → chunking → embeddings…"):
            st.session_state.n_chunks = index_documents(uploaded_files)
        st.session_state.indexed = True
        st.session_state.source_names = [f.name for f in uploaded_files]
        st.toast(f"{len(uploaded_files)} fichier(s) → "
                 f"{st.session_state.n_chunks} chunks indexés",
                 icon=":material/check_circle:")

    # Petit tableau de bord de l'index.
    if st.session_state.indexed:
        st.markdown(
            f"""<div class="stats-row">
                <div class="stat-chip"><span class="msr">folder</span>
                    {len(st.session_state.source_names)} source(s)</div>
                <div class="stat-chip"><span class="msr">grid_view</span>
                    {st.session_state.n_chunks} chunks</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="side-label">Assistant</div>', unsafe_allow_html=True)

    # Le fameux toggle du sujet : bascule entre les deux modes.
    llm_enabled = st.toggle("Générer avec le LLM", value=False)
    if llm_enabled:
        st.caption("Mode **RAG complet** : réponse rédigée par mistral "
                   "(Ollama, 100 % local) à partir de vos documents.")
    else:
        st.caption("Mode **Recherche sémantique** : extraits bruts de la "
                   "base vectorielle, aucun LLM.")


# =============================================================================
# ZONE PRINCIPALE — En-tête, historique de chat, saisie
# =============================================================================

# En-tête : titre + pastille indiquant le mode actif.
pill = (
    '<span class="pill pill-rag"><span class="msr">auto_awesome</span>'
    ' RAG complet</span>'
    if llm_enabled else
    '<span class="pill pill-search"><span class="msr">search</span>'
    ' Recherche sémantique</span>'
)
st.markdown(
    f"""<div class="app-header">
        <div class="app-title"><span class="msr">auto_stories</span>
        NotebookLM Local</div>{pill}
    </div>
    <div class="app-sub">Vos documents, vos réponses — rien ne quitte
    votre machine.</div>""",
    unsafe_allow_html=True,
)

# Écran d'accueil tant que rien n'est indexé et que le chat est vide.
if not st.session_state.indexed and not st.session_state.messages:
    st.markdown(
        """<div class="hero">
             <div class="hero-badge"><span class="msr">auto_stories</span></div>
             <h2>Discutez avec vos documents</h2>
             <p>Un assistant documentaire 100&nbsp;% local, sans API externe.</p>
             <div class="steps">
               <div class="step"><span class="msr">upload_file</span>
                 <b>1 · Ajouter</b><p>PDF, Markdown ou TXT dans la barre latérale</p></div>
               <div class="step"><span class="msr">bolt</span>
                 <b>2 · Indexer</b><p>Extraction, chunking et vectorisation</p></div>
               <div class="step"><span class="msr">forum</span>
                 <b>3 · Discuter</b><p>Recherche sémantique ou réponse générée</p></div>
             </div>
           </div>""",
        unsafe_allow_html=True,
    )

# Ré-affichage de l'historique complet à chaque exécution du script.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        # Réponses RAG : les sources restent consultables dans un expander.
        if msg.get("chunks"):
            with st.expander("Sources utilisées", icon=":material/format_quote:"):
                render_chunks(msg["chunks"])

# Saisie utilisateur — désactivée tant qu'aucun document n'est indexé
# (même comportement que NotebookLM : pas de source, pas de question).
query = st.chat_input(
    "Posez une question sur vos documents…" if st.session_state.indexed
    else "Indexez d'abord des documents dans la barre latérale",
    disabled=not st.session_state.indexed,
)

if query:
    # 1) La question de l'utilisateur.
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 2) La réponse, selon le mode choisi via le toggle.
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        if not llm_enabled:
            # ---- Mode Recherche Sémantique (Étape 3) : extraits bruts ----
            chunks = semantic_search(query)
            st.markdown("**Extraits les plus proches de votre question :**")
            render_chunks(chunks)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "**Extraits les plus proches de votre question :**\n\n"
                           + "\n\n".join(f"> {c['content']}\n> — *{c['source']}*"
                                         for c in chunks),
            })
        else:
            # ---- Mode RAG complet (Étape 4) : réponse + sources ----------
            chunks = semantic_search(query)                  # même retrieval
            answer = st.write_stream(rag_answer(query, chunks))
            with st.expander("Sources utilisées", icon=":material/format_quote:"):
                render_chunks(chunks)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "chunks": chunks}
            )
