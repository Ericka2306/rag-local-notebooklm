"""
TP — Système RAG Local (Clone de NotebookLM)
=============================================
Point d'entrée : l'INTERFACE Streamlit (Étape 1), et rien d'autre.
Le pipeline RAG vit dans le package rag/ (un module par étape du TP) ;
le style vit dans styles.css.

Lancement :  streamlit run app.py
"""

import html
from pathlib import Path

import streamlit as st

from rag.ingestion import load_documents, split_documents, build_vectorstore
from rag.retrieval import semantic_search
from rag.generation import rag_answer

# =============================================================================
# CONFIGURATION DE LA PAGE (doit être le tout premier appel Streamlit)
# =============================================================================
st.set_page_config(
    page_title="NotebookLM Local",
    page_icon=":material/auto_stories:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Feuille de style externe (voir styles.css).
st.markdown(f"<style>{Path('styles.css').read_text()}</style>",
            unsafe_allow_html=True)

# =============================================================================
# ÉTAT DE SESSION
# =============================================================================
# Streamlit ré-exécute tout le script à chaque interaction : ce qui doit
# survivre (historique, index) vit dans st.session_state — géré ICI, côté
# interface ; le backend rag/ n'y touche jamais.
if "messages" not in st.session_state:
    st.session_state.messages = []        # [{role, content, chunks?}, …]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None   # base vectorielle (Étape 2.3)
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []        # Documents extraits (aperçu 2.1)
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

    # Bouton d'indexation : enchaîne les trois sous-étapes du pipeline.
    if st.button("Indexer les documents", icon=":material/bolt:",
                 type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        with st.spinner("Extraction → chunking → embeddings…"):
            docs = load_documents(uploaded_files)                  # 2.1 ✅
            chunks = split_documents(docs)                         # 2.2 🚧
            st.session_state.vectorstore = build_vectorstore(chunks)  # 2.3 🚧
        st.session_state.raw_docs = docs
        st.session_state.n_chunks = len(chunks)
        st.session_state.source_names = [f.name for f in uploaded_files]
        st.toast(f"{len(uploaded_files)} fichier(s) → "
                 f"{st.session_state.n_chunks} chunks indexés",
                 icon=":material/check_circle:")

    # Petit tableau de bord de l'index.
    if st.session_state.source_names:
        st.markdown(
            f"""<div class="stats-row">
                <div class="stat-chip"><span class="msr">folder</span>
                    {len(st.session_state.source_names)} source(s)</div>
                <div class="stat-chip"><span class="msr">grid_view</span>
                    {st.session_state.n_chunks} chunks</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # 🔬 Aperçu de débogage (Étape 2.1) : permet de VÉRIFIER ce que
    # l'extraction a produit avant de construire la suite du pipeline.
    if st.session_state.raw_docs:
        with st.expander("Aperçu de l'extraction", icon=":material/science:"):
            docs = st.session_state.raw_docs
            st.caption(f"{len(docs)} segment(s) extrait(s) — 1 par page de PDF")
            for doc in docs[:3]:
                page = doc.metadata.get("page")
                page_info = f" · page {page + 1}" if page is not None else ""
                st.markdown(f"**{doc.metadata['source']}**{page_info} "
                            f"· {len(doc.page_content)} caractères")
                st.text(doc.page_content[:250] + "…")
            if len(docs) > 3:
                st.caption(f"… et {len(docs) - 3} autre(s) segment(s)")

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
if not st.session_state.source_names and not st.session_state.messages:
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
indexed = bool(st.session_state.source_names)
query = st.chat_input(
    "Posez une question sur vos documents…" if indexed
    else "Indexez d'abord des documents dans la barre latérale",
    disabled=not indexed,
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
            chunks = semantic_search(st.session_state.vectorstore, query)
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
            chunks = semantic_search(st.session_state.vectorstore, query)
            answer = st.write_stream(rag_answer(query, chunks))
            with st.expander("Sources utilisées", icon=":material/format_quote:"):
                render_chunks(chunks)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "chunks": chunks}
            )
