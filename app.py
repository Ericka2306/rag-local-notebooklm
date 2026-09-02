"""
TP — Système RAG Local (Clone de NotebookLM)
=============================================
Point d'entrée : le SCRIPT DE PAGE Streamlit (Étape 1). Il décrit le
déroulé de l'interface et gère l'état de session, rien d'autre :
  - les composants d'affichage vivent dans ui.py (+ styles.css) ;
  - le pipeline RAG vit dans rag/ (un module par étape du TP).

Lancement :  streamlit run app.py
"""

from collections import Counter

import streamlit as st

import auth
import chat_history
import ui
from rag.config import MIN_RELEVANCE
from rag.ingestion import (load_documents, split_documents, build_vectorstore,
                           load_vectorstore, clear_index)
from rag.retrieval import semantic_search
from rag.generation import rag_answer

# =============================================================================
# PAGE & STYLE
# =============================================================================
st.set_page_config(
    page_title="NotebookLM Local",
    page_icon=":material/auto_stories:",
    layout="centered",
    initial_sidebar_state="expanded",
)
ui.load_css()

# =============================================================================
# AUTHENTIFICATION — rien ne s'affiche ni ne se charge sans session valide
# =============================================================================
authenticator, user = auth.require_login()

# =============================================================================
# ÉTAT DE SESSION (cloisonné par utilisateur)
# =============================================================================
# Streamlit ré-exécute tout le script à chaque interaction : ce qui doit
# survivre vit dans st.session_state — géré ICI, côté interface ; le
# backend rag/ n'y touche jamais.
# Les données durables (index, historique) sont persistées sur disque PAR
# UTILISATEUR ; on les (re)charge quand l'utilisateur de la session change
# (connexion, ou déconnexion/reconnexion sous un autre compte).
if st.session_state.get("data_user") != user:
    st.session_state.messages = chat_history.load(user)
    vs, sources = load_vectorstore(user)
    st.session_state.vectorstore = vs
    st.session_state.sources = sources
    st.session_state.raw_docs = []     # aperçus de débogage (session courante)
    st.session_state.chunks = []
    # Marqueur posé EN DERNIER : si le script est interrompu en plein
    # chargement (rerun, exception), le bloc se rejouera entièrement au
    # lieu de laisser un état à moitié initialisé.
    st.session_state.data_user = user
# Ceinture de sécurité : des valeurs par défaut pour chaque clé utilisée
# plus bas, quoi qu'il arrive.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("sources", [])
st.session_state.setdefault("raw_docs", [])
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("uploader_key", 0)    # sert à vider l'uploader

# Toast différé : l'indexation se termine par un st.rerun(), et un toast
# émis juste avant serait perdu — on le stocke, il s'affiche ici, après.
if toast_msg := st.session_state.pop("index_toast", None):
    st.toast(toast_msg, icon=":material/check_circle:")

indexed = bool(st.session_state.sources)

# =============================================================================
# BARRE LATÉRALE — sources, indexation, mode
# =============================================================================
with st.sidebar:
    ui.section_label("Ajouter des documents")
    # La key change après chaque indexation : Streamlit voit un nouveau
    # widget et repart vide — les fichiers indexés quittent la zone d'ajout
    # pour n'apparaître que dans "Sources indexées" (comme NotebookLM).
    uploaded_files = st.file_uploader(
        "Glissez vos fichiers ici",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader-{st.session_state.uploader_key}",
    )

    # Bouton d'indexation : enchaîne les trois sous-étapes du pipeline.
    if st.button("Indexer les documents", icon=":material/bolt:",
                 type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        with st.spinner("Extraction → chunking → embeddings…"):
            docs = load_documents(uploaded_files)                     # 2.1
            chunks = split_documents(docs)                            # 2.2
            st.session_state.vectorstore = build_vectorstore(chunks, user)
        # Bilan de l'index : nombre de chunks obtenus pour chaque fichier.
        per_source = Counter(c.metadata["source"] for c in chunks)
        st.session_state.sources = [
            {"name": f.name, "n_chunks": per_source.get(f.name, 0)}
            for f in uploaded_files
        ]
        st.session_state.raw_docs = docs
        st.session_state.chunks = chunks
        st.session_state.index_toast = (f"{len(uploaded_files)} fichier(s) → "
                                        f"{len(chunks)} chunks indexés")
        st.session_state.uploader_key += 1   # vide la zone d'ajout
        st.rerun()

    # La liste de ce qui est RÉELLEMENT dans la base vectorielle
    # (persiste même si la sélection de l'uploader change).
    if indexed:
        st.divider()
        ui.section_label("Sources indexées")
        for source in st.session_state.sources:
            ui.source_card(source["name"], source["n_chunks"])
        ui.stats_row(len(st.session_state.sources),
                     sum(s["n_chunks"] for s in st.session_state.sources))

        # Vider l'index proprement (via le client ChromaDB, jamais en
        # supprimant chroma_db/ à la main) puis repartir de zéro.
        if st.button("Vider l'index", icon=":material/delete:",
                     use_container_width=True):
            clear_index(st.session_state.vectorstore)
            st.session_state.vectorstore = None
            st.session_state.sources = []
            st.session_state.raw_docs = []
            st.session_state.chunks = []
            st.rerun()

    # 🔬 Aperçus de débogage du pipeline (Étapes 2.1 et 2.2).
    if st.session_state.raw_docs:
        st.divider()
        ui.section_label("Débogage du pipeline")
        ui.extraction_preview(st.session_state.raw_docs)
        ui.chunking_preview(st.session_state.chunks)

    st.divider()
    ui.section_label("Assistant")
    # Le fameux toggle du sujet : bascule entre les deux modes.
    llm_enabled = st.toggle("Générer avec le LLM", value=False)
    st.caption("Mode **RAG complet** : réponse rédigée par mistral (Ollama, "
               "100 % local) à partir de vos documents." if llm_enabled else
               "Mode **Recherche sémantique** : extraits bruts de la base "
               "vectorielle, aucun LLM.")

    # Repartir d'une conversation vierge (efface aussi la sauvegarde).
    if st.session_state.messages and st.button(
            "Nouvelle conversation", icon=":material/add_comment:",
            use_container_width=True):
        chat_history.clear(user)
        st.session_state.messages = []
        st.rerun()

    st.divider()
    ui.section_label("Compte")
    st.caption(f"Connecté·e : **{st.session_state.get('name', user)}**")
    # Déconnexion : streamlit-authenticator efface le cookie de session et
    # ses clés de session_state, puis relance le script -> retour au login.
    authenticator.logout("Se déconnecter", location="sidebar")

# =============================================================================
# ZONE PRINCIPALE — en-tête, historique, saisie
# =============================================================================
ui.header(llm_enabled)

if not indexed and not st.session_state.messages:
    ui.hero()

# Ré-affichage de l'historique complet à chaque exécution du script.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=ui.AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander("Sources utilisées"):
                ui.chunk_cards(msg["chunks"])

# Saisie — désactivée tant que rien n'est indexé (pas de source, pas de
# question, comme NotebookLM).
query = st.chat_input(
    "Posez une question sur vos documents…" if indexed
    else "Indexez d'abord des documents dans la barre latérale",
    disabled=not indexed,
)

if query:
    with st.chat_message("user", avatar=ui.AVATARS["user"]):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar=ui.AVATARS["assistant"]):
        if not llm_enabled:
            # ---- Mode Recherche Sémantique (Étape 3) : AUDIT --------------
            # Pas de seuil : on montre TOUT le top-k avec les scores, y
            # compris ce que le mode RAG écarterait — c'est le but du mode.
            chunks = semantic_search(st.session_state.vectorstore, query)
            st.markdown("**Extraits les plus proches de votre question :**")
            ui.chunk_cards(chunks)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "**Extraits les plus proches de votre question :**\n\n"
                           + "\n\n".join(
                               f"> {c['content']}\n> — *{c['source']}* "
                               f"(similarité {c['score']:.2f})"
                               for c in chunks),
            })
        else:
            # ---- Mode RAG complet (Étape 4) : réponse + sources ----------
            # Seuls les chunks au-dessus du seuil de pertinence entrent
            # dans le contexte du LLM (fini le top-4 systématique).
            chunks = semantic_search(st.session_state.vectorstore, query,
                                     min_score=MIN_RELEVANCE)
            if not chunks:
                # Rien d'assez pertinent : inutile d'appeler le LLM.
                answer = ("Aucun passage de vos documents n'est assez "
                          "proche de cette question (tous les scores sont "
                          "sous le seuil de pertinence). Reformulez, ou "
                          "vérifiez en mode recherche sémantique.")
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
            else:
                answer = st.write_stream(rag_answer(query, chunks))
                with st.expander("Sources utilisées"):
                    ui.chunk_cards(chunks)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "chunks": chunks}
                )

    # Sauvegarde de l'échange complet (question + réponse) sur disque —
    # un seul point de sauvegarde couvre tous les chemins ci-dessus.
    chat_history.save(st.session_state.messages, user)
