"""
ÉTAPE 1 — Interface Streamlit du système RAG local (clone de NotebookLM)
========================================================================
Application 100 % locale : l'utilisateur charge ses documents (PDF,
Markdown, TXT) et les interroge, sans aucun appel à une API externe.

Deux modes, activables par le toggle de la barre latérale :
  - toggle désactivé : recherche sémantique pure (aucun LLM), les extraits
    bruts sont affichés avec le nom de leur fichier source ;
  - toggle activé    : RAG complet, le LLM local rédige une réponse à partir
    des extraits, consultables sous la réponse.

Organisation du code :
  - app.py          : interface (ce fichier)
  - embedding.py    : Étape 2, ingestion dans la base vectorielle
  - fonction_rag.py : Étapes 3 et 4, recherche et génération

Lancement :  streamlit run app.py
"""

import html
import json
from collections import Counter
from pathlib import Path

import streamlit as st

from embedding import (CHUNKING_STRATEGIES, CHUNKING_STRATEGY, build_vectorstore,
                       clear_index, load_documents, load_vectorstore,
                       split_documents)
from fonction_rag import LLM_MODEL, MIN_RELEVANCE, rag_answer, semantic_search

# =============================================================================
# STYLE — palette inspirée de NotebookLM
# =============================================================================
# Polices : Inter (texte) et Material Symbols Rounded (icônes).
# Couleurs : blanc, gris #F1F3F4, bleu #1A73E8.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..600,0..1,-50..200&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* La règle ci-dessus s'applique aussi aux icônes Material de Streamlit. Or
   ces icônes sont des ligatures : sans leur police, on lit le nom de
   l'icône ("bolt") au lieu du pictogramme. On rétablit donc leur police. */
[data-testid="stIconMaterial"], [class*="material-symbols"], .msr {
    font-family: 'Material Symbols Rounded' !important;
}
.msr {
    font-weight: normal; font-style: normal; line-height: 1;
    display: inline-block; vertical-align: -3px; font-size: 17px;
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* En-tête de l'application */
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
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; white-space: nowrap;
}
.pill .msr { font-size: 15px; }
.pill-rag    { background: #E8F0FE; color: #1A73E8; }
.pill-search { background: #E6F4EA; color: #188038; }

/* Barre latérale */
.side-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #5F6368; margin: 0.4rem 0 0.5rem 0;
}
.file-card {
    display: flex; align-items: center; gap: 8px;
    background: #FFFFFF; border: 1px solid #E8EAED; border-radius: 10px;
    padding: 8px 12px; margin-bottom: 6px; font-size: 0.84rem; color: #202124;
}
.file-card .file-name {
    flex: 1; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.file-card .file-badge {
    margin-left: auto; background: #E8F0FE; color: #1A73E8;
    border-radius: 999px; padding: 2px 9px; font-size: 0.72rem;
    font-weight: 600; white-space: nowrap;
}
.stats-row { display: flex; gap: 8px; margin-top: 10px; }
.stat-chip {
    flex: 1; display: flex; align-items: center; gap: 6px; justify-content: center;
    background: #FFFFFF; border: 1px solid #E8EAED; border-radius: 10px;
    padding: 7px 10px; font-size: 0.8rem; font-weight: 600; color: #202124;
}
.stat-chip .msr { color: #1A73E8; }

/* Messages du chat */
[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid #E8EAED; border-radius: 16px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.4rem;
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.06);
}
[data-testid="stChatMessage"]:has([data-testid*="User"]),
[data-testid="stChatMessage"]:has([data-testid*="user"]) {
    background: #F8F9FA;
}

/* Carte d'un extrait retourné par la recherche */
.chunk-card {
    background: #F8F9FA; border-left: 3px solid #1A73E8;
    border-radius: 8px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.9rem; color: #202124;
}
.chunk-source {
    display: inline-flex; align-items: center; gap: 5px;
    color: #1A73E8; font-weight: 600; font-size: 0.78rem; margin-bottom: 4px;
}
.chunk-card .sim-badge {
    float: right; background: #E6F4EA; color: #188038;
    border-radius: 999px; padding: 2px 9px; font-size: 0.72rem;
    font-weight: 600; margin-left: 8px;
}
.chunk-card .sim-badge .msr { font-size: 13px; vertical-align: -2px; }
/* Extrait sous le seuil de pertinence : il ne serait pas donné au LLM */
.chunk-card .sim-badge-low { background: #F1F3F4; color: #5F6368; }
.chunk-card:has(.sim-badge-low) { border-left-color: #DADCE0; opacity: 0.75; }

/* Écran d'accueil */
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
"""

# Icône Material et couleur selon le type de fichier.
FILE_ICONS = {
    "pdf": ("picture_as_pdf", "#D93025"),
    "md":  ("markdown",       "#188038"),
    "txt": ("description",    "#5F6368"),
}

# Avatars du chat (icônes Material rendues par Streamlit).
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}

# Historique de conversation sauvegardé à côté du code.
HISTORY_FILE = Path(__file__).resolve().parent / "chat_history.json"


# =============================================================================
# COMPOSANTS D'AFFICHAGE
# =============================================================================
# Le HTML est écrit sur une seule ligne : st.markdown interprète le Markdown,
# et une ligne indentée après une ligne vide y deviendrait un bloc de code
# (des balises </div> s'afficheraient alors en clair).

def section_label(text):
    """Titre de section de la barre latérale (petites majuscules)."""
    st.markdown(f'<div class="side-label">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def source_card(name, n_chunks):
    """Carte d'un document indexé : icône du format, nom, nombre de chunks."""
    icon, color = FILE_ICONS.get(name.rsplit(".", 1)[-1].lower(),
                                 FILE_ICONS["txt"])
    st.markdown(
        f'<div class="file-card"><span class="msr" style="color:{color}">'
        f'{icon}</span><span class="file-name">{html.escape(name)}</span>'
        f'<span class="file-badge">{n_chunks} chunks</span></div>',
        unsafe_allow_html=True,
    )


def stats_row(n_sources, n_chunks):
    """Compteurs de l'index : nombre de sources et de chunks."""
    st.markdown(
        f'<div class="stats-row">'
        f'<div class="stat-chip"><span class="msr">folder</span> '
        f'{n_sources} source(s)</div>'
        f'<div class="stat-chip"><span class="msr">grid_view</span> '
        f'{n_chunks} chunks</div></div>',
        unsafe_allow_html=True,
    )


def chunk_cards(chunks, threshold=None):
    """Affiche les extraits sous forme de cartes : source, score, contenu.

    threshold : fourni en mode recherche sémantique. Le badge de score
    indique alors si l'extrait passerait le seuil de pertinence du mode RAG
    (vert = donné au LLM, gris = écarté).
    """
    for chunk in chunks:
        content = html.escape(chunk["content"]).replace("\n", "<br>")
        score = chunk["score"]
        kept = threshold is None or score >= threshold
        css_class = "sim-badge" if kept else "sim-badge sim-badge-low"
        if threshold is None:
            mark = ""
        elif kept:
            mark = ' <span class="msr">check</span>'
        else:
            mark = ' <span class="msr">block</span>'
        st.markdown(
            f'<div class="chunk-card">'
            f'<span class="{css_class}">{score:.2f}{mark}</span>'
            f'<span class="chunk-source"><span class="msr">draft</span> '
            f'{html.escape(chunk["source"])}</span><br>{content}</div>',
            unsafe_allow_html=True,
        )


def header(llm_enabled):
    """En-tête : titre et pastille indiquant le mode actif."""
    if llm_enabled:
        pill = ('<span class="pill pill-rag"><span class="msr">auto_awesome'
                '</span> RAG complet</span>')
    else:
        pill = ('<span class="pill pill-search"><span class="msr">search'
                '</span> Recherche sémantique</span>')
    st.markdown(
        f'<div class="app-header"><div class="app-title">'
        f'<span class="msr">auto_stories</span> NotebookLM Local</div>{pill}'
        f'</div><div class="app-sub">Vos documents, vos réponses : rien ne '
        f'quitte votre machine.</div>',
        unsafe_allow_html=True,
    )


def hero():
    """Écran d'accueil, affiché tant qu'aucun document n'est indexé."""
    st.markdown(
        '<div class="hero">'
        '<div class="hero-badge"><span class="msr">auto_stories</span></div>'
        '<h2>Discutez avec vos documents</h2>'
        '<p>Un assistant documentaire 100&nbsp;% local, sans API externe.</p>'
        '<div class="steps">'
        '<div class="step"><span class="msr">upload_file</span>'
        '<b>1 · Ajouter</b><p>PDF, Markdown ou TXT dans la barre latérale</p></div>'
        '<div class="step"><span class="msr">bolt</span>'
        '<b>2 · Indexer</b><p>Extraction, chunking et vectorisation</p></div>'
        '<div class="step"><span class="msr">forum</span>'
        '<b>3 · Discuter</b><p>Recherche sémantique ou réponse générée</p></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def extraction_preview(docs):
    """Aperçu de l'Étape 2.1 : ce que l'extraction a produit."""
    with st.expander("Aperçu de l'extraction"):
        st.caption(f"{len(docs)} segment(s) extrait(s), 1 par page de PDF")
        for doc in docs[:3]:
            page = doc.metadata.get("page")
            page_info = f" · page {page + 1}" if page is not None else ""
            st.markdown(f"**{doc.metadata['source']}**{page_info} "
                        f"· {len(doc.page_content)} caractères")
            st.text(doc.page_content[:250] + "…")
        if len(docs) > 3:
            st.caption(f"… et {len(docs) - 3} autre(s) segment(s)")


def chunking_preview(chunks, strategy):
    """Aperçu de l'Étape 2.2 : tailles des chunks et premiers chunks."""
    with st.expander("Aperçu du chunking"):
        sizes = [len(chunk.page_content) for chunk in chunks]
        st.caption(f"Stratégie **{strategy}** · {len(chunks)} chunks · "
                   f"taille min {min(sizes)} / moy {sum(sizes) // len(sizes)} "
                   f"/ max {max(sizes)} caractères")
        for chunk in chunks[:2]:
            page = chunk.metadata.get("page")
            page_info = f" · page {page + 1}" if page is not None else ""
            st.markdown(f"**{chunk.metadata['source']}**{page_info} "
                        f"· {len(chunk.page_content)} caractères")
            st.text(chunk.page_content[:200] + "…")


# =============================================================================
# HISTORIQUE DE CONVERSATION (sauvegardé sur disque)
# =============================================================================
# st.session_state est perdu à chaque rafraîchissement de page : on garde
# donc l'historique dans un fichier JSON pour reprendre la conversation.

def load_history():
    """Renvoie l'historique sauvegardé, ou une liste vide.

    Un fichier absent ou corrompu ne doit pas faire planter l'application :
    on repart simplement d'une conversation vide.
    """
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(messages):
    """Écrit l'historique complet sur disque."""
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2),
                            encoding="utf-8")


# =============================================================================
# PAGE ET ÉTAT DE SESSION
# =============================================================================
st.set_page_config(
    page_title="NotebookLM Local",
    page_icon=":material/auto_stories:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# Streamlit ré-exécute tout le script à chaque interaction : ce qui doit
# survivre entre deux exécutions est rangé dans st.session_state.
# Au premier passage d'une session, on restaure l'historique et l'index
# persistés. Le marqueur "initialized" est posé EN DERNIER : si l'exécution
# est interrompue en cours de chargement, tout le bloc sera rejoué au lieu
# de laisser un état à moitié initialisé.
if "initialized" not in st.session_state:
    st.session_state.messages = load_history()   # [{role, content, …}]
    vectorstore, sources = load_vectorstore()
    st.session_state.vectorstore = vectorstore   # base ChromaDB (ou None)
    st.session_state.sources = sources           # [{name, n_chunks}]
    st.session_state.raw_docs = []               # aperçu de l'Étape 2.1
    st.session_state.chunks = []                 # aperçu de l'Étape 2.2
    st.session_state.chunk_strategy = CHUNKING_STRATEGY
    st.session_state.uploader_key = 0            # sert à vider l'uploader
    st.session_state.initialized = True

# Message de confirmation différé : l'indexation se termine par st.rerun(),
# qui effacerait un toast affiché juste avant. On l'affiche donc ici.
# Pas d'icône Material : dans un toast, la police globale l'afficherait sous
# forme de texte ("check_circle") au lieu du pictogramme.
if toast_message := st.session_state.pop("index_toast", None):
    st.toast(toast_message)

indexed = bool(st.session_state.sources)

# =============================================================================
# BARRE LATÉRALE — mode, documents, index
# =============================================================================
with st.sidebar:
    # --- Toggle du LLM : choix du mode --------------------------------------
    section_label("Assistant")
    llm_enabled = st.toggle("Générer avec le LLM", value=False)
    if llm_enabled:
        st.caption(f"Mode **RAG complet** : réponse rédigée par {LLM_MODEL} "
                   "(Ollama, 100 % local) à partir de vos documents.")
    else:
        st.caption("Mode **Recherche sémantique** : extraits bruts de la base "
                   "vectorielle, aucun LLM.")

    # Emplacement réservé au bouton « Nouvelle conversation ». La barre
    # latérale est dessinée AVANT le traitement de la question : le bouton
    # est donc ajouté en fin de script, pour apparaître dès le premier
    # échange et non à l'interaction suivante.
    new_conversation_slot = st.empty()

    # --- Téléchargement des documents ---------------------------------------
    st.divider()
    section_label("Ajouter des documents")
    # La clé du widget change après chaque indexation : Streamlit le recrée
    # vide, et les fichiers indexés n'apparaissent plus que dans la liste
    # « Sources indexées ».
    uploaded_files = st.file_uploader(
        "Glissez vos fichiers ici",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader-{st.session_state.uploader_key}",
    )

    strategy = st.selectbox(
        "Stratégie de découpage",
        CHUNKING_STRATEGIES,
        index=CHUNKING_STRATEGIES.index(CHUNKING_STRATEGY),
        format_func=lambda s: {"recursive": "Récursive (taille fixe)",
                               "semantic": "Sémantique (par le sens)"}[s],
        help="Récursive : coupe au séparateur le plus naturel, sous un "
             "plafond de 1000 caractères. Sémantique : coupe là où le sens "
             "change entre deux phrases (indexation plus lente).",
    )

    # --- Bouton d'indexation : pipeline de l'Étape 2 ------------------------
    if st.button("Indexer les documents", icon=":material/bolt:",
                 type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        with st.spinner("Extraction, découpage et vectorisation…"):
            docs = load_documents(uploaded_files)               # 2.1
            chunks = split_documents(docs, strategy)            # 2.2
            st.session_state.vectorstore = build_vectorstore(chunks)  # 2.3

        # Nombre de chunks obtenus pour chaque fichier.
        per_source = Counter(chunk.metadata["source"] for chunk in chunks)
        st.session_state.sources = [
            {"name": f.name, "n_chunks": per_source.get(f.name, 0)}
            for f in uploaded_files
        ]
        st.session_state.raw_docs = docs
        st.session_state.chunks = chunks
        st.session_state.chunk_strategy = strategy
        st.session_state.index_toast = (f"{len(uploaded_files)} fichier(s) → "
                                        f"{len(chunks)} chunks indexés")
        st.session_state.uploader_key += 1          # vide la zone d'ajout
        st.rerun()

    # --- Contenu réel de la base vectorielle --------------------------------
    if indexed:
        st.divider()
        section_label("Sources indexées")
        for source in st.session_state.sources:
            source_card(source["name"], source["n_chunks"])
        stats_row(len(st.session_state.sources),
                  sum(s["n_chunks"] for s in st.session_state.sources))

        # Action irréversible : elle passe par une confirmation.
        with st.popover("Vider l'index", icon=":material/delete:",
                        use_container_width=True):
            st.caption("Tous les documents indexés seront supprimés. "
                       "Il faudra les charger à nouveau.")
            if st.button("Confirmer la suppression", type="primary",
                         use_container_width=True):
                clear_index(st.session_state.vectorstore)
                st.session_state.vectorstore = None
                st.session_state.sources = []
                st.session_state.raw_docs = []
                st.session_state.chunks = []
                st.rerun()

    # --- Aperçus du pipeline (Étapes 2.1 et 2.2) ----------------------------
    if st.session_state.raw_docs:
        st.divider()
        section_label("Débogage du pipeline")
        extraction_preview(st.session_state.raw_docs)
        chunking_preview(st.session_state.chunks,
                         st.session_state.chunk_strategy)

# =============================================================================
# ZONE PRINCIPALE — interface conversationnelle avec historique
# =============================================================================
header(llm_enabled)

if not indexed and not st.session_state.messages:
    hero()

# Réaffichage de l'historique à chaque exécution du script. Les extraits
# sont rendus en cartes : visibles directement en mode recherche
# sémantique, repliés sous la réponse en mode RAG.
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message.get("chunks"):
            if message.get("mode") == "search":
                chunk_cards(message["chunks"], threshold=MIN_RELEVANCE)
            else:
                with st.expander("Sources utilisées"):
                    chunk_cards(message["chunks"])

# Zone de saisie, désactivée tant qu'aucun document n'est indexé.
query = st.chat_input(
    "Posez une question sur vos documents…" if indexed
    else "Indexez d'abord des documents dans la barre latérale",
    disabled=not indexed,
)

if query:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        if not llm_enabled:
            # ---- Étape 3 : recherche sémantique pure (aucun LLM) ----------
            # Aucun filtre : tous les extraits sont affichés, avec le nom du
            # fichier source. Le seuil sert seulement à colorer les badges,
            # pour voir ce que le mode RAG retiendrait.
            chunks = semantic_search(st.session_state.vectorstore, query)
            intro = (f"**Extraits les plus proches.** En vert, ceux qui "
                     f"passent le seuil de pertinence ({MIN_RELEVANCE:.2f}) "
                     f"et seraient donnés au LLM.")
            st.markdown(intro)
            chunk_cards(chunks, threshold=MIN_RELEVANCE)
            st.session_state.messages.append({
                "role": "assistant", "content": intro,
                "chunks": chunks, "mode": "search",
            })
        else:
            # ---- Étape 4 : RAG complet -------------------------------------
            # 1) Récupération des extraits pertinents (au-dessus du seuil).
            chunks = semantic_search(st.session_state.vectorstore, query,
                                     min_score=MIN_RELEVANCE)
            if not chunks:
                # Aucun extrait assez proche : inutile d'appeler le LLM.
                answer = ("Aucun passage de vos documents n'est assez proche "
                          "de cette question (tous les scores sont sous le "
                          "seuil de pertinence). Reformulez, ou vérifiez en "
                          "mode recherche sémantique.")
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
            else:
                # 2 et 3) Prompt strict et génération par le modèle local.
                # Ollama est un service à part : s'il est arrêté, on affiche
                # un message clair plutôt qu'une page d'erreur.
                try:
                    answer = st.write_stream(rag_answer(query, chunks))
                except Exception as exc:
                    answer = (f"⚠️ Le modèle local n'a pas répondu "
                              f"(`{type(exc).__name__}`). Vérifiez qu'Ollama "
                              f"tourne : `ollama serve`, puis `ollama run "
                              f"{LLM_MODEL}`. Les extraits trouvés restent "
                              f"consultables ci-dessous.")
                    st.warning(answer)
                # 4) Transparence : extraits utilisés, dépliables.
                with st.expander("Sources utilisées"):
                    chunk_cards(chunks)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "chunks": chunks})

    save_history(st.session_state.messages)

# Bouton « Nouvelle conversation », placé dans son emplacement de la barre
# latérale maintenant que l'historique est à jour.
if st.session_state.messages and new_conversation_slot.button(
        "Nouvelle conversation", icon=":material/add_comment:",
        use_container_width=True):
    st.session_state.messages = []
    HISTORY_FILE.unlink(missing_ok=True)
    st.rerun()
