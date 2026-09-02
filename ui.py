"""Composants d'affichage de l'interface (HTML + Material Symbols).

Séparés de app.py pour garder le script de page lisible : app.py décrit
QUOI afficher et dans quel ordre ; ui.py sait COMMENT le dessiner.
Aucune logique métier ici — uniquement du rendu.
"""

import html
from pathlib import Path

import streamlit as st

# Icône Material + couleur selon le type de fichier.
_FILE_ICONS = {
    "pdf": ("picture_as_pdf", "#D93025"),   # rouge Google
    "md":  ("markdown",       "#188038"),   # vert Google
    "txt": ("description",    "#5F6368"),   # gris
}

# Avatars du chat (icônes Material rendues nativement par Streamlit).
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}


def load_css():
    """Injecte la feuille de style externe (styles.css) dans la page."""
    css = Path(__file__).parent.joinpath("styles.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def section_label(text):
    """Libellé de section de la sidebar (petites majuscules espacées)."""
    st.markdown(f'<div class="side-label">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def source_card(name, n_chunks=None):
    """Carte d'un document indexé : icône typée + nom + badge de chunks."""
    icon, color = _FILE_ICONS.get(name.rsplit(".", 1)[-1].lower(),
                                  _FILE_ICONS["txt"])
    badge = (f'<span class="file-badge">{n_chunks} chunks</span>'
             if n_chunks is not None else "")
    st.markdown(
        f"""<div class="file-card">
            <span class="msr" style="color:{color}">{icon}</span>
            <span class="file-name">{html.escape(name)}</span>{badge}
        </div>""",
        unsafe_allow_html=True,
    )


def stats_row(n_sources, n_chunks):
    """Les deux compteurs de l'index (sources / chunks)."""
    st.markdown(
        f"""<div class="stats-row">
            <div class="stat-chip"><span class="msr">folder</span>
                {n_sources} source(s)</div>
            <div class="stat-chip"><span class="msr">grid_view</span>
                {n_chunks} chunks</div>
        </div>""",
        unsafe_allow_html=True,
    )


def chunk_cards(chunks):
    """Cartes des extraits retournés par la recherche (source + contenu).

    HTML volontairement compact (une seule ligne, sauts de ligne convertis
    en <br>) : st.markdown interprète le Markdown, et une ligne indentée
    après une ligne vide deviendrait un bloc de code — on a vu des </div>
    s'afficher en clair à cause de ça.
    """
    for chunk in chunks:
        content = html.escape(chunk["content"]).replace("\n", "<br>")
        score = chunk.get("score")
        badge = (f'<span class="sim-badge">similarité {score:.2f}</span>'
                 if score is not None else "")
        st.markdown(
            f'<div class="chunk-card">{badge}'
            f'<span class="chunk-source"><span class="msr">draft</span> '
            f'{html.escape(chunk["source"])}</span><br>{content}</div>',
            unsafe_allow_html=True,
        )


def header(llm_enabled):
    """En-tête : titre + pastille indiquant le mode actif."""
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


def hero():
    """Écran d'accueil affiché tant qu'aucun document n'est indexé."""
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


def extraction_preview(docs):
    """🔬 Débogage Étape 2.1 : ce que l'extraction a produit."""
    # (pas d'icône : les en-têtes d'expander ne rendent pas les ligatures
    # Material de façon fiable avec notre police globale)
    with st.expander("Aperçu de l'extraction"):
        st.caption(f"{len(docs)} segment(s) extrait(s) — 1 par page de PDF")
        for doc in docs[:3]:
            page = doc.metadata.get("page")
            page_info = f" · page {page + 1}" if page is not None else ""
            st.markdown(f"**{doc.metadata['source']}**{page_info} "
                        f"· {len(doc.page_content)} caractères")
            st.text(doc.page_content[:250] + "…")
        if len(docs) > 3:
            st.caption(f"… et {len(docs) - 3} autre(s) segment(s)")


def chunking_preview(chunks):
    """🔬 Débogage Étape 2.2 : distribution des tailles et premiers chunks."""
    with st.expander("Aperçu du chunking"):
        sizes = [len(c.page_content) for c in chunks]
        st.caption(f"{len(chunks)} chunks — taille min {min(sizes)} / "
                   f"moy {sum(sizes)//len(sizes)} / max {max(sizes)} car.")
        for c in chunks[:2]:
            page = c.metadata.get("page")
            page_info = f" · page {page + 1}" if page is not None else ""
            st.markdown(f"**{c.metadata['source']}**{page_info} "
                        f"· {len(c.page_content)} caractères")
            st.text(c.page_content[:200] + "…")
