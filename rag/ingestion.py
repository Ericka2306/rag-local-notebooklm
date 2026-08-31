"""ÉTAPE 2 — Pipeline d'ingestion : extraction → chunking → vectorisation.

Avancement : 2.1 extraction ✅ · 2.2 chunking ✅ · 2.3 vectorisation 🚧

Ce module ne connaît PAS Streamlit : il reçoit des fichiers, retourne des
objets — c'est l'interface (app.py) qui gère l'affichage et l'état de session.
"""

import os
import tempfile

# Les DocumentLoaders : connecteurs "fichier -> objets Document"
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# Le découpeur récursif : coupe au séparateur le plus "naturel" possible
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(uploaded_files):
    """✅ ÉTAPE 2.1 — Extraction : fichiers uploadés -> objets Document.

    Un objet Document LangChain = du texte (page_content) + des métadonnées
    (metadata). Pour un PDF, PyMuPDFLoader produit UN Document PAR PAGE,
    avec le numéro de page dans metadata — précieux pour citer les sources.

    `uploaded_files` : objets exposant .name et .getvalue() (les UploadedFile
    de Streamlit, ou n'importe quel objet compatible — testable sans UI).

    Piège résolu ici : les fichiers uploadés par Streamlit vivent EN MÉMOIRE,
    alors que les loaders LangChain attendent un CHEMIN sur disque. On écrit
    donc chaque fichier dans un fichier temporaire, on le charge, puis on le
    supprime.
    """
    documents = []
    for uploaded in uploaded_files:
        # Extension du fichier original (".pdf", ".txt", ".md")
        suffix = os.path.splitext(uploaded.name)[1].lower()

        # Copie temporaire sur disque (delete=False : on gère la suppression
        # nous-mêmes, APRÈS que le loader a lu le fichier).
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())      # contenu binaire du fichier
            tmp_path = tmp.name

        try:
            # Le bon DocumentLoader selon le format (cf. sujet, Étape 2.1).
            if suffix == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            for doc in loader.load():
                # Le loader a mis le CHEMIN TEMPORAIRE dans metadata["source"]
                # (ex: /var/folders/…/tmpx7.pdf). On l'écrase par le vrai nom
                # du fichier : l'affichage des sources (Étape 3) en dépend.
                doc.metadata["source"] = uploaded.name
                documents.append(doc)
        finally:
            # Exécuté même si le loader lève une erreur : pas de fichiers
            # temporaires orphelins qui s'accumulent.
            os.remove(tmp_path)

    return documents


def split_documents(documents):
    """✅ ÉTAPE 2.2 — Chunking : découper les Documents en segments.

    Pourquoi découper ? Deux raisons opposées à équilibrer :
      - un chunk trop GROS mélange plusieurs sujets -> son embedding est
        "dilué", la recherche devient imprécise, et on gaspille la fenêtre
        de contexte du LLM à l'Étape 4 ;
      - un chunk trop PETIT (une phrase isolée) n'a plus assez de contexte
        pour être compris, ni par la recherche ni par le LLM.
    Les valeurs retenues (et leur justification) : rag/config.py.

    "Recursive" = le splitter essaie de couper au séparateur le plus naturel
    d'abord : entre paragraphes ("\\n\\n"), sinon entre lignes ("\\n"), sinon
    entre mots (" "), en dernier recours au milieu d'un mot. Les frontières
    de chunks tombent donc (presque toujours) à des endroits sensés.

    Les métadonnées (source, page) sont propagées automatiquement à chaque
    chunk par split_documents — indispensable pour citer les sources.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # taille MAXIMALE d'un chunk (caractères)
        chunk_overlap=CHUNK_OVERLAP,  # chevauchement entre chunks consécutifs
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """🚧 ÉTAPE 2.3 — Vectorisation : chunks -> embeddings -> ChromaDB.

    À implémenter : HuggingFaceEmbeddings(EMBEDDING_MODEL) puis
    Chroma.from_documents(...). Attention : charger le modèle d'embeddings
    est coûteux — il faudra le mettre en cache côté interface
    (@st.cache_resource) ou dans ce module.

    Doit retourner : l'objet vectorstore prêt à être interrogé.
    """
    return None             # ← provisoire : pas encore de base vectorielle
