"""ÉTAPE 2 — Pipeline d'ingestion : extraction → chunking → vectorisation.

Avancement : 2.1 extraction ✅ · 2.2 chunking ✅ · 2.3 vectorisation ✅

Ce module ne connaît PAS Streamlit : il reçoit des fichiers, retourne des
objets — c'est l'interface (app.py) qui gère l'affichage et l'état de session.
"""

import os
import tempfile
from collections import Counter
from functools import lru_cache

# Les DocumentLoaders : connecteurs "fichier -> objets Document"
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# Le découpeur récursif : coupe au séparateur le plus "naturel" possible
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings locaux (sentence-transformers) et pont vers la base ChromaDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_DIR


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


@lru_cache(maxsize=1)
def get_embeddings():
    """Charge le modèle d'embeddings UNE SEULE FOIS par processus.

    Instancier HuggingFaceEmbeddings est coûteux (chargement de ~470 Mo de
    poids en mémoire). Or Streamlit ré-exécute le script à chaque clic :
    sans cache, chaque indexation rechargerait le modèle. @lru_cache
    mémorise le résultat du premier appel — même effet que
    @st.cache_resource, mais sans dépendre de Streamlit (module testable
    en dehors de l'interface).
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _open_vectorstore():
    """Ouvre (ou crée) la collection ChromaDB persistée sur disque.

    Métrique "cosine" plutôt que le L2 par défaut de Chroma : la similarité
    cosinus est normalisée, donc les scores de pertinence sont directement
    interprétables entre 0 et 1 — indispensable pour appliquer un seuil
    (MIN_RELEVANCE) et afficher des scores lisibles dans l'interface.
    NB : changer la métrique exige de ré-indexer (la collection est
    recréée avec ces réglages par reset_collection).
    """
    return Chroma(
        collection_name="tp_rag",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(chunks):
    """✅ ÉTAPE 2.3 — Vectorisation : chunks -> embeddings -> ChromaDB.

    add_documents fait tout le travail :
      1. passe le page_content de CHAQUE chunk au modèle d'embeddings
         -> un vecteur de 384 nombres par chunk ;
      2. stocke dans la base le trio (vecteur, texte, métadonnées) —
         les métadonnées (source, page) restent attachées, comme exigé
         par le sujet.

    La base est PERSISTÉE sur disque (chroma_db/) : l'index survit aux
    redémarrages de l'application et aux rafraîchissements de page —
    cf. load_vectorstore, appelé au démarrage.

    reset_collection d'abord : cliquer "Indexer" REMPLACE l'index par la
    sélection courante (sans ça, ré-indexer les mêmes fichiers ajouterait
    chaque chunk en double).

    Retourne : l'objet vectorstore prêt à être interrogé (Étape 3).
    """
    vectorstore = _open_vectorstore()
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    return vectorstore


def clear_index(vectorstore):
    """Vide proprement l'index (en passant par le client ChromaDB).

    À utiliser au lieu de supprimer le dossier chroma_db/ à la main : le
    client Chroma est mis en cache par processus, et supprimer ses fichiers
    sous ses pieds le laisse avec une base fantôme ("readonly database").
    """
    vectorstore.reset_collection()


def load_vectorstore():
    """Restaure l'index persisté par une session précédente, s'il existe.

    Retourne (vectorstore, sources) où sources = [{"name", "n_chunks"}]
    reconstruit depuis les métadonnées stockées — ou (None, []) si aucun
    index n'a été persisté.
    """
    if not os.path.isdir(CHROMA_DIR):
        return None, []

    vectorstore = _open_vectorstore()

    # Garde-fou : un index créé avec une autre métrique que la nôtre
    # (ex. le L2 par défaut, avant le passage au cosinus) produirait des
    # scores faussés — tout passerait sous le seuil de pertinence, en
    # silence. On le déclare inutilisable : l'utilisateur ré-indexe, et
    # reset_collection recrée la collection avec la bonne métrique.
    metric = (vectorstore._collection.metadata or {}).get("hnsw:space")
    if metric != "cosine":
        return None, []

    metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
    if not metadatas:
        return None, []

    counts = Counter(m.get("source", "inconnue") for m in metadatas)
    sources = [{"name": name, "n_chunks": n}
               for name, n in sorted(counts.items())]
    return vectorstore, sources
