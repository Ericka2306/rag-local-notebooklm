"""
ÉTAPE 2 — Ingestion des sources dans la base de données vectorielle
===================================================================
Pipeline déclenché par le bouton « Indexer » de l'interface :

    fichiers uploadés
      └─ 2.1 Extraction    : texte + métadonnées (DocumentLoaders LangChain)
          └─ 2.2 Chunking  : découpage en segments (taille + chevauchement)
              └─ 2.3 Vectorisation : embeddings locaux -> stockage ChromaDB

Ce fichier ne dépend pas de Streamlit : il reçoit des fichiers et renvoie
des objets. C'est l'interface (app.py) qui gère l'affichage et l'état de
session.
"""

import os
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

# Connecteurs "fichier -> objets Document" (Étape 2.1)
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
# Découpeurs de texte (Étape 2.2)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# Embeddings locaux et base vectorielle (Étape 2.3)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =============================================================================
# PARAMÈTRES
# =============================================================================

# --- Modèle d'embeddings (100 % local, via sentence-transformers) -----------
# Multilingue, car nos documents sont en français : le classique
# all-MiniLM-L6-v2 est surtout entraîné sur de l'anglais. Léger (~470 Mo)
# et produit des vecteurs de 384 dimensions.
# Alternative testée et écartée : intfloat/multilingual-e5-base. Ses scores
# sont plus élevés (0.847 contre 0.592), mais l'écart entre la bonne
# réponse et le premier extrait hors sujet est 6 fois plus faible (+0.043
# contre +0.274) : il juge tout presque aussi pertinent, et aucun seuil de
# pertinence ne peut alors être placé entre les deux.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking (justification exigée par l'Étape 2.2) ------------------------
# Taille maximale d'un chunk : 1000 caractères (~200 mots). Assez pour
# contenir une idée complète, assez peu pour rester précis : un chunk trop
# gros mélange plusieurs sujets et "dilue" son embedding.
CHUNK_SIZE = 1000

# Chevauchement : 150 caractères (15 %). Une information coupée à la
# frontière de deux chunks reste entière dans au moins l'un des deux.
CHUNK_OVERLAP = 150

# Deux stratégies implémentées pour pouvoir les comparer :
#   - "recursive" : plafond de taille fixe, coupe au séparateur le plus
#     naturel (paragraphe, sinon ligne, sinon mot) ;
#   - "semantic"  : coupe là où le sens change entre deux phrases.
# Le choix du défaut repose sur une mesure (10 questions dont le document
# attendu est connu, sur le jeu documents_test) :
#   recursive  bon document en 1re position 10/10   écart avec le bruit +0.274
#   semantic   bon document en 1re position  9/10   écart avec le bruit +0.258
# Sur des documents bien structurés, le sémantique coupe au milieu des
# sections (ex. : un chunk mêlant la fin du télétravail et le début des
# congés), là où le récursif respecte les paragraphes.
CHUNKING_STRATEGIES = ("recursive", "semantic")
CHUNKING_STRATEGY = "recursive"

# Réglages propres au chunking sémantique :
#   - on coupe quand la distance entre deux phrases consécutives dépasse le
#     75e percentile des distances du document (~25 % des transitions) ;
#   - un chunk de moins de 200 caractères (titre isolé) est fusionné avec
#     le suivant, faute de contexte suffisant pour être retrouvé ;
#   - un chunk de plus de 2000 caractères (texte sans rupture de sujet) est
#     redécoupé, pour ne pas diluer son embedding.
SEMANTIC_BREAKPOINT_PERCENTILE = 75
SEMANTIC_MIN_CHUNK_SIZE = 200
SEMANTIC_MAX_CHUNK_SIZE = 2 * CHUNK_SIZE

# --- Base vectorielle --------------------------------------------------------
# Index persisté sur disque : les documents indexés survivent aux
# redémarrages de l'application et aux rafraîchissements de page. Chemin
# ancré au dossier du projet, et non au dossier courant (qui dépend de
# l'endroit d'où la commande est lancée).
CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
COLLECTION_NAME = "rag_documents"


# =============================================================================
# 2.1 — EXTRACTION : fichiers uploadés -> objets Document
# =============================================================================

def load_documents(uploaded_files):
    """Lit les fichiers uploadés et renvoie une liste d'objets Document.

    Un Document LangChain = du texte (page_content) + des métadonnées
    (metadata). Pour un PDF, PyMuPDFLoader produit un Document PAR PAGE,
    avec le numéro de page dans les métadonnées, utile pour citer la source.

    `uploaded_files` : objets exposant .name et .getvalue(), c'est-à-dire
    les UploadedFile de Streamlit (ou tout objet compatible).

    Les fichiers uploadés par Streamlit vivent EN MÉMOIRE, alors que les
    loaders LangChain attendent un CHEMIN sur disque : chaque fichier est
    donc écrit dans un fichier temporaire, chargé, puis supprimé.
    """
    documents = []
    for uploaded in uploaded_files:
        # Extension du fichier d'origine (".pdf", ".txt", ".md")
        suffix = os.path.splitext(uploaded.name)[1].lower()

        # Copie temporaire sur disque. delete=False : on supprime nous-mêmes
        # le fichier, APRÈS que le loader l'a lu.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            # DocumentLoader adapté au format
            if suffix == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            else:                                 # .txt et .md : texte brut
                loader = TextLoader(tmp_path, encoding="utf-8")

            for doc in loader.load():
                # Le loader a placé le CHEMIN TEMPORAIRE dans
                # metadata["source"] (ex. /var/folders/…/tmpx7.pdf). On le
                # remplace par le vrai nom du fichier : c'est ce nom que
                # l'interface affiche à côté de chaque extrait (Étape 3).
                doc.metadata["source"] = uploaded.name
                documents.append(doc)
        finally:
            # Exécuté même si le loader échoue (fichier corrompu) : aucun
            # fichier temporaire ne reste sur le disque.
            os.remove(tmp_path)

    return documents


# =============================================================================
# 2.2 — CHUNKING : découpage des Documents en segments
# =============================================================================

def split_documents(documents, strategy=CHUNKING_STRATEGY):
    """Découpe les Documents en chunks selon la stratégie choisie.

    Pourquoi découper ? Deux contraintes opposées à équilibrer :
      - un chunk trop GROS mélange plusieurs sujets : son embedding est
        dilué, la recherche perd en précision, et il occupe inutilement la
        fenêtre de contexte du LLM (Étape 4) ;
      - un chunk trop PETIT (une phrase isolée) n'a plus assez de contexte
        pour être compris, ni par la recherche ni par le LLM.

    "recursive" : le découpeur essaie d'abord de couper entre paragraphes
    ("\\n\\n"), sinon entre lignes ("\\n"), sinon entre mots (" "), et en
    dernier recours au milieu d'un mot. Les coupures tombent donc presque
    toujours à des endroits sensés, sous un plafond de taille garanti.

    "semantic" : chaque phrase est vectorisée, et on coupe là où la
    similarité entre deux phrases consécutives chute (changement de sujet).
    Les chunks suivent les idées plutôt que la mise en page, au prix d'une
    indexation plus lente et de tailles variables.

    Dans les deux cas, les métadonnées (source, page) sont recopiées sur
    chaque chunk : indispensable pour afficher la source d'un extrait.
    """
    # Pages vides (page blanche, PDF scanné sans texte) : rien à découper.
    documents = [doc for doc in documents if doc.page_content.strip()]

    if strategy == "semantic":
        semantic = SemanticChunker(
            get_embeddings(),              # le MÊME modèle que l'index
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_PERCENTILE,
            min_chunk_size=SEMANTIC_MIN_CHUNK_SIZE,
        )
        chunks = semantic.split_documents(documents)
        # Garde-fou : ne redécoupe que les chunks trop longs.
        cap = RecursiveCharacterTextSplitter(
            chunk_size=SEMANTIC_MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return cap.split_documents(chunks)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # taille MAXIMALE d'un chunk
        chunk_overlap=CHUNK_OVERLAP,    # chevauchement entre chunks voisins
    )
    return splitter.split_documents(documents)


# =============================================================================
# 2.3 — VECTORISATION : chunks -> embeddings -> ChromaDB
# =============================================================================

@lru_cache(maxsize=1)
def get_embeddings():
    """Charge le modèle d'embeddings UNE SEULE FOIS par processus.

    Le chargement est coûteux (~470 Mo de poids en mémoire), or Streamlit
    ré-exécute tout le script à chaque interaction. @lru_cache mémorise le
    résultat du premier appel : même effet que @st.cache_resource, sans
    rendre ce fichier dépendant de Streamlit.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _open_vectorstore():
    """Ouvre (ou crée) la collection ChromaDB persistée sur disque.

    Métrique "cosine" au lieu de la distance L2 par défaut : la similarité
    cosinus est bornée entre 0 et 1, donc les scores sont lisibles tels
    quels et un seuil de pertinence a un sens (cf. fonction_rag.py).
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(chunks):
    """Vectorise les chunks et les stocke dans la base. Renvoie la base.

    add_documents fait l'essentiel :
      1. passe le texte de CHAQUE chunk au modèle d'embeddings, ce qui
         donne un vecteur de 384 nombres par chunk ;
      2. stocke dans ChromaDB le trio (vecteur, texte, métadonnées). Les
         métadonnées, dont le nom du fichier source, restent attachées.

    reset_collection d'abord : « Indexer » REMPLACE l'index par la
    sélection courante. Sans cela, réindexer les mêmes fichiers ajouterait
    chaque chunk en double.
    """
    vectorstore = _open_vectorstore()
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    return vectorstore


def clear_index(vectorstore):
    """Vide l'index en passant par le client ChromaDB.

    Ne jamais supprimer le dossier chroma_db/ à la main pendant que
    l'application tourne : le client garde la base ouverte et se retrouve
    avec une base fantôme ("attempt to write a readonly database").
    """
    vectorstore.reset_collection()


def load_vectorstore():
    """Restaure l'index persisté par une session précédente, s'il existe.

    Renvoie (vectorstore, sources), où sources = [{"name", "n_chunks"}] est
    reconstruit à partir des métadonnées stockées ; ou (None, []) si aucun
    index exploitable n'existe.
    """
    if not os.path.isdir(CHROMA_DIR):
        return None, []

    vectorstore = _open_vectorstore()

    # Garde-fou : une collection créée avec une autre métrique (ex. L2)
    # donnerait des scores faussés, et tout passerait silencieusement sous
    # le seuil de pertinence. On la déclare inutilisable : la prochaine
    # indexation la recrée avec la bonne métrique (reset_collection).
    metric = (vectorstore._collection.metadata or {}).get("hnsw:space")
    if metric != "cosine":
        return None, []

    metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
    if not metadatas:
        return None, []

    counts = Counter(meta.get("source", "inconnue") for meta in metadatas)
    sources = [{"name": name, "n_chunks": n}
               for name, n in sorted(counts.items())]
    return vectorstore, sources
