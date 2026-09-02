"""Configuration du pipeline RAG — toutes les constantes au même endroit."""

# --- Modèles (100 % locaux) --------------------------------------------------

# Modèle d'embeddings : multilingue (nos documents sont en français) et léger
# (~470 Mo, vecteurs de 384 dimensions). Le classique all-MiniLM-L6-v2 est
# surtout entraîné sur de l'anglais — mauvais choix pour du français.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modèle génératif servi par Ollama (cf. sujet : mistral ou qwen2.5-coder).
LLM_MODEL = "mistral"

# --- Stratégie de chunking (justification exigée par l'Étape 2.2) ------------

# chunk_size = 1000 caractères (~200-250 mots) : assez grand pour qu'un chunk
#   contienne une idée complète, assez petit pour que la recherche reste
#   précise (un gros chunk "dilue" son embedding en mélangeant les sujets).
CHUNK_SIZE = 1000

# chunk_overlap = 150 caractères (15 %) : le chevauchement évite qu'une idée
#   coupée à la frontière de deux chunks devienne introuvable.
CHUNK_OVERLAP = 150

# --- Persistance -------------------------------------------------------------

# Dossier où ChromaDB persiste l'index sur disque : les documents indexés
# survivent aux redémarrages de l'application et aux rafraîchissements de
# page. Chemin absolu ancré à la racine du projet (et non au dossier
# courant, qui dépend d'où l'on lance la commande).
from pathlib import Path

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")

# --- Recherche ---------------------------------------------------------------

# Nombre de chunks récupérés par requête (le "top-k" de la similarité).
TOP_K = 4

# Seuil de pertinence (similarité cosinus, 0..1) appliqué en mode RAG :
# un chunk sous ce score n'entre pas dans le contexte du LLM. Calibré
# empiriquement sur les documents de test : les vraies cibles scorent
# 0.48-0.72, le bruit d'accompagnement 0.22-0.44, le hors-sujet < 0.16.
# Le mode audit (toggle désactivé) ignore ce seuil : on veut tout voir.
MIN_RELEVANCE = 0.40
