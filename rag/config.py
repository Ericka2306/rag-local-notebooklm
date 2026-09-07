"""Configuration du pipeline RAG — toutes les constantes au même endroit."""

# --- Modèles (100 % locaux) --------------------------------------------------

# Modèle d'embeddings : multilingue (nos documents sont en français) et léger
# (~470 Mo, vecteurs de 384 dimensions). Le classique all-MiniLM-L6-v2 est
# surtout entraîné sur de l'anglais — mauvais choix pour du français.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Alternative testée et ÉCARTÉE : intfloat/multilingual-e5-base (~1,1 Go).
# Scores bien plus élevés (0.847 contre 0.592) mais marge par rapport au
# bruit 6x plus faible (+0.043 contre +0.274), y compris avec ses préfixes
# "query:"/"passage:" obligatoires : il juge tout également pertinent,
# donc aucun seuil (MIN_RELEVANCE) n'est plaçable. Un score élevé ne vaut
# rien sans séparation — c'est la marge qui décide.

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

# --- Stratégie de découpage ---------------------------------------------------
# La leçon (p. 13) demande des passages "suffisamment PETITS et COHÉRENTS" :
#   - "recursive" : plafond de taille fixe, coupe au séparateur le plus
#     naturel (paragraphe > ligne > mot). Garantit "petits" ; "cohérents"
#     seulement si la mise en page l'est. Rapide, prévisible — le DÉFAUT.
#   - "semantic"  : vectorise chaque phrase et coupe là où la similarité
#     entre phrases consécutives chute (changement de sujet). Vise
#     "cohérents" directement, au prix d'une indexation plus lente (un
#     passage du modèle d'embeddings par phrase) et de tailles variables.
# Le défaut n'est pas une intuition mais une MESURE (voir benchmark.py,
# 10 questions à document attendu connu, sur documents_test/) :
#   recursive  top-1 10/10   score 0.592   marge +0.274
#   semantic   top-1  9/10   score 0.560   marge +0.258
# Le sémantique échoue notamment sur "combien de jour de conge?" : sur ces
# documents Markdown BIEN STRUCTURÉS, il coupe en travers des sections
# (un chunk mêlant fin du télétravail et début des congés), là où le
# récursif respecte les titres. Le sémantique paierait sur du texte au
# kilomètre sans structure — pas ici.
CHUNKING_STRATEGIES = ("recursive", "semantic")
CHUNKING_STRATEGY = "recursive"

# Chunking sémantique : on coupe quand la distance entre deux phrases
# consécutives dépasse le 75e percentile des distances du document
# (= ~25 % des transitions les plus fortes deviennent des frontières).
SEMANTIC_BREAKPOINT_PERCENTILE = 75

# Garde-fous de taille, dans les deux sens :
#   - un chunk sémantique trop COURT (titre isolé, phrase orpheline) n'a
#     pas assez de contexte pour être retrouvé -> fusionné avec le suivant ;
#   - un chunk trop LONG (texte au kilomètre sans rupture de sujet) dilue
#     son embedding -> re-découpé au-delà du plafond.
SEMANTIC_MIN_CHUNK_SIZE = 200
SEMANTIC_MAX_CHUNK_SIZE = 2 * CHUNK_SIZE

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
