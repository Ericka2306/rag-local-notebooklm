"""ÉTAPE 3 — Recherche sémantique pure (AUCUN appel au LLM).  ✅

Amélioration par rapport au top-k statique : chaque extrait est retourné
avec son SCORE de similarité, et un seuil de pertinence optionnel permet
d'écarter le bruit — un top-4 systématique charge sinon des extraits hors
sujet dans le contexte du LLM.

Ce module ne connaît pas Streamlit : il reçoit la base vectorielle et une
question, il retourne des extraits.
"""

from rag.config import TOP_K


def semantic_search(vectorstore, query, k=TOP_K, min_score=None):
    """✅ Retourne les chunks dont le SENS est le plus proche de la question.

    Sous le capot, similarity_search_with_relevance_scores :
      1. vectorise la question avec le MÊME modèle d'embeddings que les
         chunks (règle d'or : sinon les vecteurs sont incomparables) ;
      2. calcule la similarité cosinus (0..1) entre ce vecteur et tous
         ceux de la base ;
      3. renvoie les k Documents les mieux classés, avec leur score.

    min_score : seuil de pertinence (cf. MIN_RELEVANCE, calibré dans
    rag/config.py). None = aucun filtre — c'est le réglage du mode audit
    (toggle désactivé), qui doit montrer TOUT le top-k, même le faible.
    Le mode RAG passe le seuil pour ne donner au LLM que de l'utile ; la
    liste peut alors revenir VIDE (question sans rapport avec les
    documents) — l'interface gère ce cas sans appeler le LLM.

    Retourne : une liste de dicts {"content", "source", "score"}, où
    "source" inclut le numéro de page pour les PDF — l'interface n'a pas
    besoin de connaître LangChain.
    """
    results = []
    for doc, score in vectorstore.similarity_search_with_relevance_scores(
            query, k=k):
        if min_score is not None and score < min_score:
            continue                        # sous le seuil : écarté
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page")
        if page is not None:                # PDF : on précise la page
            source += f" · page {page + 1}"
        results.append({"content": doc.page_content,
                        "source": source,
                        "score": round(score, 3)})
    return results
