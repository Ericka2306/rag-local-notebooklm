"""ÉTAPE 3 — Recherche sémantique pure (AUCUN appel au LLM).  ✅

Ce module ne connaît pas Streamlit : il reçoit la base vectorielle et une
question, il retourne des extraits.
"""

from rag.config import TOP_K


def semantic_search(vectorstore, query, k=TOP_K):
    """✅ Retourne les k chunks dont le SENS est le plus proche de la question.

    Sous le capot, similarity_search :
      1. vectorise la question avec le MÊME modèle d'embeddings que les
         chunks (règle d'or : sinon les vecteurs sont incomparables) ;
      2. calcule la similarité entre ce vecteur et tous ceux de la base ;
      3. renvoie les k Documents les mieux classés.

    Aucun modèle génératif ici : c'est une pure recherche vectorielle —
    exactement ce que le mode "toggle désactivé" doit exposer pour auditer
    la base (cf. sujet, Étape 3).

    Retourne : une liste de dicts {"content": str, "source": str}, où
    "source" inclut le numéro de page pour les PDF (métadonnées conservées
    à l'Étape 2) — l'interface n'a pas besoin de connaître LangChain.
    """
    results = []
    for doc in vectorstore.similarity_search(query, k=k):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page")
        if page is not None:                    # PDF : on précise la page
            source += f" · page {page + 1}"
        results.append({"content": doc.page_content, "source": source})
    return results
