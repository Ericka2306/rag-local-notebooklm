"""ÉTAPE 3 — Recherche sémantique pure (AUCUN appel au LLM).  🚧

Ce module ne connaît pas Streamlit : il reçoit la base vectorielle et une
question, il retourne des extraits.
"""

from rag.config import TOP_K


def semantic_search(vectorstore, query, k=TOP_K):
    """🚧 À implémenter : interroger la base vectorielle (similarity_search)
    et retourner les k chunks dont le sens est le plus proche de la question.

    Doit retourner : une liste de dicts {"content": str, "source": str}.
    """
    return [                              # ← extraits factices
        {"content": f"Extrait factice n°{i} en lien avec « {query} ». "
                    "Remplace-moi par un vrai similarity_search !",
         "source": f"document_{i}.pdf"}
        for i in (1, 2, 3)
    ]
