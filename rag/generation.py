"""ÉTAPE 4 — Génération RAG : prompt strict + LLM local via Ollama.  🚧

Ce module ne connaît pas Streamlit : il reçoit la question et les extraits,
il retourne un flux de texte.
"""

import time

from rag.config import LLM_MODEL  # noqa: F401 (servira à l'implémentation)


def rag_answer(query, chunks):
    """🚧 À implémenter : construire un PromptTemplate injectant {context}
    (les chunks) et {question}, puis appeler le modèle Ollama en streaming.

    Doit retourner : un GÉNÉRATEUR de morceaux de texte (pour l'affichage
    progressif avec st.write_stream).
    """
    fake = (f"Réponse factice à « {query} » générée à partir de "
            f"{len(chunks)} extraits. Branche-moi sur Ollama !")
    for word in fake.split(" "):          # simule le streaming token par token
        yield word + " "
        time.sleep(0.05)
