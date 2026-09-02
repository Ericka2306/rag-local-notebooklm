"""ÉTAPE 4 — Génération RAG : prompt strict + LLM local via Ollama.  ✅

Ce module ne connaît pas Streamlit : il reçoit la question et les extraits,
il retourne un flux de texte.
"""

from functools import lru_cache

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from rag.config import LLM_MODEL

# --- Ingénierie de prompt (cf. sujet, Étape 4.2) -----------------------------
# Le prompt est un CONTRAT, pas une formule magique (cf. leçon, p. 9) :
#   - rôle défini, règles numérotées et impératives ;
#   - la consigne clé d'un RAG : répondre EXCLUSIVEMENT depuis le contexte,
#     et avouer son ignorance plutôt qu'halluciner ;
#   - {context} et {question} injectés dynamiquement par PromptTemplate.
RAG_PROMPT = PromptTemplate.from_template(
    """Tu es un assistant documentaire rigoureux.

Règles impératives :
1. Réponds à la question en te basant EXCLUSIVEMENT sur le contexte
   ci-dessous. N'utilise AUCUNE connaissance extérieure.
2. Si le contexte ne permet pas de répondre, réponds exactement :
   "Je ne trouve pas cette information dans les documents fournis."
3. Réponds en français, de façon claire et concise.
4. Quand c'est pertinent, mentionne le document d'où vient l'information.

### CONTEXTE :
{context}

### QUESTION :
{question}

### RÉPONSE :"""
)


@lru_cache(maxsize=1)
def get_llm():
    """Client vers le modèle local servi par Ollama (une instance suffit).

    temperature=0.1 : on veut des réponses factuelles, collées aux sources —
    la créativité est un défaut pour un assistant documentaire.
    """
    return OllamaLLM(model=LLM_MODEL, temperature=0.1)


def rag_answer(query, chunks):
    """✅ Construit le prompt final (contexte + question) et interroge le LLM.

    1. Concatène les chunks retrouvés (Étape 3), chacun étiqueté par sa
       source : le LLM peut ainsi citer le bon document (règle 4 du prompt).
    2. Injecte {context} et {question} dans le PromptTemplate.
    3. Appelle le modèle en STREAMING : retourne un générateur de morceaux
       de texte, consommé par st.write_stream côté interface — la réponse
       s'affiche au fil de la génération au lieu d'attendre la fin.
    """
    context = "\n\n".join(
        f"[Source : {chunk['source']}]\n{chunk['content']}"
        for chunk in chunks
    )
    prompt = RAG_PROMPT.format(context=context, question=query)
    return get_llm().stream(prompt)
