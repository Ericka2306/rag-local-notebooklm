"""
ÉTAPES 3 ET 4 — Recherche dans la base vectorielle et génération RAG
====================================================================
Étape 3 : recherche sémantique pure. La question est vectorisée et comparée
          aux chunks de la base ; aucun modèle génératif n'intervient.
Étape 4 : RAG complet. Les extraits retrouvés servent de contexte à un
          prompt strict, envoyé au modèle local servi par Ollama.

Ce fichier ne dépend pas de Streamlit : il reçoit la base vectorielle, la
question et les extraits, et renvoie des données ou un flux de texte.
"""

from functools import lru_cache

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# =============================================================================
# PARAMÈTRES
# =============================================================================

# Nombre d'extraits récupérés par question (le "top-k" de la recherche).
TOP_K = 4

# Seuil de pertinence (similarité cosinus entre 0 et 1), appliqué en mode
# RAG uniquement : un extrait sous ce score n'entre pas dans le contexte du
# LLM. Calibré sur le jeu de test : les bons extraits obtiennent 0.48 à
# 0.72, les extraits voisins mais hors sujet 0.22 à 0.44, et les questions
# sans rapport avec les documents moins de 0.16.
MIN_RELEVANCE = 0.40

# Modèle génératif servi localement par Ollama (http://localhost:11434).
LLM_MODEL = "mistral"


# =============================================================================
# ÉTAPE 3 — RECHERCHE SÉMANTIQUE
# =============================================================================

def semantic_search(vectorstore, query, k=TOP_K, min_score=None):
    """Renvoie les extraits dont le SENS est le plus proche de la question.

    similarity_search_with_relevance_scores :
      1. vectorise la question avec le MÊME modèle d'embeddings que les
         chunks (sinon les vecteurs ne seraient pas comparables) ;
      2. calcule la similarité cosinus entre ce vecteur et ceux de la base ;
      3. renvoie les k chunks les plus proches, avec leur score.

    min_score : seuil de pertinence. None = aucun filtre, c'est le réglage
    du mode recherche sémantique (toggle désactivé), qui doit montrer tous
    les extraits, même les moins pertinents. Le mode RAG passe le seuil pour
    ne donner au LLM que des extraits utiles : la liste peut alors revenir
    vide, et l'interface répond sans appeler le LLM.

    Renvoie une liste de dicts {"content", "source", "score"}. "source" est
    le nom exact du fichier, lu dans les métadonnées du chunk, complété du
    numéro de page pour un PDF.
    """
    results = []
    for doc, score in vectorstore.similarity_search_with_relevance_scores(
            query, k=k):
        if min_score is not None and score < min_score:
            continue                        # sous le seuil : écarté
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page")
        if page is not None:                # PDF : pages numérotées dès 0
            source += f" · page {page + 1}"
        results.append({"content": doc.page_content,
                        "source": source,
                        "score": round(score, 3)})
    return results


# =============================================================================
# ÉTAPE 4 — GÉNÉRATION RAG
# =============================================================================

# Ingénierie de prompt : le prompt est un contrat, pas une formule magique.
#   - rôle défini, règles numérotées et impératives ;
#   - la règle clé d'un RAG : répondre EXCLUSIVEMENT à partir du contexte,
#     et reconnaître qu'on ne sait pas plutôt que d'inventer ;
#   - {context} et {question} sont injectés dynamiquement par PromptTemplate.
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

    temperature=0.1 : on veut des réponses factuelles, fidèles aux sources.
    La créativité est un défaut pour un assistant documentaire.
    """
    return OllamaLLM(model=LLM_MODEL, temperature=0.1)


def rag_answer(query, chunks):
    """Construit le prompt final et interroge le LLM local en streaming.

    1. Concatène les extraits retrouvés (Étape 3), chacun précédé de sa
       source, pour que le LLM puisse citer le bon document (règle 4).
    2. Injecte {context} et {question} dans le PromptTemplate.
    3. Appelle le modèle en streaming : renvoie un générateur de morceaux de
       texte, que l'interface affiche au fil de la génération.
    """
    context = "\n\n".join(
        f"[Source : {chunk['source']}]\n{chunk['content']}"
        for chunk in chunks
    )
    prompt = RAG_PROMPT.format(context=context, question=query)
    return get_llm().stream(prompt)
