"""
Banc d'essai du retrieval — justifier les choix par la MESURE
==============================================================
Le sujet demande de justifier la stratégie de chunking ; ce script étend
l'exigence à tout l'étage de recherche. Il compare des configurations sur
un jeu de questions dont on connaît le document attendu, et mesure :

  - top-1  : le bon document arrive-t-il en première position ?
  - score  : similarité de la cible (0..1)
  - marge  : écart entre la cible et le premier intrus  <-- LE critère

Pourquoi la marge plutôt que le score ? Un modèle qui donne 0.85 à la
bonne réponse ET 0.81 au bruit ne discrimine rien : aucun seuil de
pertinence ne peut être posé entre les deux. Un modèle plus "modeste"
mais qui sépare nettement est bien plus exploitable.

Usage :  python benchmark.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from embedding import EMBEDDING_MODEL, load_documents, split_documents

DOCS_FOLDER = "documents_test"

# Questions -> document attendu. Deux formulations dégradées (sans
# accents, style télégraphique) sont incluses volontairement : c'est ce
# qu'un vrai utilisateur tape.
CASES = [
    ("Combien de jours de congés par an ?", "reglement_interieur_entreprise.md"),
    ("combien de jour de conge?", "reglement_interieur_entreprise.md"),
    ("Comment réinitialiser mon mot de passe ?", "faq_support_informatique.md"),
    ("mot de passe oublie", "faq_support_informatique.md"),
    ("Quel document dois-je signer avant de commencer en entreprise ?",
     "guide_stage_master.txt"),
    ("Qui a inventé le test pour savoir si une machine pense ?",
     "histoire_intelligence_artificielle.txt"),
    ("Ma machine à café affiche une lumière orange",
     "manuel_cafetiere_expresso.pdf"),
    ("Le chiffre d'affaires a-t-il augmenté ?", "rapport_ventes_2025.pdf"),
    ("Je reçois un email bizarre qui me demande un virement urgent",
     "politique_securite_donnees.txt"),
    ("teletravail combien de jours", "reglement_interieur_entreprise.md"),
]


class LocalFile:
    """Expose un fichier du disque comme un UploadedFile de Streamlit."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        self._data = open(path, "rb").read()

    def getvalue(self):
        return self._data


def evaluate(name, chunks, embeddings, k=4):
    """Indexe les chunks dans une base jetable et mesure les trois métriques."""
    store = Chroma(
        collection_name=f"bench_{abs(hash(name)) % 99999}",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks)

    hits, scores, margins, failures = 0, [], [], []
    for query, expected in CASES:
        results = store.similarity_search_with_relevance_scores(query, k=k)
        if results[0][0].metadata["source"] == expected:
            hits += 1
        else:
            failures.append(query)
        target = next((s for d, s in results
                       if d.metadata["source"] == expected), 0.0)
        intruder = next((s for d, s in results
                         if d.metadata["source"] != expected), 0.0)
        scores.append(target)
        margins.append(target - intruder)

    store.delete_collection()          # base jetable : on ne laisse rien
    print(f"{name:32s} top-1 {hits:2d}/{len(CASES)}   "
          f"score {sum(scores)/len(scores):.3f}   "
          f"marge {sum(margins)/len(margins):+.3f}")
    for query in failures:
        print(f"{'':32s}   ↳ échec : {query}")


def main():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), DOCS_FOLDER)
    if not os.path.isdir(folder):
        sys.exit(f"Dossier introuvable : {folder}")

    documents = load_documents([LocalFile(os.path.join(folder, n))
                                for n in sorted(os.listdir(folder))])
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"{len(documents)} documents · {len(CASES)} questions\n")
    for strategy in ("recursive", "semantic"):
        chunks = split_documents(documents, strategy)
        evaluate(f"{strategy} ({len(chunks)} chunks)", chunks, embeddings)


if __name__ == "__main__":
    main()
