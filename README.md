# 📚 NotebookLM Local — Système RAG 100 % local

Clone local de [NotebookLM](https://notebooklm.google.com/) : chargez vos propres
documents (PDF, Markdown, TXT) et discutez avec eux, **sans qu'aucune donnée ne
quitte votre machine** — aucun appel à une API externe.

> Projet réalisé dans le cadre d'un TP de Master 1 (IA & développement).

## ✨ Fonctionnalités

- **Interface conversationnelle** type chat, avec historique et réponses en streaming
- **Deux modes**, commutables par un toggle :
  - 🔎 **Recherche sémantique pure** — affiche les extraits bruts les plus
    proches de la question (aucun LLM), idéal pour auditer la base vectorielle
  - 🤖 **RAG complet** — un LLM local rédige une réponse contrainte par les
    documents, avec les sources consultables sous chaque réponse
- **Confidentialité totale** : embeddings et génération s'exécutent en local
- Design inspiré de NotebookLM (Material Symbols, palette Google)

## 🏗️ Architecture

```
Documents ─► Extraction (PyMuPDF) ─► Chunking (LangChain) ─► Embeddings ─► ChromaDB
                                                    (sentence-transformers)
Question ─► Embedding ─► Recherche par similarité ─┬─► extraits bruts + sources
                                                   └─► PromptTemplate ─► Mistral (Ollama) ─► réponse + sources
```

## 🧰 Stack technique

| Brique | Rôle |
|---|---|
| [Streamlit](https://streamlit.io/) | Interface web |
| [LangChain](https://www.langchain.com/) | Orchestration du pipeline |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extraction du texte des PDF |
| [sentence-transformers](https://www.sbert.net/) | Embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`) |
| [ChromaDB](https://www.trychroma.com/) | Base vectorielle locale |
| [Ollama](https://ollama.com/) + Mistral 7B | Génération locale |

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Ericka2306/rag-local-notebooklm.git
cd rag-local-notebooklm

# 2. Environnement Python (3.10+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ollama + modèle (https://ollama.com pour l'installation)
ollama pull mistral

# 4. Lancer l'application
streamlit run app.py
```

Puis ouvrez http://localhost:8501, ajoutez vos documents dans la barre
latérale, cliquez sur **Indexer** et posez vos questions.

## 📁 Structure du projet

```
├── app.py             # Script de page Streamlit (Étape 1) — point d'entrée
├── ui.py              # Composants d'affichage réutilisables
├── styles.css         # Feuille de style (Material Symbols, palette Google)
├── requirements.txt
└── rag/               # Backend, indépendant de Streamlit
    ├── config.py      # Constantes : modèles, chunking, top-k
    ├── ingestion.py   # Étape 2 : extraction → chunking → vectorisation
    ├── retrieval.py   # Étape 3 : recherche sémantique
    └── generation.py  # Étape 4 : prompt strict + génération LLM
```

## 🚧 Statut

- [x] Étape 1 — Interface (sidebar sources, chat, toggle deux modes)
- [x] Étape 2 — Pipeline d'ingestion (extraction → chunking → vectorisation)
- [x] Étape 3 — Mode recherche sémantique
- [ ] Étape 4 — Mode RAG complet (prompt strict + génération)
