---
title: Swift MLX documentation research
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.43.1
app_file: step04_chatbot.py
pinned: false
license: mit
hardware: zerogpu
short_description: Search in the Swift MLX documentation
models:
- Qwen/Qwen3-Embedding-4B
- Qwen/Qwen3-Reranker-4B
- Qwen/Qwen3-4B-Instruct-2507
datasets:
- VincentGOURBIN/swift-mlx-Qwen3-Embedding-4B
tags:
- rag
- retrieval-augmented-generation
- qwen3
- semantic-search
- question-answering
- zero-gpu
- mcp-server
- faiss
---

# 🔍 LocalRAG - Système RAG Complet avec Qwen3

✅ **Système complet et fonctionnel** - Compatible MPS (Mac), CUDA et ZeroGPU Spaces

Système RAG (Retrieval-Augmented Generation) utilisant les modèles Qwen3 de dernière génération avec pipeline 2 étapes : recherche vectorielle + reranking + génération streamée.

## ⚡ Fonctionnalités

### 🧠 **Modèles IA Avancés**
- **Embeddings**: Qwen3-Embedding-4B (2560 dimensions)
- **Reranking**: Qwen3-Reranker-4B pour l'affinage des résultats
- **Génération**: Qwen3-4B-Instruct-2507 avec streaming
- **Optimisation ZeroGPU**: Support natif avec décorateurs @spaces.GPU

### 📄 **Support Multi-Formats avec PDF Intelligent**
- **HTML/Markdown**: Parsing structurel avec préservation hiérarchique
- **PDF avancé**: Chunking sémantique intelligent pour documents longs (50+ pages)
- **Détection automatique**: Structure, TOC et titres des PDFs
- **Overlap contextuel**: Préservation du contexte entre chunks (100-200 mots)
- **Batch processing**: Indexation par lots pour éviter les crashes mémoire

### 🔍 **Recherche Sémantique Avancée**
- **Pipeline 2 étapes**: Recherche vectorielle + reranking
- **Index FAISS**: Recherche haute performance sur de gros volumes
- **Scores détaillés**: Embedding + reranking pour chaque document
- **Sélection intelligente**: Top-K adaptatif selon pertinence

### 🚀 **Modes de Déploiement Flexibles**
- **Mode HuggingFace Hub**: Téléchargement automatique des embeddings
- **Mode FAISS Local**: Utilisation directe de l'index local (5x plus rapide)
- **Mode Offline**: Fonctionnement complet sans connexion internet
- **Mode Public Sécurisé**: Partage via URL avec authentification admin

### 🔐 **Sécurité et Authentification**
- **Mot de passe aléatoire**: 16 caractères cryptographiquement sécurisés
- **Génération automatique**: Combinaison lettres + chiffres + symboles
- **Session unique**: Nouveau mot de passe à chaque démarrage
- **Contrôle d'accès**: Interface publique avec protection admin

### 💬 **Génération Contextuelle**
- **Streaming**: Réponse progressive token par token
- **Contexte enrichi**: Intégration des documents les plus pertinents
- **Références**: Sources avec scores de pertinence
- **Qualité**: Réponses basées uniquement sur le contexte fourni

### 🔌 **Intégration MCP**
- **Serveur MCP natif**: Fonction `ask_rag_question()` exposée
- **Paramètres configurables**: Nombre documents, activation reranking
- **Compatible**: Claude Desktop, VS Code, Cursor IDE
- **API structurée**: Réponses JSON avec sources et métadonnées

## 🛠️ Architecture & Étapes

### Pipeline de Traitement
1. **Step01** : Téléchargement automatique de documentation web (optionnel)
2. **Step02** : Indexation universelle (HTML/Markdown/PDF → FAISS + métadonnées)
3. **Step03** : Upload embeddings vers HuggingFace Hub (optionnel)
4. **Step04** : Interface de chat RAG (mode local ou cloud)

### Recherche en 2 Étapes
1. **Recherche vectorielle** : FAISS IndexFlatIP (cosine similarity)
2. **Reranking** : Qwen3-Reranker-4B pour affiner la pertinence
3. **Génération** : Qwen3-4B-Instruct-2507 avec streaming

## 📥 Step01 - Téléchargeur de Documentation

Le nouveau **step01_downloader.py** permet de télécharger automatiquement de la documentation depuis internet pour alimenter votre système RAG.

### Fonctionnalités
- **Interface simplifiée** : Utilisation directe avec --start-url
- **URLs de départ multiples** : Support de plusieurs points d'entrée
- **Téléchargement intelligent** : Respect des robots.txt et limitations
- **Formats supportés** : HTML, Markdown, PDF
- **Filtrage automatique** : Exclusion des fichiers non pertinents
- **Détection automatique** : Base URL automatiquement détectée

### Utilisation
```bash
# Télécharger depuis une URL de départ (stocké dans ./data/docs/)
python step01_downloader.py --start-url https://pytorch.org/docs/stable/ --output docs

# Télécharger depuis plusieurs URLs de départ (stocké dans ./data/api_docs/)
python step01_downloader.py --start-url https://site1.com/docs --start-url https://site2.com/api --output api_docs

# Options avancées avec base URL personnalisée (stocké dans ./data/swift_docs/)
python step01_downloader.py --start-url https://docs.swift.org/swift-book/ --base-url https://docs.swift.org --output swift_docs --workers 5

# Reprendre un téléchargement interrompu
python step01_downloader.py --start-url https://site.com/docs --output my_docs --resume
```

### Options disponibles
- `--start-url` : URL de départ pour le crawling (requis, peut être répété)
- `--base-url` : URL de base (optionnel, auto-détecté depuis la première start-url)
- `--output` : Nom du dossier (sera créé dans ./data/, défaut: downloaded_docs)
- `--workers` : Nombre de workers parallèles (défaut: 3)
- `--resume` : Reprendre un téléchargement interrompu (flag optionnel)

## 🚀 Utilisation

### Installation & Démarrage Rapide
```bash
# Cloner le projet
git clone [repo-url]
cd LocalRagModel

# Installer les dépendances (inclut Selenium pour téléchargement web)
pip install -r requirements.txt

# Option 1: Télécharger documentation web (optionnel)
# Les données sont automatiquement stockées dans ./data/[nom_dossier]
python step01_downloader.py --start-url https://pytorch.org/docs/stable/ --output pytorch_docs
# Ou plusieurs URLs de départ
python step01_downloader.py --start-url https://site1.com/docs --start-url https://site2.com/api --output docs_mixtes

# Option 2: Indexer vos documents locaux (HTML/Markdown/PDF)
python step02_indexer.py docs/ --no-flash-attention

# Lancer en mode local rapide (recommandé)
python step04_chatbot.py --local-faiss

# Ou lancer en mode public sécurisé
python step04_chatbot.py --local-faiss --share
```

### Modes de Lancement

#### Mode Local (Défaut HuggingFace)
```bash
python step04_chatbot.py
```

#### Mode Local FAISS ⭐ **Recommandé**
```bash
python step04_chatbot.py --local-faiss
```
- 🚀 **5x plus rapide** au démarrage
- 🔌 **Fonctionne offline**
- 📁 **Utilise directement** vos données indexées

#### Mode Public Sécurisé
```bash
python step04_chatbot.py --local-faiss --share
```
- 🌐 **Interface publique** accessible via URL Gradio
- 🔐 **Authentification automatique** (admin / mot_de_passe_16_chars)
- 🔑 **Nouveau mot de passe** à chaque démarrage

#### Options Avancées
```bash
# Chemin FAISS personnalisé
python step04_chatbot.py --local-faiss --faiss-path ./mon_index

# Utilisateur admin personnalisé
python step04_chatbot.py --share --admin-user myuser
```

### Interface Web
1. **Posez votre question** dans le chat
2. **Observez la recherche** en 2 étapes (vectorielle → reranking)
3. **Lisez la réponse** générée en streaming
4. **Consultez les sources** avec scores de pertinence

### Configuration des Modèles
- **MPS (Mac)** : Support natif, Flash Attention désactivé
- **CUDA (NVIDIA)** : Flash Attention disponible (désactivable)  
- **ZeroGPU** : Décorateurs `@spaces.GPU` automatiques

### Intégration MCP
Connectez votre client MCP pour un accès programmatique :
```python
# Exemple d'utilisation MCP
result = mcp_client.call_tool(
    "ask_rag_question",
    question="Comment implémenter des réseaux de neurones complexes?",
    num_documents=3,
    use_reranking=True
)
```

## 🎯 Cas d'Usage Parfaits

- **Documentation technique**: Recherche dans APIs, guides, tutoriels (HTML/Markdown)
- **Manuels PDF**: Documents techniques, rapports, livres (PDF chunking intelligent)
- **Support client**: Réponses basées sur une base de connaissances
- **Recherche académique**: Analyse de corpus documentaires longs
- **Assistance développeur**: Aide contextuelle sur frameworks/librairies
- **Formation**: Système de questions-réponses intelligent
- **Démonstrations**: Partage public sécurisé d'assistants personnalisés

## 📊 Performance

- **Recherche**: ~50ms pour 10K+ documents
- **Reranking**: ~200ms pour 20 candidats  
- **Génération**: ~2-4s avec streaming
- **Mémoire**: ~6-8GB optimisé pour ZeroGPU

## 🔒 Sécurité & Confidentialité

- **ZeroGPU**: Traitement sécurisé sans stockage persistant
- **Données temporaires**: Pas de rétention des questions/réponses
- **Modèles locaux**: Traitement dans l'environnement HF Spaces

## 📚 Source des Données

Ce Space utilise des embeddings pré-calculés depuis le dataset :
**[VincentGOURBIN/swift-mlx-Qwen3-Embedding-4B](https://huggingface.co/datasets/VincentGOURBIN/swift-mlx-Qwen3-Embedding-4B)**

## 📋 TODO / Roadmap

### ✅ Fonctionnalités Complètes
- [x] Pipeline RAG complet (recherche + reranking + génération)
- [x] Interface Gradio avec streaming
- [x] Support multi-plateforme (MPS/CUDA/ZeroGPU)
- [x] **Support PDF avec chunking sémantique intelligent**
- [x] **Mode FAISS local (5x plus rapide, offline)**
- [x] **Mode public sécurisé avec authentification**
- [x] **Batch processing FAISS (anti-crash)**
- [x] Intégration MCP native
- [x] Déploiement automatique HuggingFace Spaces

### 🔄 Améliorations Futures
- [ ] **Extraction d'images PDF** : Analyse des diagrammes et schémas
- [ ] **OCR pour PDFs scannés** : Support des documents numérisés
- [ ] **Upload documents sources** vers HuggingFace Hub
- [x] **Step01** : Téléchargement automatique de documentation technique depuis internet
- [ ] Support formats additionnels (DOCX, PowerPoint)
- [ ] Interface d'administration pour gestion des documents

---

🚀 **Projet complet et fonctionnel** - Commencez à poser vos questions pour découvrir la puissance du RAG avec Qwen3! 🔍✨
