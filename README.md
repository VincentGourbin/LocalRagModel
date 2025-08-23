---
title: Swift MLX documentation research
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.43.1
app_file: step03_chatbot.py
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

### 🔍 **Recherche Sémantique Avancée**
- **Pipeline 2 étapes**: Recherche vectorielle + reranking
- **Index FAISS**: Recherche haute performance sur de gros volumes
- **Scores détaillés**: Embedding + reranking pour chaque document
- **Sélection intelligente**: Top-K adaptatif selon pertinence

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
1. **Step01** : Indexation des documents (FAISS + métadonnées)
2. **Step02** : Upload embeddings vers HuggingFace Hub
3. **Step03** : Interface de chat RAG (compatible MPS/CUDA/ZeroGPU)
4. **Step04** : Déploiement automatique sur HuggingFace Spaces

### Recherche en 2 Étapes
1. **Recherche vectorielle** : FAISS IndexFlatIP (cosine similarity)
2. **Reranking** : Qwen3-Reranker-4B pour affiner la pertinence
3. **Génération** : Qwen3-4B-Instruct-2507 avec streaming

## 🚀 Utilisation

### Interface Web
1. **Posez votre question** dans le chat
2. **Observez la recherche** en 2 étapes (vectorielle → reranking)
3. **Lisez la réponse** générée en streaming
4. **Consultez les sources** avec scores de pertinence

### Installation Locale
```bash
# Cloner le projet
git clone [repo-url]
cd LocalRagModel

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'interface
python step03_chatbot.py
```

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

- **Documentation technique**: Recherche dans APIs, guides, tutoriels
- **Support client**: Réponses basées sur une base de connaissances
- **Recherche académique**: Analyse de corpus documentaires
- **Assistance développeur**: Aide contextuelle sur frameworks/librairies
- **Formation**: Système de questions-réponses intelligent

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
- [x] Intégration MCP native
- [x] Déploiement automatique HuggingFace Spaces

### 🔄 Améliorations Futures
- [ ] **Optimisation performances** (caching, batch processing)
- [ ] **Upload documents sources** vers HuggingFace Hub
- [ ] **Step00** : Téléchargement automatique de documentation technique depuis internet
- [ ] Support formats additionnels (PDF, DOCX, HTML)
- [ ] Interface d'administration pour gestion des documents

---

🚀 **Projet complet et fonctionnel** - Commencez à poser vos questions pour découvrir la puissance du RAG avec Qwen3! 🔍✨
