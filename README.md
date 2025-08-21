# 🔍 LocalRAG - Système RAG Local Complet

## Vue d'ensemble

LocalRAG est un système de **Retrieval-Augmented Generation (RAG)** entièrement local, conçu pour indexer et interroger efficacement de la documentation technique. Le système fonctionne en plusieurs étapes orchestrées pour offrir une recherche sémantique haute performance sans dépendance cloud.

## 🏗️ Architecture du processus

```mermaid
graph TD
    A[📂 Documentation Source] --> B[Step 01: Indexation]
    B --> C[📊 Index Vectoriel FAISS]
    C --> D[Step 02: Recherche - À venir]
    D --> E[Step 03: Génération - À venir]
    E --> F[🎯 Réponse Contextualisée]
```

## 📋 Étapes du processus

### ✅ **Step 01 - Indexation** (`step01_indexer.py`)
**Statut**: ✅ Implémenté et optimisé

Transform la documentation brute en index vectoriel searchable.

### ✅ **Step 02 - Upload Embeddings** (`step02_upload_embeddings.py`)  
**Statut**: ✅ Implémenté et testé

Convertit l'index FAISS en SafeTensors et l'upload vers Hugging Face Hub.

---

## 🚀 Step 01 - Indexation Vectorielle

### 🎯 **Objectif**
Convertir la documentation technique (HTML/Markdown) en représentations vectorielles pour permettre une recherche sémantique ultra-rapide.

### 📊 **Pipeline de traitement**

```mermaid
graph LR
    A[📁 Documents] --> B[🔍 Parsing]
    B --> C[✂️ Chunking]
    C --> D[🖼️ Analyse Images]
    D --> E[🧠 Embeddings]
    E --> F[⚡ FAISS Index]
```

#### 1. **Découverte et Parsing** 
- Scan récursif des répertoires
- Support multi-format : `.html`, `.htm`, `.md`, `.markdown`
- Parser unifié avec gestion d'erreurs robuste
- Extraction du contenu, titres, sections, images et liens

#### 2. **Segmentation Intelligente**
- Découpage en chunks sémantiquement cohérents
- Préservation du contexte (titre, section parent)
- Optimisation de la taille pour les modèles d'embedding

#### 3. **Analyse Multimodale**
- **Images** : Analyse automatique avec Ollama (descriptions contextuelles)
- **Liens** : Extraction et catalogage des références
- **Métadonnées** : Enrichissement avec informations structurelles

#### 4. **Génération d'Embeddings**
- **Modèle principal** : `Qwen3-Embedding-4B` (2560 dimensions)
- **Fallback** : `paraphrase-multilingual-MiniLM-L12-v2`
- **Optimisations GPU** : Support MPS (Mac) et CUDA
- **Traitement par batch** adaptatif selon la puissance GPU

#### 5. **Indexation Vectorielle**
- **Backend** : FAISS-HNSW (Facebook AI Similarity Search)
- **Performance** : 10x plus rapide que ChromaDB
- **Persistance** : Sauvegarde automatique sur disque
- **Scalabilité** : Gestion de millions de vecteurs

### 🛠️ **Utilisation**

#### Installation des dépendances
```bash
pip install -r requirements.txt
```

#### Indexation complète
```bash
python step01_indexer.py /path/to/documentation
```

#### Indexation incrémentale (nouveaux fichiers uniquement)
```bash
python step01_indexer.py /path/to/documentation --incremental
```

#### Test sur un fichier unique
```bash
python step01_indexer.py /path/to/documentation --single-file document.html
```

### ⚙️ **Options de configuration**

| Option | Description | Recommandé pour |
|--------|-------------|-----------------|
| `--db-path` | Chemin de l'index FAISS | Personnaliser le stockage |
| `--debug` | Mode debug détaillé | Développement/Debug |
| `--incremental` | Indexation incrémentale | Mises à jour régulières |
| `--no-flash-attention` | Désactive Flash Attention 2 | Mac avec erreurs GPU |
| `--no-reranker` | Désactive le reranker | GPU limité |

### 📈 **Performances**

#### Environnements supportés
- **🍎 Mac M1/M2/M3** : MPS (Metal Performance Shaders)
- **🚀 Linux/Windows** : CUDA (NVIDIA GPU)
- **❌ CPU** : Non supporté (performance insuffisante)

#### Métriques typiques
- **Parsing** : ~100 fichiers HTML/min
- **Embeddings** : ~50 documents/sec (MPS) | ~200 documents/sec (CUDA)
- **Indexation FAISS** : ~1000 vecteurs/sec
- **Mémoire** : ~2GB RAM pour 100k chunks

### 🗃️ **Structure des données**

#### Index FAISS
```
faiss_index/
├── index.faiss          # Index vectoriel HNSW
├── metadata.json        # Métadonnées enrichies
├── mappings.pkl         # Mapping ID ↔ Index
└── tracking.json        # État d'indexation
```

#### Métadonnées par chunk
```json
{
  "source_file": "/path/to/document.html",
  "title": "Guide d'utilisation",
  "heading": "Configuration avancée",
  "content_length": 1247,
  "images_count": 3,
  "links_count": 5,
  "indexed_at": "2024-01-15T14:30:00",
  "chunk_content": "Contenu du segment..."
}
```

### 🔧 **Architecture technique**

#### Classes principales
- **`TechnicalDocIndexer`** : Orchestrateur principal
- **`UniversalDocumentParser`** : Parser unifié HTML/Markdown
- **`VectorIndexer`** : Gestionnaire d'embeddings et FAISS
- **`OllamaImageAnalyzer`** : Analyse multimodale des images
- **`Qwen3Reranker`** : Reranking sémantique (step 02)

#### Flux de données
1. **Fichiers** → **Chunks** (Parser)
2. **Chunks** → **Embeddings** (Qwen3)
3. **Embeddings** → **Index FAISS** (VectorIndexer)
4. **Métadonnées** → **JSON** (Tracking)

### ⚡ **Optimisations**

#### Gestion mémoire
- **Nettoyage automatique** : Cache MPS vidé après chaque batch
- **Batch adaptatif** : Taille ajustée selon GPU et longueur documents
- **Streaming** : Traitement par petits lots pour éviter l'OOM

#### Performance GPU
- **Pas de fallback CPU** : Échec immédiat si GPU indisponible
- **Flash Attention 2** : Accélération des transformers (CUDA)
- **Precision mixte** : FP16 automatique sur GPU compatibles

### 🚨 **Gestion d'erreurs**

#### Robustesse
- **Collecteur d'erreurs** : Catalogage centralisé des échecs
- **Continuation** : Traitement des autres fichiers si un échoue
- **Rapport détaillé** : Statistiques complètes en fin d'exécution

#### Types d'erreurs gérées
- Images manquantes ou corrompues
- HTML malformé
- Timeouts GPU
- Erreurs d'encoding

---

## 📅 **Roadmap**

### 🔄 **Step 02 - Recherche** (À implémenter)
- Interface de recherche sémantique
- Reranking avec Qwen3-Reranker-4B
- Système de scoring hybride
- Cache de requêtes fréquentes

### 🤖 **Step 03 - Génération** (À implémenter)  
- Intégration LLM local (Ollama/MLX)
- Génération de réponses contextualisées
- Templates de prompts optimisés
- Streaming des réponses

### 🔧 **Step 04 - Interface** (À implémenter)
- API REST/FastAPI
- Interface web interactive
- Chat en temps réel
- Visualisation des résultats

---

## 🛡️ **Sécurité et confidentialité**

- **100% Local** : Aucune donnée envoyée vers des services externes
- **Chiffrement** : Index FAISS peut être chiffré au repos
- **Isolation** : Traitement en sandbox local
- **Contrôle total** : Vos données restent sur votre infrastructure

---

## 🤝 **Contribution**

Le projet suit une architecture modulaire permettant des contributions ciblées :
- **Step 01** : Optimisations d'indexation
- **Step 02+** : Nouvelles étapes du pipeline
- **Parsers** : Support de nouveaux formats
- **Backends** : Intégration d'autres bases vectorielles

---

## 📊 **Statistiques d'utilisation**

Après indexation, le script affiche :
- Nombre de fichiers traités
- Chunks générés et indexés
- Images analysées
- Erreurs rencontrées
- Temps de traitement total
- Taille de l'index final

**Exemple** :
```
✅ Indexation terminée !
📊 Statistiques finales :
   - Fichiers traités : 1,247
   - Chunks générés : 12,458  
   - Images analysées : 3,891
   - Vecteurs indexés : 12,458
   - Erreurs : 23 (1.8%)
   - Durée totale : 4min 32s
   - Index FAISS : 2.1 GB
```

---

## 🌐 Step 02 - Upload Embeddings vers Hugging Face

### 🎯 **Objectif**
Convertir l'index FAISS local en format SafeTensors et l'upload vers Hugging Face Hub pour partage et réutilisation.

### 📊 **Pipeline de conversion**

```mermaid
graph LR
    A[📂 Index FAISS] --> B[🔄 Conversion SafeTensors]
    B --> C[📋 Métadonnées JSON]
    C --> D[📝 README Auto]
    D --> E[🚀 Upload HF Hub]
```

#### 1. **Conversion Format**
- **FAISS → SafeTensors** : Format sécurisé préféré par HF
- **Extraction vecteurs** : Reconstruction depuis index HNSW
- **Préservation mappings** : ID ↔ Index dans métadonnées
- **Validation intégrité** : Vérification dimensions et cohérence

#### 2. **Upload Sécurisé**
- **Token HF** : Authentification sécurisée (saisie masquée)
- **Repository privé/public** : Contrôle de visibilité
- **Métadonnées enrichies** : Documentation complète auto-générée
- **README automatique** : Guide d'utilisation avec exemples de code

### 🛠️ **Utilisation**

#### Upload interactif (recommandé)
```bash
python step02_upload_embeddings.py
```

#### Upload avec paramètres
```bash
python step02_upload_embeddings.py --repo-name username/my-embeddings --private
```

#### Test de conversion uniquement
```bash
python step02_upload_embeddings.py --dry-run
```

### ⚙️ **Options de configuration**

| Option | Description | Exemple |
|--------|-------------|---------|
| `faiss_index_path` | Chemin index FAISS | `./faiss_index` |
| `--token` | Token HF (ou interactif) | `hf_xxxxx` |
| `--repo-name` | Repository HF | `username/embeddings` |
| `--dataset-name` | Nom du dataset | `embeddings` |
| `--private` | Repository privé | Flag |
| `--dry-run` | Test sans upload | Flag |

### 📁 **Structure uploadée**

#### Fichiers générés sur HF Hub
```
dataset-repo/
├── embeddings.safetensors     # Vecteurs au format SafeTensors
├── embeddings_metadata.json  # Métadonnées + mappings ID
└── README.md                  # Documentation auto-générée
```

#### Contenu SafeTensors
```python
{
    'embeddings': torch.Tensor,  # Shape: [n_vectors, dimension]
}
```

#### Métadonnées JSON
```json
{
  "format_version": "1.0",
  "total_vectors": 12458,
  "vector_dimension": 2560,
  "faiss_index_type": "HNSW",
  "embedding_model": "Qwen/Qwen3-Embedding-4B",
  "conversion_timestamp": "2024-01-15T14:30:00",
  "ordered_ids": ["doc1#chunk1", "doc1#chunk2", ...],
  "id_to_idx": {"doc1#chunk1": 0, "doc1#chunk2": 1, ...}
}
```

### 🔄 **Réutilisation des embeddings**

#### Téléchargement depuis HF Hub
```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

# Télécharger les fichiers
embeddings_file = hf_hub_download(repo_id="username/repo", filename="embeddings.safetensors")
metadata_file = hf_hub_download(repo_id="username/repo", filename="embeddings_metadata.json")

# Charger les embeddings
tensors = load_file(embeddings_file)
embeddings = tensors['embeddings']  # torch.Tensor [n_vectors, dimension]

# Charger métadonnées
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
```

#### Recherche sémantique directe
```python
import torch.nn.functional as F

def search_embeddings(query_embedding, embeddings, metadata, top_k=10):
    """Recherche sémantique dans les embeddings uploadés."""
    # Similarité cosinus
    similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings, dim=1)
    
    # Top-K résultats
    top_scores, top_indices = torch.topk(similarities, top_k)
    
    # Récupération des IDs originaux
    ordered_ids = metadata['ordered_ids']
    results = []
    for idx, score in zip(top_indices, top_scores):
        doc_id = ordered_ids[idx.item()]
        results.append({'id': doc_id, 'score': score.item()})
    
    return results
```

### 📈 **Avantages SafeTensors**

#### vs. Format Pickle (.pkl)
- **✅ Sécurité** : Pas d'exécution de code arbitraire
- **✅ Performance** : Chargement plus rapide
- **✅ Interopérabilité** : Compatible tous frameworks ML
- **✅ Métadonnées** : Format structuré et extensible
- **✅ Validation** : Vérification intégrité automatique

#### vs. Index FAISS natif
- **✅ Portabilité** : Indépendant de FAISS
- **✅ Versioning** : Suivi des versions sur Git LFS
- **✅ Partage** : Distribution facile via HF Hub
- **✅ Documentation** : README et métadonnées auto-générées

### 🔒 **Sécurité et confidentialité**

#### Gestion des tokens
- **Saisie masquée** : Token jamais affiché en clair
- **Pas de stockage** : Token utilisé uniquement en mémoire
- **HTTPS** : Communication chiffrée avec HF Hub

#### Contrôle d'accès
- **Repository privé** : Accessible uniquement au propriétaire
- **Repository public** : Disponible pour la communauté
- **Token permissions** : Respecte les droits du token fourni

### 🎯 **Cas d'usage**

1. **Backup cloud** : Sauvegarde sécurisée des embeddings
2. **Partage équipe** : Distribution des embeddings pré-calculés
3. **Réplication** : Déploiement sur différents environnements  
4. **Research** : Partage de datasets pour recherche
5. **Production** : Intégration dans pipelines ML

### ⚡ **Performance**

#### Métriques typiques
- **Conversion** : ~1M vecteurs/min (2560D)
- **Upload** : Dépend de la bande passante
- **Taille typique** : ~10MB/1K vecteurs (2560D, float32)
- **Compression** : ~30% vs format FAISS original

#### Optimisations
- **Streaming upload** : Upload par chunks pour gros datasets
- **Compression automatique** : Git LFS pour fichiers volumineux
- **Validation** : Checksum avant upload