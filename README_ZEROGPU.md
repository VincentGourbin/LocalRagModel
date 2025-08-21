# 🚀 Déploiement ZeroGPU - LocalRAG Step 03

## Vue d'ensemble

Ce guide explique comment déployer l'interface LocalRAG Step 03 sur un Hugging Face Space avec ZeroGPU pour bénéficier d'un GPU gratuit.

## 📋 Prérequis

### 1. Comptes requis
- **Compte Hugging Face** avec accès ZeroGPU
- **Repository HF** contenant vos embeddings (Step 02 complété)

### 2. Limitations ZeroGPU
- **Comptes Personal PRO** : Maximum 10 Spaces ZeroGPU
- **Comptes Enterprise** : Maximum 50 Spaces ZeroGPU
- **GPU** : NVIDIA H200 slice (70GB VRAM)

## 🛠️ Configuration Space

### 1. Fichiers requis

#### `app.py` (point d'entrée)
```python
#!/usr/bin/env python3
import sys
import os

# Configuration pour ZeroGPU
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

# Lancer l'application
if __name__ == "__main__":
    from step03_chatbot import main
    main()
```

#### `requirements.txt`
Utiliser `requirements_zerogpu.txt` fourni

#### `README.md` du Space
```markdown
---
title: LocalRAG Chat
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
hardware: zero-gpu
---

# LocalRAG Chat Interface

Interface de chat RAG utilisant des embeddings pré-calculés et les modèles Qwen3.
```

### 2. Structure des fichiers
```
your-space/
├── app.py              # Point d'entrée
├── step03_chatbot.py   # Code principal (copié)
├── requirements.txt    # Dépendances ZeroGPU
├── README.md          # Configuration Space
└── step03_config.json # Configuration (à créer)
```

## ⚙️ Configuration

### 1. Créer `step03_config.json`

Pour un Space ZeroGPU, créez manuellement le fichier de configuration :

```json
{
  "step02_completed": true,
  "completion_timestamp": "2024-01-01T00:00:00.000000",
  "huggingface": {
    "repo_id": "VOTRE_USERNAME/VOTRE_REPO",
    "dataset_name": "VOTRE_DATASET",
    "repo_type": "dataset",
    "files": {
      "embeddings": "VOTRE_DATASET.safetensors",
      "metadata": "VOTRE_DATASET_metadata.json",
      "readme": "README.md"
    }
  },
  "embeddings_info": {
    "total_vectors": 1529,
    "vector_dimension": 2560,
    "embedding_model": "Qwen/Qwen3-Embedding-4B",
    "index_type": "faiss_flat",
    "created_at": "2024-01-01T00:00:00.000000"
  }
}
```

### 2. Variables d'environnement (optionnel)

Dans les **Settings** de votre Space :
```
HF_TOKEN=hf_votre_token_si_repo_prive
TRANSFORMERS_CACHE=/tmp/transformers_cache
```

## 🚀 Déploiement

### Étape 1: Créer le Space
1. Aller sur https://huggingface.co/spaces
2. **New Space** → **ZeroGPU** hardware
3. Copier les fichiers dans le Space

### Étape 2: Configuration
1. Modifier `step03_config.json` avec vos paramètres
2. Vérifier que votre repository d'embeddings est accessible
3. **Build** automatique du Space

### Étape 3: Test
- Le Space se lancera automatiquement
- Interface disponible sous l'URL de votre Space
- GPU alloué automatiquement lors des inférences

## 🔧 Optimisations ZeroGPU

### 1. Décorateurs appliqués
```python
@spaces.GPU(duration=60)   # Reranking
def rerank(self, ...):

@spaces.GPU(duration=120)  # Génération
def generate_response(self, ...):
```

### 2. Gestion mémoire
- Cache optimisé pour `/tmp`
- Cleanup automatique des modèles
- Chargement à la demande

### 3. Performance
- **Recherche** : CPU (instantané)
- **Reranking** : GPU (60s max)
- **Génération** : GPU (120s max)

## 📊 Monitoring

### Logs d'activité
```bash
🚀 ZeroGPU détecté - activation du support
🚀 Environnement ZeroGPU détecté - optimisations cloud
🚀 Index FAISS optimisé pour ZeroGPU (IndexHNSWFlat)
```

### Métriques disponibles
- Temps d'allocation GPU
- Utilisation mémoire
- Nombre de requêtes

## 🐛 Dépannage

### Problèmes courants

1. **"No GPU available"**
   - Vérifier que le hardware est configuré sur `zero-gpu`
   - Quota ZeroGPU peut être atteint

2. **Timeout GPU**
   - Réduire la durée des décorateurs
   - Optimiser la taille des modèles

3. **Mémoire insuffisante**
   - Utiliser des modèles plus petits
   - Implémenter le cleanup mémoire

### Support
- Documentation : https://huggingface.co/docs/hub/en/spaces-zerogpu
- Communauté : https://huggingface.co/spaces/zero-gpu-explorers/README/discussions

## 💡 Conseils

1. **Modèles plus petits** : Considérer Qwen3-1.5B pour des réponses plus rapides
2. **Cache intelligent** : Conserver les embeddings en mémoire
3. **Batch processing** : Grouper les requêtes similaires
4. **Monitoring** : Surveiller l'usage GPU pour optimiser

---

🎯 **Résultat** : Interface RAG complète accessible publiquement avec GPU gratuit !