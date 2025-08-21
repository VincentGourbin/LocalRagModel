# 🔌 Guide MCP - LocalRAG Step 03

## Vue d'ensemble

LocalRAG Step 03 est maintenant un **serveur MCP (Model Control Protocol)** complet, exposant le système RAG comme outil utilisable dans Claude Desktop, VS Code, ou d'autres clients MCP compatibles.

L'application Gradio fonctionne simultanément comme :
- **Interface web** : `http://localhost:7860`
- **Serveur MCP** : `http://localhost:7860/gradio_api/mcp/sse`

## 🚀 Fonctionnalités MCP

### Fonction exposée : `ask_rag_question`

```python
def ask_rag_question(question: str, num_documents: int = 3, use_reranking: bool = True) -> str
```

**Paramètres :**
- `question` (str) : La question à poser au système RAG
- `num_documents` (int) : Nombre de documents à utiliser (1-10, défaut: 3)
- `use_reranking` (bool) : Utiliser le reranking Qwen3 (défaut: True)

**Retour :**
- Réponse générée avec sources détaillées et scores

## 🛠️ Installation

### 1. Dépendances
```bash
pip install "gradio[mcp]">=4.0.0
```

### 2. Vérification du support MCP
```python
import gradio as gr
# Si pas d'erreur, MCP est supporté
```

## 🚀 Utilisation

### Mode HTTP (par défaut)
```bash
python step03_chatbot.py
```

Lance l'application avec :
- Interface web de chat : `http://localhost:7860`
- Serveur MCP : `http://localhost:7860/gradio_api/mcp/sse`

### Mode HTTPS (recommandé pour Claude Desktop)
```bash
# Générer les certificats SSL
python step03_ssl_generator_optional.py

# Configurer les variables d'environnement
export SSL_KEYFILE="$(pwd)/ssl_certs/localhost.key"
export SSL_CERTFILE="$(pwd)/ssl_certs/localhost.crt"

# Lancer l'application
python step03_chatbot.py
```

Lance l'application avec :
- Interface web de chat : `https://localhost:7860`
- Serveur MCP : `https://localhost:7860/gradio_api/mcp/sse`

## 🔧 Configuration Claude Desktop

### 1. Fichier de configuration
Créer/modifier `~/Library/Application Support/Claude/claude_desktop_config.json` :

**Configuration HTTPS (recommandée) :**
```json
{
  "mcpServers": {
    "localrag": {
      "command": "python",
      "args": ["/path/to/LocalRagModel/step03_chatbot.py"],
      "env": {
        "SSL_KEYFILE": "/path/to/LocalRagModel/ssl_certs/localhost.key",
        "SSL_CERTFILE": "/path/to/LocalRagModel/ssl_certs/localhost.crt",
        "PYTHONPATH": "/path/to/LocalRagModel"
      }
    }
  }
}
```

**Configuration HTTP (fallback) :**
```json
{
  "mcpServers": {
    "localrag": {
      "command": "python",
      "args": ["/path/to/LocalRagModel/step03_chatbot.py"],
      "env": {
        "PYTHONPATH": "/path/to/LocalRagModel"
      }
    }
  }
}
```

### 2. Redémarrer Claude Desktop

Après la configuration, redémarrer Claude Desktop pour charger le serveur MCP.

## 📝 Exemples d'utilisation

### Dans Claude Desktop

Une fois configuré, vous pouvez utiliser des prompts comme :

```
@localrag Qu'est-ce que Swift MLX et quelles sont ses principales fonctionnalités?
```

```
@localrag Comment installer MLX sur macOS? (utilise 5 documents)
```

### Réponse type

```
Swift MLX est un framework de machine learning développé par Apple pour optimiser...

📚 Documents sources utilisés (avec reranking Qwen3):

• [1] Introduction to Swift MLX (Embedding: 0.945 | Reranking: 0.923)
  └ Source: swift_mlx_intro.html

• [2] Swift MLX Architecture (Embedding: 0.891 | Reranking: 0.887)
  └ Source: swift_mlx_architecture.html

• [3] MLX Performance Guide (#3→#1)
  └ Source: mlx_performance.html
```

## 🔌 API MCP complète

### Fonction principale
- **Nom** : `ask_rag_question`
- **Description** : Pose une question au système RAG LocalRAG
- **Type** : Tool (outil MCP)

### Paramètres détaillés

| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `question` | string | - | requis | Question en langage naturel |
| `num_documents` | integer | 1-10 | 3 | Nombre de documents pour la réponse |
| `use_reranking` | boolean | - | true | Activation du reranking Qwen3 |

### Format de réponse

```
[RÉPONSE GÉNÉRÉE]

📚 Documents sources utilisés ([avec/sans] reranking Qwen3):

• [1] [Titre du document] [(changement de rang optionnel)]
  └ [Scores détaillés]
  └ Source: [fichier source]

• [2] [Titre du document 2]
  └ [Scores détaillés]
  └ Source: [fichier source 2]

[...]
```

## 🎯 Avantages MCP

### Pour les développeurs
- **Intégration directe** dans IDE et éditeurs
- **API standardisée** pour l'accès aux connaissances
- **Pas d'interface web** nécessaire

### Pour les utilisateurs
- **Accès contextuel** aux documents depuis n'importe quelle application
- **Recherche intelligente** avec reranking automatique
- **Sources détaillées** avec scores de pertinence

## 🐛 Dépannage

### "MCP server not found"
```bash
# Vérifier le chemin dans claude_desktop_config.json
which python
# Utiliser le chemin complet
```

### "Module not found"
```bash
# Vérifier l'environnement Python
pip list | grep gradio
pip install "gradio[mcp]"
```

### "RAG system initialization failed"
```bash
# Vérifier que step03_config.json existe
ls step03_config.json
# Vérifier l'accès aux embeddings HF
```

### Logs de débogage
```bash
# Lancer en mode debug
export GRADIO_DEBUG=1
python step03_mcp_server.py
```

## ⚡ Optimisations

### Performance
- **Cache automatique** : Les modèles restent en mémoire
- **Lazy loading** : Chargement à la demande
- **Batch processing** : Optimisation des requêtes multiples

### Mémoire
- **Cleanup automatique** : Libération périodique
- **Seuils configurables** : Adaptation selon RAM disponible
- **Fallback CPU** : Si GPU indisponible

## 🌐 Intégrations compatibles

### Applications testées
- ✅ **Claude Desktop** (recommandé)
- ✅ **VS Code** avec extensions MCP
- ✅ **Cursor IDE**
- 🔄 **Autres clients MCP** (à tester)

### Serveurs MCP
- **Local** : Serveur local sur machine
- **Remote** : Via tunnel/proxy (configuration avancée)
- **Docker** : Conteneurisé (à implémenter)

## 📊 Monitoring

### Logs d'activité
```
🔍 Question MCP: Qu'est-ce que Swift MLX?
📊 Paramètres: 3 documents, reranking: True
🔍 Recherche en deux étapes: 20 candidats → reranking → 3 finaux
✅ Réponse MCP générée (3 documents utilisés)
```

### Métriques disponibles
- Nombre de requêtes MCP
- Temps de réponse moyen
- Usage GPU/CPU
- Cache hit ratio

---

🎯 **Résultat** : Système RAG intégré directement dans vos outils de développement via MCP !