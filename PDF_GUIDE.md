# 📄 Guide d'utilisation PDF pour LocalRAG

## 🎯 Nouveau Support PDF avec Chunking Intelligent

Votre système RAG local supporte maintenant les **documents PDF** avec un découpage sémantique optimisé pour les documents de plusieurs dizaines de pages.

## ✨ Fonctionnalités Avancées

### 🧠 Chunking Sémantique Intelligent
- **Analyse de similarité** : Regroupement des phrases par cohérence thématique
- **Respect de la structure** : Préservation des sections et sous-sections
- **Chevauchement contextuel** : Maintien du contexte entre chunks consécutifs
- **Taille adaptive** : Découpage intelligent selon le contenu

### 📏 Paramètres Optimisés pour PDFs Longs
```python
# Configuration recommandée pour documents 50+ pages
chunk_size=1000         # Taille cible (caractères)
max_chunk_size=1500     # Taille max avant split forcé
min_chunk_size=300      # Taille min pour éviter micro-chunks
chunk_overlap=100       # Chevauchement entre chunks
```

## 🚀 Utilisation

### 1. Installation des Dépendances
```bash
pip install PyMuPDF sentence-transformers
```

### 2. Ajout de Vos PDFs
```bash
# Placez vos PDFs dans le répertoire docs/
mkdir -p docs/
cp votre_document.pdf docs/
```

### 3. Indexation Automatique
```bash
# Lancez l'indexation (inclut automatiquement les PDFs)
python step01_indexer.py

# Résultat attendu :
# 📁 15 fichiers trouvés :
#    • HTML: 8 fichiers
#    • Markdown: 4 fichiers
#    • PDF: 3 fichiers        ← Nouveau !
```

### 4. Utilisation avec le Chatbot
```bash
# Lancez l'interface de chat
python step03_chatbot.py

# Posez des questions sur le contenu de vos PDFs !
```

## 🔧 Test et Validation

### Script de Test Intégré
```bash
# Test du système PDF complet
python test_pdf_integration.py
```

Le script teste :
- ✅ Parsing des PDFs avec différentes configurations
- ✅ Conversion en chunks compatibles avec le système
- ✅ Intégration avec l'indexeur principal
- ✅ Comparaison des stratégies de chunking

## 📊 Avantages du Chunking Sémantique

### Stratégie Traditionnelle (Taille Fixe)
```
Chunk 1: [Introduction au machine learning...]
Chunk 2: [...learning. Les réseaux de neurones...] ← Coupure au milieu d'une section
Chunk 3: [...sont composés de couches...]
```

### Stratégie Sémantique Intelligente
```
Chunk 1: [Introduction complète au machine learning]
Chunk 2: [Section complète sur les réseaux de neurones] ← Respect de la structure
Chunk 3: [Architecture et composants détaillés]
```

## ⚙️ Configuration Avancée

### Pour Documents Techniques Très Longs (100+ pages)
```python
parser = PDFDocumentParser(
    chunk_size=800,          # Chunks plus petits
    max_chunk_size=1200,     # Limite stricte
    min_chunk_size=250,      # Minimum réduit
    chunk_overlap=200,       # Plus de contexte
    use_semantic_chunking=True
)
```

### Pour Documents Simples (< 20 pages)
```python
parser = PDFDocumentParser(
    chunk_size=1200,         # Chunks plus gros
    max_chunk_size=1800,     # Limite élevée
    min_chunk_size=400,      # Minimum standard
    chunk_overlap=100,       # Contexte normal
    use_semantic_chunking=False  # Mode rapide
)
```

## 📋 Métadonnées Extraites

Chaque chunk PDF contient :
- **Pages sources** : `[1, 2, 3]` - Pages d'origine du contenu
- **Section/sous-section** : Hiérarchie automatiquement détectée
- **Statistiques** : Nombre de mots, caractères
- **Identifiant unique** : Hash pour la déduplication

## 🔍 Exemple d'Utilisation Pratique

```python
# Import direct du parser PDF
from pdf_parser import PDFDocumentParser

# Configuration pour manuel technique de 150 pages
parser = PDFDocumentParser(
    chunk_size=800,
    chunk_overlap=150,
    use_semantic_chunking=True
)

# Traitement du document
chunks = parser.parse_pdf_file(Path("manuel_technique.pdf"))

print(f"📄 {len(chunks)} chunks créés")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}: Pages {chunk.page_numbers}")
    print(f"Section: {chunk.section_title}")
    print(f"Contenu: {chunk.content[:100]}...")
```

## 🎯 Cas d'Usage Optimaux

### ✅ Parfait Pour :
- **Manuels techniques** : Documentation logicielle, APIs
- **Rapports de recherche** : Papers, études, analyses
- **Guides d'installation** : Procédures étape par étape
- **Documentation produit** : Spécifications, architectures
- **Livres techniques** : Programmation, ingénierie

### ⚠️ Limitations :
- **PDFs scannés** : Nécessitent OCR préalable
- **PDFs protégés** : Mot de passe requis
- **Tableaux complexes** : Extraction basique du texte
- **Formules mathématiques** : Rendu en texte simple

## 📈 Performance

### Métriques Typiques (Document 50 pages)
- **Parsing** : ~2-5 secondes
- **Chunking sémantique** : ~3-8 secondes
- **Embedding** : ~10-20 secondes
- **Indexation FAISS** : ~1-2 secondes

### Optimisations Automatiques
- **Cache des embeddings** : Évite le recalcul
- **Chunking adaptatif** : Ajuste selon la complexité
- **Parallélisation** : Traitement multi-thread quand possible

## 🐛 Dépannage

### Erreur "PyMuPDF non trouvé"
```bash
pip install PyMuPDF
# ou
pip install pymupdf
```

### Erreur "Modèle de chunking non disponible"
```bash
pip install sentence-transformers
```

### PDF non reconnu
- Vérifiez que le fichier n'est pas corrompu
- Assurez-vous qu'il n'est pas protégé par mot de passe
- Testez avec un PDF simple d'abord

### Chunks trop petits/grands
Ajustez les paramètres dans `pdf_parser.py` :
```python
DEFAULT_CHUNK_SIZE = 1000  # Augmentez pour chunks plus gros
MIN_CHUNK_SIZE = 300       # Réduisez pour plus de chunks
```

## 📚 Prochaines Améliorations

- [ ] **Extraction d'images** : Analyse des diagrammes et schémas
- [ ] **Reconnaissance OCR** : Support des PDFs scannés
- [ ] **Tables avancées** : Parsing structuré des tableaux
- [ ] **Liens internes** : Navigation entre sections
- [ ] **Métadonnées enrichies** : Auteur, date, version

---

🚀 **Votre système RAG est maintenant compatible PDF avec chunking intelligent !**

Testez avec vos documents techniques et découvrez la puissance du découpage sémantique pour les documents longs.