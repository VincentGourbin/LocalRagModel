#!/usr/bin/env python3
"""
Indexeur FAISS pour remplacer ChromaDB - Plus rapide et sans blocage
"""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time


class FAISSIndexer:
    """Gestionnaire d'index vectoriel utilisant FAISS au lieu de ChromaDB.
    
    FAISS est significativement plus rapide que ChromaDB pour :
    - L'ajout de vecteurs (pas de blocage)
    - La recherche de similarité
    - La gestion de grandes collections
    
    Attributes:
        index_path: Chemin vers les fichiers d'index FAISS
        dimension: Dimension des vecteurs d'embedding
        index: Index FAISS pour la recherche vectorielle
        metadata: Dictionnaire des métadonnées par ID
        id_to_idx: Mapping des IDs vers les indices FAISS
        idx_to_id: Mapping inverse des indices vers les IDs
    """
    
    def __init__(self, index_path: str = "./faiss_index", dimension: int = 2560):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.metadata = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.vectors_added = 0
        
        # Créer le répertoire d'index si nécessaire
        self.index_path.mkdir(exist_ok=True)
        
        # Charger ou créer l'index
        self._initialize_index()
        
        print(f"✅ FAISS indexer initialisé")
        print(f"   - Dimension: {self.dimension}")
        print(f"   - Index path: {self.index_path}")
        print(f"   - Vecteurs existants: {self.index.ntotal if self.index else 0}")
    
    def _initialize_index(self):
        """Initialise ou charge l'index FAISS existant."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.json"
        mapping_file = self.index_path / "mappings.pkl"
        
        if index_file.exists() and metadata_file.exists():
            # Charger l'index existant
            print("📂 Chargement de l'index FAISS existant...")
            try:
                self.index = faiss.read_index(str(index_file))
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                if mapping_file.exists():
                    with open(mapping_file, 'rb') as f:
                        mappings = pickle.load(f)
                        self.id_to_idx = mappings['id_to_idx']
                        self.idx_to_id = mappings['idx_to_id']
                
                print(f"✅ Index chargé: {self.index.ntotal} vecteurs")
                
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement, création nouvel index: {e}")
                self._create_new_index()
        else:
            # Créer un nouvel index
            self._create_new_index()
    
    def _create_new_index(self):
        """Crée un nouvel index FAISS."""
        # Index HNSW pour la recherche rapide de similarité
        # HNSW est optimal pour la recherche de voisins proches
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections
        self.index.hnsw.efConstruction = 200  # Qualité de construction
        
        print(f"🆕 Nouvel index FAISS créé (HNSW)")
    
    def add_vectors(self, embeddings: List[List[float]], ids: List[str], metadatas: List[Dict]):
        """Ajoute des vecteurs à l'index FAISS.
        
        Cette méthode est beaucoup plus rapide que ChromaDB.add()
        
        Args:
            embeddings: Liste des vecteurs d'embedding
            ids: Liste des identifiants uniques
            metadatas: Liste des métadonnées associées
        """
        if not embeddings or not ids:
            return
            
        print(f"📥 Ajout de {len(embeddings)} vecteurs à FAISS...")
        start_time = time.time()
        
        # Conversion en numpy array pour FAISS
        vectors = np.array(embeddings, dtype=np.float32)
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Dimension incorrecte: {vectors.shape[1]} != {self.dimension}")
        
        # Obtenir les indices de départ
        start_idx = self.index.ntotal
        
        # Ajout à l'index FAISS (très rapide !)
        self.index.add(vectors)
        
        # Mise à jour des mappings et métadonnées
        for i, (id_, metadata) in enumerate(zip(ids, metadatas)):
            idx = start_idx + i
            self.id_to_idx[id_] = idx
            self.idx_to_id[idx] = id_
            self.metadata[id_] = metadata
        
        self.vectors_added += len(embeddings)
        duration = time.time() - start_time
        
        print(f"✅ Ajout FAISS terminé en {duration:.2f}s")
        print(f"📊 Total vecteurs: {self.index.ntotal}")
        
        # Sauvegarde immédiate
        self._save_index()
    
    def search(self, query_embedding: List[float], k: int = 10) -> List[Dict]:
        """Recherche les k vecteurs les plus similaires.
        
        Args:
            query_embedding: Vecteur de requête
            k: Nombre de résultats à retourner
            
        Returns:
            Liste des résultats avec scores et métadonnées
        """
        if self.index.ntotal == 0:
            return []
        
        # Conversion en numpy
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Recherche FAISS (très rapide !)
        start_time = time.time()
        distances, indices = self.index.search(query_vector, k)
        duration = time.time() - start_time
        
        print(f"🔍 Recherche FAISS terminée en {duration:.3f}s")
        
        # Construction des résultats
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Pas de résultat
                continue
                
            id_ = self.idx_to_id.get(idx)
            if id_ and id_ in self.metadata:
                result = {
                    'id': id_,
                    'distance': float(distance),
                    'similarity': 1 / (1 + distance),  # Conversion distance->similarité
                    'rank': i + 1,
                    'metadata': self.metadata[id_]
                }
                results.append(result)
        
        return results
    
    def _save_index(self):
        """Sauvegarde l'index et métadonnées sur disque."""
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.json"
            mapping_file = self.index_path / "mappings.pkl"
            
            # Sauvegarde index FAISS
            faiss.write_index(self.index, str(index_file))
            
            # Sauvegarde métadonnées JSON
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Sauvegarde mappings
            mappings = {
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }
            with open(mapping_file, 'wb') as f:
                pickle.dump(mappings, f)
            
            print(f"💾 Index FAISS sauvegardé")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques de l'index."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'vectors_added_session': self.vectors_added,
            'index_type': 'FAISS-HNSW',
            'metadata_count': len(self.metadata)
        }
    
    def count(self) -> int:
        """Retourne le nombre de vecteurs dans l'index."""
        return self.index.ntotal if self.index else 0


def test_faiss_performance():
    """Test de performance FAISS vs ChromaDB concept."""
    print("🧪 Test de performance FAISS")
    
    # Créer un indexeur test
    indexer = FAISSIndexer("./test_faiss", dimension=2560)
    
    # Générer des données de test
    test_embeddings = np.random.rand(100, 2560).astype(np.float32).tolist()
    test_ids = [f"test_{i}" for i in range(100)]
    test_metadata = [{'content': f'Test document {i}', 'length': i*10} for i in range(100)]
    
    # Test d'ajout
    print("\n📥 Test ajout 100 vecteurs...")
    start = time.time()
    indexer.add_vectors(test_embeddings, test_ids, test_metadata)
    add_duration = time.time() - start
    print(f"⚡ Ajout terminé en {add_duration:.2f}s ({len(test_embeddings)/add_duration:.1f} vecteurs/s)")
    
    # Test de recherche
    print("\n🔍 Test recherche...")
    query = test_embeddings[0]  # Utiliser le premier comme requête
    start = time.time()
    results = indexer.search(query, k=10)
    search_duration = time.time() - start
    print(f"⚡ Recherche terminée en {search_duration:.3f}s")
    print(f"📊 {len(results)} résultats trouvés")
    
    # Nettoyage
    import shutil
    shutil.rmtree("./test_faiss", ignore_errors=True)
    
    print(f"\n✅ Test terminé - FAISS est prêt à remplacer ChromaDB!")


if __name__ == "__main__":
    test_faiss_performance()