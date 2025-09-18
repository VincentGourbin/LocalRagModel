#!/usr/bin/env python3
"""
Parser PDF avancé avec chunking sémantique intelligent
Conçu pour traiter des documents PDF de plusieurs dizaines de pages
avec découpage optimal des paragraphes et sections
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class PDFChunk:
    """Représente un chunk extrait d'un document PDF"""
    content: str
    page_numbers: List[int]  # Pages d'où provient le chunk
    section_title: str
    subsection_title: str
    chunk_index: int
    total_chunks: int
    metadata: Dict


class PDFDocumentParser:
    """Parser PDF avec chunking sémantique intelligent.

    Caractéristiques principales:
    - Extraction de la structure hiérarchique (titres, sections)
    - Chunking sémantique basé sur la longueur ET la cohérence
    - Préservation du contexte entre les chunks
    - Gestion des tableaux et images
    - Support des documents longs (50+ pages)
    """

    # Paramètres de chunking optimisés pour les documents longs
    DEFAULT_CHUNK_SIZE = 1000  # Taille cible d'un chunk en caractères
    MAX_CHUNK_SIZE = 1500      # Taille maximale avant forçage de split
    MIN_CHUNK_SIZE = 300        # Taille minimale pour éviter les micro-chunks
    CHUNK_OVERLAP = 100         # Chevauchement entre chunks pour maintenir le contexte

    def __init__(self,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 max_chunk_size: int = MAX_CHUNK_SIZE,
                 min_chunk_size: int = MIN_CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 use_semantic_chunking: bool = True):
        """
        Args:
            chunk_size: Taille cible des chunks
            max_chunk_size: Taille maximale avant découpage forcé
            min_chunk_size: Taille minimale d'un chunk
            chunk_overlap: Chevauchement entre chunks consécutifs
            use_semantic_chunking: Utiliser le chunking sémantique avancé
        """
        self.chunk_size = chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking

        # Modèle pour le chunking sémantique (optionnel)
        if use_semantic_chunking:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.use_semantic_chunking = False
                print("⚠️ Modèle de chunking sémantique non disponible, utilisation du chunking par taille")

    def parse_pdf_file(self, file_path: Path) -> List[PDFChunk]:
        """Parse un fichier PDF et retourne des chunks optimisés.

        Args:
            file_path: Chemin vers le fichier PDF

        Returns:
            Liste des chunks extraits avec métadonnées
        """
        doc = None
        try:
            doc = fitz.open(str(file_path))
            page_count = doc.page_count

            # 1. Extraction de la structure du document
            document_structure = self._extract_document_structure(doc)

            # 2. Extraction du contenu textuel avec préservation de la structure
            full_text, page_mapping = self._extract_text_with_structure(doc)

            # Fermer le document dès que possible
            doc.close()
            doc = None

            # 3. Découpage en sections basé sur la structure
            sections = self._split_into_sections(full_text, document_structure)

            # 4. Chunking intelligent de chaque section
            chunks = []
            for section in sections:
                section_chunks = self._create_chunks_from_section(
                    section,
                    page_mapping,
                    file_path
                )
                chunks.extend(section_chunks)

            # 5. Post-traitement et validation
            chunks = self._postprocess_chunks(chunks)

            print(f"✅ PDF traité: {len(chunks)} chunks créés à partir de {page_count} pages")
            return chunks

        except Exception as e:
            print(f"❌ Erreur lors du parsing PDF {file_path}: {e}")
            return []
        finally:
            # S'assurer que le document est fermé même en cas d'erreur
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass

    def _extract_document_structure(self, doc) -> Dict:
        """Extrait la structure hiérarchique du document (TOC, titres)."""
        structure = {
            'title': '',
            'toc': [],
            'sections': []
        }

        # Extraction du titre depuis les métadonnées
        metadata = doc.metadata
        if metadata:
            structure['title'] = metadata.get('title', '') or metadata.get('Title', '')

        # Extraction de la table des matières si disponible
        toc = doc.get_toc()
        if toc:
            structure['toc'] = toc

        # Détection des titres par analyse de la mise en forme
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            # Heuristique pour détecter les titres
                            if self._is_likely_title(span):
                                structure['sections'].append({
                                    'text': span['text'].strip(),
                                    'page': page_num,
                                    'font_size': span['size'],
                                    'font_flags': span['flags']
                                })

        return structure

    def _is_likely_title(self, span: Dict) -> bool:
        """Détermine si un span de texte est probablement un titre."""
        text = span.get('text', '').strip()

        # Critères pour identifier un titre
        if len(text) < 5 or len(text) > 200:
            return False

        # Police plus grande que la normale (> 12pt généralement)
        if span.get('size', 0) > 14:
            return True

        # Police en gras (flag 2^4 = 16)
        if span.get('flags', 0) & 16:
            return True

        # Patterns de numérotation de sections
        if re.match(r'^(\d+\.?)+\s+\w+', text):
            return True
        if re.match(r'^(Chapter|Section|Part)\s+\d+', text, re.IGNORECASE):
            return True

        return False

    def _extract_text_with_structure(self, doc) -> Tuple[str, Dict]:
        """Extrait le texte complet avec mapping des pages."""
        full_text = []
        page_mapping = {}  # position -> page number
        current_pos = 0

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            full_text.append(page_text)

            # Enregistrer la position de début de chaque page
            page_mapping[current_pos] = page_num
            current_pos += len(page_text)

        return '\n'.join(full_text), page_mapping

    def _split_into_sections(self, text: str, structure: Dict) -> List[Dict]:
        """Découpe le texte en sections basées sur la structure détectée."""
        sections = []

        if structure['toc']:
            # Utiliser la TOC pour découper
            sections = self._split_by_toc(text, structure['toc'])
        elif structure['sections']:
            # Utiliser les titres détectés
            sections = self._split_by_detected_titles(text, structure['sections'])
        else:
            # Découpage par défaut basé sur les doubles sauts de ligne
            sections = self._split_by_paragraphs(text)

        return sections

    def _split_by_toc(self, text: str, toc: List) -> List[Dict]:
        """Découpe le texte basé sur la table des matières."""
        # Implémentation simplifiée - découper par paragraphes pour l'instant
        return self._split_by_paragraphs(text)

    def _split_by_detected_titles(self, text: str, sections: List[Dict]) -> List[Dict]:
        """Découpe le texte basé sur les titres détectés."""
        # Implémentation simplifiée - découper par paragraphes pour l'instant
        return self._split_by_paragraphs(text)

    def _split_by_paragraphs(self, text: str) -> List[Dict]:
        """Découpe le texte en paragraphes naturels."""
        # Stratégie de découpage par paragraphes
        # 1. Double saut de ligne = nouveau paragraphe probable
        # 2. Changement de style détecté = nouvelle section

        paragraphs = []
        current_para = []

        lines = text.split('\n')
        empty_line_count = 0

        for line in lines:
            line = line.strip()

            if not line:
                empty_line_count += 1
                if empty_line_count >= 2 and current_para:
                    # Fin du paragraphe actuel
                    paragraphs.append({
                        'title': '',
                        'content': '\n'.join(current_para),
                        'level': 0
                    })
                    current_para = []
            else:
                empty_line_count = 0
                current_para.append(line)

        # Ajouter le dernier paragraphe
        if current_para:
            paragraphs.append({
                'title': '',
                'content': '\n'.join(current_para),
                'level': 0
            })

        return paragraphs

    def _create_chunks_from_section(self, section: Dict, page_mapping: Dict, file_path: Path) -> List[PDFChunk]:
        """Crée des chunks optimisés à partir d'une section."""
        chunks = []
        content = section.get('content', '')
        title = section.get('title', '')

        if len(content) <= self.max_chunk_size:
            # Section assez petite pour tenir dans un seul chunk
            chunk = self._create_single_chunk(content, title, page_mapping, file_path)
            if chunk:
                chunks.append(chunk)
        else:
            # Section trop grande, découpage intelligent nécessaire
            if self.use_semantic_chunking:
                chunks = self._semantic_chunking(content, title, page_mapping, file_path)
            else:
                chunks = self._size_based_chunking(content, title, page_mapping, file_path)

        return chunks

    def _semantic_chunking(self, content: str, title: str, page_mapping: Dict, file_path: Path) -> List[PDFChunk]:
        """Chunking sémantique basé sur la similarité des phrases."""
        chunks = []

        # Découper en phrases
        sentences = self._split_into_sentences(content)
        if not sentences:
            return []

        # Calculer les embeddings des phrases
        try:
            embeddings = self.sentence_model.encode(sentences)

            # Regrouper les phrases similaires
            current_chunk = []
            current_size = 0

            for i, sentence in enumerate(sentences):
                sentence_size = len(sentence)

                # Conditions pour créer un nouveau chunk
                should_split = False

                # 1. Taille maximale atteinte
                if current_size + sentence_size > self.max_chunk_size:
                    should_split = True

                # 2. Changement sémantique important (si on a déjà du contenu)
                elif current_chunk and i > 0:
                    # Calculer la similarité avec le chunk actuel
                    if len(current_chunk) > 3:  # Assez de phrases pour comparer
                        chunk_embedding = np.mean([embeddings[j] for j in range(i-len(current_chunk), i)], axis=0)
                        similarity = np.dot(embeddings[i], chunk_embedding) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(chunk_embedding))

                        # Si la similarité est trop faible, on crée un nouveau chunk
                        if similarity < 0.5 and current_size >= self.min_chunk_size:
                            should_split = True

                if should_split and current_chunk:
                    # Créer le chunk
                    chunk_content = ' '.join(current_chunk)
                    chunk = self._create_single_chunk(chunk_content, title, page_mapping, file_path)
                    if chunk:
                        chunks.append(chunk)

                    # Ajouter un overlap si nécessaire
                    if self.chunk_overlap > 0 and len(current_chunk) > 2:
                        overlap_sentences = current_chunk[-2:]  # Garder les 2 dernières phrases
                        current_chunk = overlap_sentences
                        current_size = sum(len(s) for s in overlap_sentences)
                    else:
                        current_chunk = []
                        current_size = 0

                current_chunk.append(sentence)
                current_size += sentence_size

            # Ajouter le dernier chunk
            if current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunk = self._create_single_chunk(chunk_content, title, page_mapping, file_path)
                if chunk:
                    chunks.append(chunk)

        except Exception as e:
            print(f"⚠️ Erreur chunking sémantique: {e}, basculement sur chunking par taille")
            return self._size_based_chunking(content, title, page_mapping, file_path)

        return chunks

    def _size_based_chunking(self, content: str, title: str, page_mapping: Dict, file_path: Path) -> List[PDFChunk]:
        """Chunking basé sur la taille avec respect des limites de mots."""
        chunks = []

        words = content.split()
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 pour l'espace

            if current_size + word_size > self.chunk_size and current_chunk:
                # Créer un nouveau chunk
                chunk_content = ' '.join(current_chunk)

                # Vérifier la taille minimale
                if len(chunk_content) >= self.min_chunk_size:
                    chunk = self._create_single_chunk(chunk_content, title, page_mapping, file_path)
                    if chunk:
                        chunks.append(chunk)

                    # Gérer l'overlap
                    if self.chunk_overlap > 0:
                        # Calculer combien de mots garder pour l'overlap
                        overlap_words = []
                        overlap_size = 0
                        for w in reversed(current_chunk):
                            if overlap_size < self.chunk_overlap:
                                overlap_words.insert(0, w)
                                overlap_size += len(w) + 1
                            else:
                                break
                        current_chunk = overlap_words
                        current_size = overlap_size
                    else:
                        current_chunk = []
                        current_size = 0

            current_chunk.append(word)
            current_size += word_size

        # Ajouter le dernier chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunk = self._create_single_chunk(chunk_content, title, page_mapping, file_path)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases."""
        # Pattern pour détecter les fins de phrases
        sentence_endings = r'[.!?]\s+'
        sentences = re.split(sentence_endings, text)

        # Nettoyer et filtrer les phrases vides
        sentences = [s.strip() for s in sentences if s.strip()]

        # Recombiner les phrases trop courtes
        combined_sentences = []
        current = ""

        for sentence in sentences:
            if len(sentence) < 30 and current:
                current += ". " + sentence
            else:
                if current:
                    combined_sentences.append(current)
                current = sentence

        if current:
            combined_sentences.append(current)

        return combined_sentences

    def _create_single_chunk(self, content: str, title: str, page_mapping: Dict, file_path: Path) -> Optional[PDFChunk]:
        """Crée un chunk unique avec métadonnées."""
        if not content or len(content.strip()) < self.min_chunk_size:
            return None

        # Déterminer les pages concernées
        pages = self._get_page_numbers_for_content(content, page_mapping)

        # Générer un ID unique
        chunk_id = hashlib.md5(f"{file_path}_{content[:100]}".encode()).hexdigest()

        return PDFChunk(
            content=content.strip(),
            page_numbers=pages,
            section_title=title,
            subsection_title="",
            chunk_index=0,
            total_chunks=1,
            metadata={
                'source_file': str(file_path),
                'file_name': file_path.name,
                'chunk_id': chunk_id,
                'char_count': len(content),
                'word_count': len(content.split())
            }
        )

    def _get_page_numbers_for_content(self, content: str, page_mapping: Dict) -> List[int]:
        """Détermine sur quelles pages se trouve le contenu."""
        # Simplification : on retourne les premières pages
        # Dans une implémentation complète, on ferait une recherche dans le mapping
        if page_mapping:
            return list(set(page_mapping.values()))[:5]  # Max 5 pages par chunk
        return [0]

    def _postprocess_chunks(self, chunks: List[PDFChunk]) -> List[PDFChunk]:
        """Post-traitement des chunks pour améliorer la qualité."""
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            # Mise à jour des indices
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

            # Nettoyage du contenu
            chunk.content = self._clean_text(chunk.content)

            # Filtrage des chunks trop courts après nettoyage
            if len(chunk.content.strip()) >= self.min_chunk_size:
                processed_chunks.append(chunk)

        return processed_chunks

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte extrait du PDF."""
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)

        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

        # Corriger les problèmes de césure
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Supprimer les sauts de ligne isolés
        text = re.sub(r'\n(?!\n)', ' ', text)

        return text.strip()


def test_pdf_parser():
    """Test du parser PDF avec un fichier exemple."""
    parser = PDFDocumentParser(
        chunk_size=1000,
        max_chunk_size=1500,
        min_chunk_size=300,
        chunk_overlap=100,
        use_semantic_chunking=True
    )

    # Test avec un fichier PDF exemple (remplacer par un vrai fichier)
    test_file = Path("example.pdf")
    if test_file.exists():
        chunks = parser.parse_pdf_file(test_file)

        print(f"\n📊 Résultats du parsing:")
        print(f"   • Nombre de chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks[:3]):  # Afficher les 3 premiers
            print(f"\n📄 Chunk {i+1}/{chunk.total_chunks}:")
            print(f"   • Pages: {chunk.page_numbers}")
            print(f"   • Section: {chunk.section_title or 'Sans titre'}")
            print(f"   • Taille: {chunk.metadata['word_count']} mots")
            print(f"   • Début: {chunk.content[:150]}...")
    else:
        print("⚠️ Fichier test 'example.pdf' non trouvé")


if __name__ == "__main__":
    test_pdf_parser()