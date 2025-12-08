"""
ScArlet-Sails Vector Store v2.0

FAISS-based semantic search for trading patterns.
Supports Multi-HyDE query expansion.

Key features:
- Semantic pattern representation (not just numbers)
- Incremental index updates
- W_box quality filtering
- Multi-perspective embeddings
"""

import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Sentence-BERT для embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("⚠️ sentence-transformers not installed. Using fallback.")


class PatternVectorStore:
    """
    Vector database for trading patterns.
    
    Uses FAISS for fast similarity search and
    Sentence-BERT for semantic embeddings.
    """
    
    # Embedding model - small but effective
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(
        self,
        patterns_dir: str = "rag/patterns",
        model_name: str = None,
        min_w_box: float = 0.0,  # Minimum quality threshold
    ):
        """
        Initialize vector store.
        
        Args:
            patterns_dir: Directory with JSON patterns
            model_name: Sentence-BERT model name
            min_w_box: Minimum W_box score to include (0.0 = all)
        """
        self.patterns_dir = Path(patterns_dir)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.min_w_box = min_w_box
        
        # File paths
        self.index_file = self.patterns_dir / "embeddings.faiss"
        self.metadata_file = self.patterns_dir / "metadata.pkl"
        self.config_file = self.patterns_dir / "index_config.json"
        
        # Initialize embedding model
        self.model = None
        if HAS_SBERT:
            print(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.pattern_ids: set = set()  # For duplicate detection
        
        # Load existing index if available
        if self.index_file.exists():
            self.load()
        else:
            self._init_index()
    
    def _init_index(self):
        """Initialize empty FAISS index."""
        # IndexFlatIP = Inner Product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.metadata = []
        self.pattern_ids = set()
        print(f"🔧 Created FAISS index (dim={self.EMBEDDING_DIM})")
    
    # =========================================================================
    # PATTERN TO TEXT CONVERSION (Multi-Perspective)
    # =========================================================================
    
    def pattern_to_text(self, pattern: dict) -> str:
        """
        Convert pattern to semantic text representation.
        
        Uses MULTIPLE PERSPECTIVES for richer embeddings:
        1. Technical perspective (indicators)
        2. Quality perspective (W_box components)
        3. Context perspective (coin, timeframe, session)
        
        This is key for Multi-HyDE - we embed MEANING not numbers.
        """
        perspectives = []
        
        # Extract components
        meta = pattern.get('meta', {})
        ind = pattern.get('indicators_before', {})
        box = pattern.get('box', {})
        w_box = pattern.get('w_box', {})
        
        # === PERSPECTIVE 1: Technical Setup ===
        tech_parts = []
        
        # Asset context
        tech_parts.append(f"{meta.get('coin', 'BTC')} {meta.get('timeframe', '1h')}")
        tech_parts.append(f"{meta.get('direction', 'long')} setup")
        
        # RSI state
        rsi_z = ind.get('rsi_zscore', 0) or 0
        if ind.get('rsi_low') or rsi_z < -1:
            tech_parts.append("RSI oversold")
        elif ind.get('rsi_high') or rsi_z > 1:
            tech_parts.append("RSI overbought")
        else:
            tech_parts.append("RSI neutral")
        
        # Trend
        if ind.get('trend_up'):
            tech_parts.append("uptrend")
        elif ind.get('trend_down'):
            tech_parts.append("downtrend")
        else:
            tech_parts.append("ranging")
        
        # Volume
        vol_z = ind.get('volume_zscore', 0) or 0
        if vol_z > 1.5:
            tech_parts.append("very high volume")
        elif vol_z > 0.5:
            tech_parts.append("above average volume")
        elif vol_z < -0.5:
            tech_parts.append("low volume")
        
        # Divergence
        if ind.get('div_rsi_bullish'):
            tech_parts.append("bullish RSI divergence")
        elif ind.get('div_rsi_bearish'):
            tech_parts.append("bearish RSI divergence")
        
        perspectives.append(" | ".join(tech_parts))
        
        # === PERSPECTIVE 2: Box Quality ===
        quality_parts = []
        
        # Box metrics
        ts = box.get('touches_support', 0) or 0
        tr = box.get('touches_resistance', 0) or 0
        total_touches = ts + tr
        
        if total_touches >= 6:
            quality_parts.append("strong box multiple touches")
        elif total_touches >= 4:
            quality_parts.append("moderate box")
        elif total_touches >= 2:
            quality_parts.append("weak box few touches")
        
        # Box range
        box_range = box.get('box_range_pct', 0) or 0
        if box_range > 5:
            quality_parts.append("wide range")
        elif box_range < 2:
            quality_parts.append("tight range")
        
        # W_box score
        w_score = w_box.get('W_box', 0) or 0
        if w_score > 0.7:
            quality_parts.append("high quality setup")
        elif w_score > 0.4:
            quality_parts.append("medium quality")
        elif w_score > 0:
            quality_parts.append("low quality setup")
        
        if quality_parts:
            perspectives.append(" | ".join(quality_parts))
        
        # === PERSPECTIVE 3: Market Context ===
        context_parts = []
        
        # Volatility
        if ind.get('vol_low'):
            context_parts.append("low volatility environment")
        elif ind.get('vol_high'):
            context_parts.append("high volatility environment")
        
        # Session
        if ind.get('session_asian') or ind.get('time_asian'):
            context_parts.append("Asian session")
        elif ind.get('session_european') or ind.get('time_european'):
            context_parts.append("European session")
        elif ind.get('session_american') or ind.get('time_american'):
            context_parts.append("American session")
        
        if context_parts:
            perspectives.append(" | ".join(context_parts))
        
        # Combine all perspectives
        return " || ".join(perspectives)
    
    def state_to_text(self, state: dict) -> str:
        """
        Convert current market state S(t) to text for search.
        
        Args:
            state: Current state from Council/FeatureEngine
            
        Returns:
            Text representation for embedding
        """
        # Wrap state into pattern-like structure
        pseudo_pattern = {
            'meta': {
                'coin': state.get('symbol', state.get('coin', 'BTC')),
                'timeframe': state.get('timeframe', '1h'),
                'direction': state.get('direction', 'long'),
            },
            'indicators_before': state.get('indicators', state),
            'box': state.get('box', {}),
            'w_box': state.get('w_box', {}),
        }
        return self.pattern_to_text(pseudo_pattern)
    
    # =========================================================================
    # INDEX OPERATIONS
    # =========================================================================
    
    def add_pattern(self, pattern_file: Path, verbose: bool = False) -> bool:
        """
        Add single pattern to index.
        
        Args:
            pattern_file: Path to JSON file
            verbose: Print progress
            
        Returns:
            True if added successfully
        """
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                pattern = json.load(f)
            
            pattern_id = pattern.get('id', pattern_file.stem)
            
            # Skip duplicates
            if pattern_id in self.pattern_ids:
                if verbose:
                    print(f"⏭️ Skip duplicate: {pattern_id}")
                return False
            
            # Check W_box threshold
            w_box_score = pattern.get('w_box', {}).get('W_box', 0) or 0
            if w_box_score < self.min_w_box:
                if verbose:
                    print(f"⏭️ Skip low quality (W_box={w_box_score:.2f}): {pattern_id}")
                return False
            
            # Convert to text
            text = self.pattern_to_text(pattern)
            
            # Generate embedding
            if self.model is None:
                return False
            
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            embedding = np.array([embedding], dtype=np.float32)
            
            # Add to FAISS
            self.index.add(embedding)
            
            # Store metadata
            self.metadata.append({
                'pattern_id': pattern_id,
                'file': str(pattern_file),
                'text_repr': text,
                'coin': pattern.get('meta', {}).get('coin'),
                'timeframe': pattern.get('meta', {}).get('timeframe'),
                'direction': pattern.get('meta', {}).get('direction'),
                'w_box': w_box_score,
                'created_at': pattern.get('created_at'),
            })
            
            self.pattern_ids.add(pattern_id)
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Error adding {pattern_file.name}: {e}")
            return False
    
    def build_from_directory(self, verbose: bool = True) -> int:
        """
        Build index from all patterns in directory.
        
        Args:
            verbose: Show progress bar
            
        Returns:
            Number of patterns indexed
        """
        # Find all JSON pattern files
        pattern_files = [
            f for f in self.patterns_dir.glob("*.json")
            if f.name not in ['library.json', 'outcomes.json', 'index_config.json']
        ]
        
        if verbose:
            print(f"📦 Found {len(pattern_files)} pattern files")
        
        # Reset index
        self._init_index()
        
        # Add patterns
        success_count = 0
        iterator = tqdm(pattern_files, desc="Indexing") if verbose else pattern_files
        
        for pf in iterator:
            if self.add_pattern(pf, verbose=False):
                success_count += 1
        
        # Save index
        self.save()
        
        if verbose:
            print(f"✅ Indexed {success_count}/{len(pattern_files)} patterns")
            if self.min_w_box > 0:
                print(f"   (W_box threshold: {self.min_w_box})")
        
        return success_count
    
    def update_index(self, new_patterns: List[Path] = None, verbose: bool = True) -> int:
        """
        Incrementally update index with new patterns.
        
        Args:
            new_patterns: Specific files to add, or None to scan directory
            verbose: Print progress
            
        Returns:
            Number of new patterns added
        """
        if new_patterns is None:
            # Find files not in index
            all_files = set(
                f for f in self.patterns_dir.glob("*.json")
                if f.name not in ['library.json', 'outcomes.json', 'index_config.json']
            )
            indexed_files = set(Path(m['file']) for m in self.metadata)
            new_patterns = list(all_files - indexed_files)
        
        if not new_patterns:
            if verbose:
                print("📭 No new patterns to index")
            return 0
        
        if verbose:
            print(f"📥 Adding {len(new_patterns)} new patterns")
        
        added = 0
        for pf in new_patterns:
            if self.add_pattern(pf, verbose=verbose):
                added += 1
        
        if added > 0:
            self.save()
        
        if verbose:
            print(f"✅ Added {added} patterns")
        
        return added
    
    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    
    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filter_coin: str = None,
        filter_timeframe: str = None,
        filter_direction: str = None,
        min_w_box: float = None,
    ) -> List[Dict]:
        """
        Search for similar patterns.
        
        Args:
            query_text: Text query (from state_to_text or manual)
            top_k: Number of results
            filter_coin: Filter by coin (BTC, ETH, ...)
            filter_timeframe: Filter by timeframe (15m, 1h, ...)
            filter_direction: Filter by direction (long, short)
            min_w_box: Minimum W_box score
            
        Returns:
            List of similar patterns with similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        if self.model is None:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search more if filtering
        has_filter = any([filter_coin, filter_timeframe, filter_direction, min_w_box])
        search_k = min(top_k * 5 if has_filter else top_k, self.index.ntotal)
        
        # FAISS search (inner product = cosine similarity for normalized vectors)
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Build results
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            # Apply filters
            if filter_coin and meta.get('coin') != filter_coin:
                continue
            if filter_timeframe and meta.get('timeframe') != filter_timeframe:
                continue
            if filter_direction and meta.get('direction') != filter_direction:
                continue
            if min_w_box and (meta.get('w_box', 0) or 0) < min_w_box:
                continue
            
            # Load full pattern
            try:
                with open(meta['file'], 'r', encoding='utf-8') as f:
                    pattern = json.load(f)
            except:
                continue
            
            results.append({
                'similarity': float(score),
                'pattern_id': meta['pattern_id'],
                'pattern': pattern,
                'text_repr': meta['text_repr'],
                'coin': meta.get('coin'),
                'timeframe': meta.get('timeframe'),
                'w_box': meta.get('w_box'),
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_state(
        self,
        state: dict,
        top_k: int = 5,
        **filters
    ) -> List[Dict]:
        """
        Search by market state.
        
        Args:
            state: Current market state S(t)
            top_k: Number of results
            **filters: Passed to search()
            
        Returns:
            Similar patterns
        """
        query_text = self.state_to_text(state)
        return self.search(query_text, top_k=top_k, **filters)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'pattern_ids': self.pattern_ids,
            }, f)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.EMBEDDING_DIM,
            'min_w_box': self.min_w_box,
            'total_patterns': len(self.metadata),
            'updated_at': datetime.now().isoformat(),
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Index saved: {self.index.ntotal} patterns")
    
    def load(self):
        """Load index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_file))
        
        # Load metadata
        with open(self.metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.pattern_ids = data.get('pattern_ids', set())
        
        print(f"📂 Index loaded: {self.index.ntotal} patterns")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if not self.metadata:
            return {'total': 0}
        
        coins = {}
        timeframes = {}
        w_box_scores = []
        
        for m in self.metadata:
            coin = m.get('coin', 'unknown')
            tf = m.get('timeframe', 'unknown')
            w = m.get('w_box', 0)
            
            coins[coin] = coins.get(coin, 0) + 1
            timeframes[tf] = timeframes.get(tf, 0) + 1
            if w:
                w_box_scores.append(w)
        
        return {
            'total': len(self.metadata),
            'by_coin': coins,
            'by_timeframe': timeframes,
            'w_box_avg': np.mean(w_box_scores) if w_box_scores else 0,
            'w_box_min': min(w_box_scores) if w_box_scores else 0,
            'w_box_max': max(w_box_scores) if w_box_scores else 0,
        }
