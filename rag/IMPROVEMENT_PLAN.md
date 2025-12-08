# 🛣️ RAG IMPROVEMENT ROADMAP

**Цель:** Превратить базовую RAG v1.0 в production-ready Multi-HyDE систему v2.0

---

## 📊 STATUS OVERVIEW

### Current State (v1.0)
```
✅ PatternExtractor (extractor.py)     - 10/10 Production-ready
✅ Config (config.py)                  - 9/10  Well organized
⚠️ RAGRetriever (retriever.py)        - 6/10  Works but primitive
✅ CLI (cli.py)                        - 8/10  User-friendly
❌ Data (patterns/library.json)       - 0/10  Empty
```

### Target State (v2.0)
```
✅ PatternExtractor                   - KEEP AS IS
✅ Config                             - KEEP AS IS
🆕 VectorStore (vector_store.py)     - CREATE NEW
🆕 MultiHyDE (multi_hyde.py)          - CREATE NEW
🔄 RAGRetriever                       - UPGRADE
🔄 AutoPopulator (auto_populator.py)  - CREATE NEW
📊 Data                               - POPULATE
```

---

## 📅 TIMELINE

| Phase | Duration | Priority | Tasks |
|-------|----------|----------|-------|
| **Phase 0: Data** | 1-2 days | 🔴 CRITICAL | Populate library.json |
| **Phase 1: Foundation** | 3-4 days | 🔴 HIGH | Vector DB + basic search |
| **Phase 2: Intelligence** | 4-5 days | 🟡 MEDIUM | Multi-HyDE + reranking |
| **Phase 3: Automation** | 3-4 days | 🟢 LOW | Auto-population + learning |

**Total: 11-15 days**

---

## 💡 PHASE 0: DATA POPULATION (CRITICAL!)

### 🎯 Objective
Заполнить RAG данными для тестирования и работы Council.

### ⚠️ Why Critical?
- **Council получает пустой контекст** → нет win_rate, нет recommendations
- **Невозможно тестировать retrieval** → нечего искать
- **Нет статистики** → нет confidence scores

### 🛠️ Implementation

**File:** `scripts/populate_rag.py` (NEW)

```python
"""
Populate RAG with historical patterns.

Использование:
    python scripts/populate_rag.py --coin BTC --timeframe 1h --count 50
    python scripts/populate_rag.py --all-coins --timeframe 4h --count 20
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from tqdm import tqdm

from rag import PatternExtractor, COINS, TIMEFRAMES
from rag.config import get_file_path, PATTERNS_DIR


class RAGPopulator:
    """
    Автоматическое заполнение RAG историческими паттернами.
    
    Strategy:
    1. Найти box range с помощью алгоритма
    2. Извлечь features через PatternExtractor
    3. Фильтровать по W_box > 0.3 (качество)
    4. Сохранить в RAG
    """
    
    def __init__(self, coin: str, timeframe: str):
        self.coin = coin
        self.timeframe = timeframe
        self.extractor = PatternExtractor(coin, timeframe)
        
    def detect_box_ranges(self, lookback: int = 500) -> list:
        """
        Автоматическое обнаружение box ranges.
        
        Algorithm:
        1. Найти области с низкой волатильностью (consolidation)
        2. Проверить касания support/resistance (>= 3)
        3. Найти пробой (breakout)
        
        Returns:
            List of (breakout_time, direction) tuples
        """
        df = self.extractor.df
        candidates = []
        
        # Rolling window для поиска consolidation
        window = 48  # 48 bars lookback
        
        for i in range(window + 100, len(df) - 50):  # Leave room for future
            box_data = df.iloc[i-window:i]
            current_bar = df.iloc[i]
            
            # Check consolidation (low volatility)
            box_range = (box_data['high'].max() - box_data['low'].min())
            avg_price = box_data['close'].mean()
            range_pct = (box_range / avg_price) * 100
            
            if not (0.5 <= range_pct <= 5.0):  # Reasonable box
                continue
            
            # Check support/resistance touches
            support = box_data['low'].min()
            resistance = box_data['high'].max()
            
            tolerance = 0.003
            touches_support = sum(
                (box_data['low'] <= support * (1 + tolerance)) & 
                (box_data['low'] >= support * (1 - tolerance))
            )
            touches_resistance = sum(
                (box_data['high'] >= resistance * (1 - tolerance)) & 
                (box_data['high'] <= resistance * (1 + tolerance))
            )
            
            if touches_support < 2 or touches_resistance < 2:
                continue
            
            # Check breakout
            if current_bar['close'] > resistance * 1.005:  # Long breakout
                candidates.append((df.index[i], 'long'))
            elif current_bar['close'] < support * 0.995:  # Short breakout
                candidates.append((df.index[i], 'short'))
        
        return candidates[:lookback]  # Limit results
    
    def populate(self, max_patterns: int = 50, min_w_box: float = 0.3) -> int:
        """
        Заполнить RAG паттернами.
        
        Args:
            max_patterns: Максимум паттернов
            min_w_box: Минимальный W_box для сохранения
        
        Returns:
            Количество сохранённых паттернов
        """
        print(f"\n🔍 Searching for patterns in {self.coin} {self.timeframe}...")
        
        candidates = self.detect_box_ranges(lookback=max_patterns * 2)
        print(f"Found {len(candidates)} candidate patterns")
        
        saved_count = 0
        
        for breakout_time, direction in tqdm(candidates, desc="Extracting"):
            try:
                # Extract pattern
                pattern = self.extractor.extract(
                    breakout_time=str(breakout_time),
                    pattern_type="box_range",
                    direction=direction,
                    notes=f"Auto-detected {direction} breakout"
                )
                
                if 'error' in pattern:
                    continue
                
                # Filter by quality
                w_box = pattern.get('w_box', {}).get('W_box', 0)
                if w_box < min_w_box:
                    continue
                
                # Save
                self.extractor.save(pattern)
                saved_count += 1
                
                if saved_count >= max_patterns:
                    break
                    
            except Exception as e:
                print(f"\u26a0️ Error extracting {breakout_time}: {e}")
                continue
        
        print(f"\n✅ Saved {saved_count} patterns (W_box >= {min_w_box})")
        return saved_count


def main():
    parser = argparse.ArgumentParser(description="Populate RAG with historical patterns")
    parser.add_argument("--coin", type=str, help="Single coin to process")
    parser.add_argument("--all-coins", action="store_true", help="Process all coins")
    parser.add_argument("--timeframe", type=str, default="4h", choices=TIMEFRAMES)
    parser.add_argument("--count", type=int, default=20, help="Patterns per coin")
    parser.add_argument("--min-quality", type=float, default=0.3, help="Min W_box")
    
    args = parser.parse_args()
    
    coins = COINS if args.all_coins else [args.coin] if args.coin else ["BTC"]
    
    total_saved = 0
    for coin in coins:
        try:
            populator = RAGPopulator(coin, args.timeframe)
            count = populator.populate(max_patterns=args.count, min_w_box=args.min_quality)
            total_saved += count
        except Exception as e:
            print(f"\u274c Failed for {coin}: {e}")
    
    print(f"\n\n🎉 DONE! Total patterns saved: {total_saved}")
    print(f"\nVerify: python -m rag.cli --list")


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Populate BTC 4h (20 patterns)
python scripts/populate_rag.py --coin BTC --timeframe 4h --count 20

# Populate all coins 1h (10 patterns each)
python scripts/populate_rag.py --all-coins --timeframe 1h --count 10

# High quality only
python scripts/populate_rag.py --all-coins --timeframe 4h --min-quality 0.5
```

### ✅ Success Criteria
- [ ] 50+ patterns in library.json
- [ ] Win rate calculable
- [ ] Council gets non-empty context

---

## 🔵 PHASE 1: VECTOR DATABASE (FOUNDATION)

### 🎯 Objective
Добавить semantic search через vector embeddings.

### Why?
**Current problem:**
```python
# retriever.py сейчас:
patterns = json.load(file)
return patterns[:top_k]  # ❌ Просто первые N!
```

**With vector DB:**
```python
# Vector search:
query = "BTC consolidation breakout with high volume"
patterns = vector_store.search(query, top_k=5)
# ✅ Находит СЕМАНТИЧЕСКИ похожие!
```

### 🛠️ Implementation

#### Step 1.1: Install Dependencies

**File:** `rag/requirements.txt` (UPDATE)

```txt
# Existing
pandas>=2.0.0
pyarrow>=12.0.0
numpy>=1.24.0

# NEW - Vector Search
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

```bash
pip install -r rag/requirements.txt
```

---

#### Step 1.2: Create VectorStore

**File:** `rag/vector_store.py` (NEW)

```python
"""
Vector Database для семантического поиска паттернов.

Usage:
    store = PatternVectorStore()
    
    # Add pattern
    store.add_pattern(pattern_dict)
    
    # Search
    results = store.search(
        "BTC consolidation with RSI divergence",
        top_k=5,
        filters={"symbol": "BTC", "timeframe": "4h"}
    )
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class PatternVectorStore:
    """
    ChromaDB wrapper для торговых паттернов.
    
    Features:
    - Semantic search через embeddings
    - Metadata filtering (symbol, timeframe, direction)
    - Persistent storage
    - Fast retrieval (HNSW index)
    """
    
    def __init__(
        self, 
        persist_dir: str = "./rag/chroma_db",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_dir: Директория для ChromaDB
            embedding_model: Sentence-transformers model
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="trading_patterns",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        # Load embedding model
        print(f"🧠 Loading embedding model: {embedding_model}...")
        self.embed_model = SentenceTransformer(embedding_model)
        print("✅ Model loaded")
    
    def _pattern_to_text(self, pattern: Dict[str, Any]) -> str:
        """
        Преобразовать паттерн в текстовое представление.
        
        Этот текст будет преобразован в embedding.
        """
        meta = pattern.get('meta', {})
        box = pattern.get('box', {})
        indicators = pattern.get('indicators_before', {})
        w_box = pattern.get('w_box', {})
        future = pattern.get('future_path', {})
        
        # Build descriptive text
        text_parts = [
            f"Trading pattern: {meta.get('pattern_type', 'unknown')}",
            f"Symbol: {meta.get('coin', 'unknown')}",
            f"Timeframe: {meta.get('timeframe', 'unknown')}",
            f"Direction: {meta.get('direction', 'unknown')}",
            "",
            "Box characteristics:",
            f"- Range: {box.get('box_range_pct', 0):.2f}%",
            f"- Support touches: {box.get('touches_support', 0)}",
            f"- Resistance touches: {box.get('touches_resistance', 0)}",
            f"- Duration: {box.get('duration_bars', 0)} bars",
            "",
            "Setup indicators:",
            f"- RSI z-score: {indicators.get('rsi_zscore', 0):.2f}",
            f"- MACD z-score: {indicators.get('macd_zscore', 0):.2f}",
            f"- Volume z-score: {indicators.get('volume_zscore', 0):.2f}",
            f"- Bullish divergence: {indicators.get('div_rsi_bullish', 0)}",
            "",
            f"Quality score (W_box): {w_box.get('W_box', 0):.4f}",
            "",
            "Outcome:",
            f"- Max profit: {future.get('max_profit_pct', 0):.2f}%",
            f"- Max drawdown: {future.get('max_drawdown_pct', 0):.2f}%",
        ]
        
        if meta.get('notes'):
            text_parts.append(f"\nNotes: {meta['notes']}")
        
        return "\n".join(text_parts)
    
    def _extract_metadata(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечь metadata для фильтрации.
        """
        meta = pattern.get('meta', {})
        w_box = pattern.get('w_box', {})
        future = pattern.get('future_path', {})
        
        return {
            "symbol": meta.get('coin', 'unknown'),
            "timeframe": meta.get('timeframe', 'unknown'),
            "direction": meta.get('direction', 'unknown'),
            "pattern_type": meta.get('pattern_type', 'unknown'),
            "w_box": float(w_box.get('W_box', 0)),
            "max_profit_pct": float(future.get('max_profit_pct', 0)),
            "created_at": pattern.get('created_at', ''),
        }
    
    def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Добавить паттерн в vector store.
        
        Args:
            pattern: Pattern dict from PatternExtractor
        """
        pattern_id = pattern['id']
        
        # Convert to text
        text = self._pattern_to_text(pattern)
        
        # Generate embedding
        embedding = self.embed_model.encode(text, convert_to_numpy=True)
        
        # Extract metadata
        metadata = self._extract_metadata(pattern)
        
        # Add to collection
        self.collection.add(
            ids=[pattern_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[text]
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search по паттернам.
        
        Args:
            query: Текстовый запрос (например, "BTC consolidation breakout")
            top_k: Количество результатов
            filters: Metadata filters (e.g., {"symbol": "BTC"})
        
        Returns:
            List of {
                "id": pattern_id,
                "distance": cosine_distance,
                "metadata": {...},
                "text": document_text
            }
        """
        # Generate query embedding
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
        
        # Build where clause
        where_clause = filters if filters else None
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i],
                "text": results['documents'][0][i]
            })
        
        return formatted
    
    def bulk_add(self, patterns: List[Dict[str, Any]]) -> int:
        """
        Добавить множество паттернов.
        """
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for pattern in patterns:
            text = self._pattern_to_text(pattern)
            embedding = self.embed_model.encode(text, convert_to_numpy=True)
            metadata = self._extract_metadata(pattern)
            
            ids.append(pattern['id'])
            embeddings.append(embedding.tolist())
            metadatas.append(metadata)
            documents.append(text)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        return len(ids)
    
    def count(self) -> int:
        """Количество паттернов в базе."""
        return self.collection.count()
    
    def reset(self) -> None:
        """Очистить базу."""
        self.client.delete_collection("trading_patterns")
        self.collection = self.client.create_collection(
            name="trading_patterns",
            metadata={"hnsw:space": "cosine"}
        )
```

---

#### Step 1.3: Index Existing Patterns

**File:** `scripts/index_patterns.py` (NEW)

```python
"""
Индексировать существующие паттерны в vector DB.

Usage:
    python scripts/index_patterns.py
"""

import json
from pathlib import Path
from tqdm import tqdm

from rag.vector_store import PatternVectorStore
from rag.config import PATTERNS_DIR


def main():
    print("📦 Indexing patterns into vector database...\n")
    
    # Initialize vector store
    store = PatternVectorStore()
    
    # Find all pattern JSON files
    pattern_files = list(PATTERNS_DIR.glob("*.json"))
    pattern_files = [f for f in pattern_files if f.name != "library.json"]
    
    print(f"Found {len(pattern_files)} patterns")
    
    if len(pattern_files) == 0:
        print("⚠️ No patterns found. Run populate_rag.py first.")
        return
    
    # Load and index
    indexed = 0
    for pattern_file in tqdm(pattern_files, desc="Indexing"):
        try:
            with open(pattern_file, 'r') as f:
                pattern = json.load(f)
            
            store.add_pattern(pattern)
            indexed += 1
            
        except Exception as e:
            print(f"\u26a0️ Failed to index {pattern_file.name}: {e}")
    
    print(f"\n✅ Indexed {indexed}/{len(pattern_files)} patterns")
    print(f"Total in vector DB: {store.count()}")
    
    # Test search
    print("\n🔍 Testing search...")
    results = store.search("BTC consolidation breakout", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['id']}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Symbol: {result['metadata']['symbol']}")
        print(f"   W_box: {result['metadata']['w_box']:.4f}")


if __name__ == "__main__":
    main()
```

---

#### Step 1.4: Update RAGRetriever

**File:** `rag/retriever.py` (UPDATE)

```python
# Add at top:
from .vector_store import PatternVectorStore

class RAGRetriever:
    def __init__(self, rag_root: str = "./rag", use_multi_hyde: bool = False, use_vector_db: bool = True):
        self.rag_root = Path(rag_root)
        self.patterns_path = self.rag_root / "patterns" / "library.json"
        self.trades_path = self.rag_root / "trades" / "trade_log.json"
        self.lessons_path = self.rag_root / "lessons" / "lessons.json"
        self.use_multi_hyde = use_multi_hyde
        self.use_vector_db = use_vector_db
        
        # Initialize vector store if enabled
        self.vector_store = None
        if use_vector_db:
            try:
                self.vector_store = PatternVectorStore()
                print(f"✅ Vector DB initialized ({self.vector_store.count()} patterns)")
            except Exception as e:
                print(f"⚠️ Vector DB failed, falling back to JSON: {e}")
                self.use_vector_db = False
    
    def retrieve_patterns(
        self, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None  # NEW!
    ) -> List[Dict[str, Any]]:
        """
        Retrieve patterns using vector search or JSON fallback.
        
        Args:
            top_k: Number of patterns
            filters: Metadata filters
            query: Text query for semantic search (if vector DB enabled)
        """
        # Try vector search first
        if self.use_vector_db and self.vector_store and query:
            try:
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
                
                # Load full patterns from JSON
                pattern_ids = [r['id'] for r in results]
                patterns = self._load_patterns_by_ids(pattern_ids)
                return patterns
                
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
        
        # Fallback: JSON-based retrieval
        patterns = self.load_json(self.patterns_path)
        
        # Apply filters if provided
        if filters:
            filtered = []
            for p in patterns:
                match = True
                if 'symbol' in filters and p.get('meta', {}).get('coin') != filters['symbol']:
                    match = False
                if 'timeframe' in filters and p.get('meta', {}).get('timeframe') != filters['timeframe']:
                    match = False
                if 'direction' in filters and p.get('meta', {}).get('direction') != filters['direction']:
                    match = False
                if match:
                    filtered.append(p)
            patterns = filtered
        
        return patterns[:top_k]
    
    def _load_patterns_by_ids(self, pattern_ids: List[str]) -> List[Dict]:
        """Load full pattern data by IDs."""
        patterns = []
        for pid in pattern_ids:
            try:
                pattern_file = self.rag_root / "patterns" / f"{pid}.json"
                with open(pattern_file, 'r') as f:
                    patterns.append(json.load(f))
            except Exception as e:
                print(f"⚠️ Failed to load {pid}: {e}")
        return patterns
```

### ✅ Success Criteria (Phase 1)
- [ ] ChromaDB working
- [ ] Patterns indexed
- [ ] Semantic search returns relevant results
- [ ] 50+ patterns in vector DB

### 📊 Expected Improvement
- **Relevance:** +30-40% better matches
- **Speed:** 10x faster for large databases
- **Scalability:** Works with 10,000+ patterns

---

## 🔵 PHASE 2: MULTI-HYDE RETRIEVAL (INTELLIGENCE)

### 🎯 Objective
Realize Multi-HyDE algorithm for state-of-the-art retrieval accuracy.

### What is Multi-HyDE?

**Traditional RAG:**
```
Query: "BTC falling, what to do?"
  → Embed query
  → Search
  → Return top-5
```

**Multi-HyDE:**
```
Query: "BTC falling, what to do?"
  → Generate 4 diverse queries:
     1. "Technical: reversal patterns for BTC"
     2. "Risk: stop-loss strategies for downtrend"
     3. "Historical: similar BTC crashes"
     4. "Volume: accumulation signals"
  → For each query, generate hypothetical IDEAL pattern
  → Embed hypotheticals (not queries!)
  → Search with each
  → Combine + rerank
  → Return top-5
```

**Result:** +11% accuracy (IIT Madras, 2024)

### 🛠️ Implementation

**File:** `rag/multi_hyde.py` (NEW)

```python
"""
Multi-HyDE Retrieval for Trading Patterns.

Based on: "Multi-Hypothesis Document Retrieval" (IIT Madras, 2024)

Key idea:
- Generate MULTIPLE diverse queries from one question
- Generate HYPOTHETICAL ideal pattern for each query
- Search with hypotheticals (not original query)
- Rerank combined results

Usage:
    retriever = MultiHyDERetriever(vector_store, llm)
    patterns = retriever.retrieve(current_setup, top_k=5)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class HyDEQuery:
    """Single HyDE query with hypothetical document."""
    original_query: str
    perspective: str  # "technical", "risk", "historical", "volume"
    hypothetical_pattern: str
    weight: float = 1.0


class MultiHyDERetriever:
    """
    Multi-HyDE retriever for trading patterns.
    
    Requires:
    - VectorStore for semantic search
    - LLM for query generation (can be simple prompts)
    """
    
    def __init__(self, vector_store, llm=None):
        """
        Args:
            vector_store: PatternVectorStore instance
            llm: Optional LLM for query generation (defaults to templates)
        """
        self.vector_store = vector_store
        self.llm = llm
    
    def retrieve(
        self,
        current_setup: Dict[str, Any],
        top_k: int = 5,
        perspectives: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Multi-HyDE retrieval.
        
        Args:
            current_setup: Current market state
            top_k: Final number of patterns to return
            perspectives: List of perspectives to use
        
        Returns:
            Ranked list of patterns
        """
        if perspectives is None:
            perspectives = ["technical", "risk", "historical", "volume"]
        
        # Step 1: Generate diverse queries
        base_query = self._setup_to_query(current_setup)
        diverse_queries = self._generate_diverse_queries(base_query, perspectives)
        
        # Step 2: Generate hypothetical patterns
        hyde_queries = []
        for query, perspective in zip(diverse_queries, perspectives):
            hypothetical = self._generate_hypothetical_pattern(
                query, 
                current_setup, 
                perspective
            )
            hyde_queries.append(HyDEQuery(
                original_query=query,
                perspective=perspective,
                hypothetical_pattern=hypothetical,
                weight=1.0
            ))
        
        # Step 3: Search with each hypothetical
        all_results = []
        for hyde in hyde_queries:
            results = self.vector_store.search(
                query=hyde.hypothetical_pattern,
                top_k=top_k * 2,  # Get more candidates
                filters=self._build_filters(current_setup)
            )
            
            # Tag results with perspective
            for r in results:
                r['perspective'] = hyde.perspective
                r['weight'] = hyde.weight
            
            all_results.extend(results)
        
        # Step 4: Rerank
        ranked = self._rerank_results(all_results, current_setup)
        
        # Step 5: Deduplicate
        unique = self._deduplicate(ranked)
        
        return unique[:top_k]
    
    def _setup_to_query(self, setup: Dict[str, Any]) -> str:
        """Convert current setup to base query."""
        symbol = setup.get('symbol', 'unknown')
        timeframe = setup.get('timeframe', 'unknown')
        direction = setup.get('direction', 'long')
        features = setup.get('features', {})
        
        # Build descriptive query
        query_parts = [
            f"{symbol} {timeframe} {direction} setup:",
        ]
        
        if 'rsi' in features:
            rsi = features['rsi']
            if rsi < 30:
                query_parts.append("oversold RSI")
            elif rsi > 70:
                query_parts.append("overbought RSI")
        
        if 'price_to_sma50' in features:
            dist = features['price_to_sma50']
            if dist < -2:
                query_parts.append("below MA50")
            elif dist > 2:
                query_parts.append("above MA50")
        
        if 'volume_ratio' in features:
            vol = features['volume_ratio']
            if vol > 1.5:
                query_parts.append("high volume")
            elif vol < 0.5:
                query_parts.append("low volume")
        
        return " ".join(query_parts)
    
    def _generate_diverse_queries(
        self, 
        base_query: str, 
        perspectives: List[str]
    ) -> List[str]:
        """
        Generate diverse queries from different perspectives.
        
        If LLM available, use it. Otherwise, use templates.
        """
        if self.llm:
            return self._llm_generate_queries(base_query, perspectives)
        else:
            return self._template_generate_queries(base_query, perspectives)
    
    def _template_generate_queries(
        self, 
        base_query: str, 
        perspectives: List[str]
    ) -> List[str]:
        """
        Template-based query generation (no LLM needed).
        """
        templates = {
            "technical": f"Technical analysis: {base_query} what indicators signal?",
            "risk": f"Risk management: {base_query} what stop-loss and position size?",
            "historical": f"Historical patterns: similar to {base_query} what happened?",
            "volume": f"Volume analysis: {base_query} what volume profile?"
        }
        
        return [templates.get(p, base_query) for p in perspectives]
    
    def _generate_hypothetical_pattern(
        self,
        query: str,
        setup: Dict[str, Any],
        perspective: str
    ) -> str:
        """
        Generate hypothetical IDEAL pattern description.
        
        This is the KEY to HyDE:
        - We don't search with the query
        - We search with what the IDEAL ANSWER would look like
        """
        symbol = setup.get('symbol', 'BTC')
        timeframe = setup.get('timeframe', '4h')
        direction = setup.get('direction', 'long')
        
        templates = {
            "technical": f"""
Trading pattern: box_range
Symbol: {symbol}
Timeframe: {timeframe}
Direction: {direction}

Box characteristics:
- Range: 2.5%
- Support touches: 4
- Resistance touches: 3
- Duration: 48 bars

Setup indicators:
- RSI z-score: -0.5 (slight oversold)
- MACD z-score: 0.3 (bullish crossover)
- Volume z-score: 0.8 (above average)
- Bullish divergence: 1

Quality score (W_box): 0.65

Outcome:
- Max profit: 3.5%
- Max drawdown: -0.8%
""",
            "risk": f"""
Trading pattern with excellent risk/reward
Symbol: {symbol}
Direction: {direction}

Risk metrics:
- Stop-loss: 1.5%
- Take-profit: 3.0%
- Risk/Reward: 1:2
- Win rate: 68%

Position sizing:
- Safe size: 2% account risk
- Quality score: 0.70

Outcome:
- Stopped out only 32% of time
- Average win: +3.2%
- Average loss: -1.4%
""",
            "historical": f"""
Historical {symbol} {direction} pattern
Timeframe: {timeframe}

Similar past setups (last 6 months):
- Win rate: 65%
- Average PnL: +2.8%
- Best: +5.2%
- Worst: -1.1%

Common characteristics:
- Consolidation 40-50 bars
- Breakout on volume
- MA50 support
- RSI 30-50 range

Quality: High confidence (W_box > 0.6)
""",
            "volume": f"""
Volume-confirmed {direction} pattern
Symbol: {symbol}

Volume profile:
- Consolidation: Low volume (0.7x average)
- Breakout: High volume (1.8x average)
- Follow-through: Sustained (1.2x average)

Price action:
- Tight range during consolidation
- Clean breakout
- No immediate pullback

Quality indicators:
- Volume confirmation: Strong
- W_box: 0.68
- Success rate: 72%
"""
        }
        
        return templates.get(perspective, templates["technical"])
    
    def _build_filters(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata filters from setup."""
        filters = {}
        
        if 'symbol' in setup:
            filters['symbol'] = setup['symbol']
        
        if 'timeframe' in setup:
            filters['timeframe'] = setup['timeframe']
        
        if 'direction' in setup:
            filters['direction'] = setup['direction']
        
        return filters if filters else None
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        setup: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results by:
        1. Diversity (different perspectives)
        2. Quality (W_box score)
        3. Relevance (distance)
        """
        # Calculate composite score
        for r in results:
            # Distance score (lower is better, normalize to 0-1)
            distance_score = 1.0 / (1.0 + r['distance'])
            
            # Quality score (W_box)
            quality_score = r['metadata'].get('w_box', 0.5)
            
            # Perspective weight
            perspective_weight = r.get('weight', 1.0)
            
            # Composite
            r['composite_score'] = (
                0.4 * distance_score +
                0.4 * quality_score +
                0.2 * perspective_weight
            )
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return results
    
    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns."""
        seen = set()
        unique = []
        
        for r in results:
            pattern_id = r['id']
            if pattern_id not in seen:
                seen.add(pattern_id)
                unique.append(r)
        
        return unique
```

---

#### Update RAGRetriever to use Multi-HyDE

**File:** `rag/retriever.py` (UPDATE)

```python
# Add import
from .multi_hyde import MultiHyDERetriever

class RAGRetriever:
    def __init__(self, rag_root: str = "./rag", use_multi_hyde: bool = False, use_vector_db: bool = True):
        # ... existing code ...
        
        # Initialize Multi-HyDE if enabled
        self.multi_hyde = None
        if use_multi_hyde and self.vector_store:
            self.multi_hyde = MultiHyDERetriever(self.vector_store)
            print("✅ Multi-HyDE retriever initialized")
    
    def build_council_context(
        self,
        current_state: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Build enriched context for Council."""
        
        # Use Multi-HyDE if enabled
        if self.use_multi_hyde and self.multi_hyde:
            patterns = self.multi_hyde.retrieve(current_state, top_k=top_k)
        else:
            # Fallback to simple retrieval
            filters = {
                'symbol': current_state.get('symbol'),
                'timeframe': current_state.get('timeframe'),
                'direction': current_state.get('direction')
            }
            patterns = self.retrieve_patterns(top_k=top_k, filters=filters)
        
        # ... rest of method unchanged ...
```

### ✅ Success Criteria (Phase 2)
- [ ] Multi-HyDE working
- [ ] 4 perspectives generating queries
- [ ] Reranking combining results
- [ ] A/B test shows +10% improvement

### 📊 Expected Improvement
- **Accuracy:** +11% relevant patterns
- **Diversity:** Better coverage of different aspects
- **Robustness:** Works even with ambiguous queries

---

## 🔵 PHASE 3: AUTO-POPULATION (AUTOMATION)

### 🎯 Objective
Automatic RAG population after each trade closes.

### Why?
**Manual now:**
```
1. Trade closes
2. Manually find pattern on TradingView
3. Run rag.cli extract
4. Commit to git
```

**Auto with Dexter:**
```
1. Trade closes → trigger
2. Dexter analyzes: why win/loss?
3. Extract pattern automatically
4. Save to RAG if confidence > 0.7
5. Index in vector DB
```

### 🛠️ Implementation

**File:** `rag/auto_populator.py` (NEW)

```python
"""
Automatic RAG population via Dexter post-mortem.

Integrates:
- Dexter (why did trade work/fail?)
- PatternExtractor (extract features)
- VectorStore (index)

Usage:
    populator = AutoPopulator()
    await populator.on_trade_close(trade_data)
"""

from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

from .extractor import PatternExtractor
from .vector_store import PatternVectorStore


class AutoPopulator:
    """
    Automatic pattern extraction and RAG population.
    
    Workflow:
    1. Trade closes → trigger
    2. Dexter post-mortem analysis
    3. If confident (>0.7) → extract pattern
    4. Save + index
    5. Log to lessons learned
    """
    
    def __init__(self, dexter_client=None, min_confidence: float = 0.7):
        """
        Args:
            dexter_client: Dexter API client (optional)
            min_confidence: Minimum confidence to save pattern
        """
        self.dexter = dexter_client
        self.min_confidence = min_confidence
        self.vector_store = PatternVectorStore()
    
    async def on_trade_close(self, trade: Dict[str, Any]) -> Optional[str]:
        """
        Called when trade closes.
        
        Args:
            trade: {
                'symbol': 'BTC',
                'timeframe': '4h',
                'direction': 'long',
                'entry_time': '2024-11-26 14:00',
                'exit_time': '2024-11-28 10:00',
                'pnl_pct': 2.3,
                'outcome': 'win',
                ...
            }
        
        Returns:
            Pattern ID if saved, None otherwise
        """
        print(f"\n🔍 Analyzing closed trade: {trade['symbol']} {trade['direction']}")
        
        # Step 1: Dexter post-mortem
        if self.dexter:
            analysis = await self._dexter_analyze(trade)
        else:
            # Simple heuristic if no Dexter
            analysis = self._simple_analysis(trade)
        
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Learning: {analysis['learnings']}")
        
        # Step 2: Check confidence threshold
        if analysis['confidence'] < self.min_confidence:
            print(f"   ⚠️ Low confidence, skipping")
            return None
        
        # Step 3: Extract pattern
        try:
            extractor = PatternExtractor(trade['symbol'], trade['timeframe'])
            
            pattern = extractor.extract(
                breakout_time=trade['entry_time'],
                pattern_type=analysis.get('pattern_type', 'box_range'),
                direction=trade['direction'],
                notes=f"Auto: {analysis['learnings']}"
            )
            
            if 'error' in pattern:
                print(f"   ❌ Extraction failed: {pattern['error']}")
                return None
            
            # Add outcome info
            pattern['outcome'] = {
                'pnl_pct': trade.get('pnl_pct', 0),
                'result': trade.get('outcome', 'unknown'),
                'exit_time': trade.get('exit_time'),
                'dexter_confidence': analysis['confidence'],
                'dexter_learnings': analysis['learnings']
            }
            
        except Exception as e:
            print(f"   ❌ Extraction error: {e}")
            return None
        
        # Step 4: Save pattern
        try:
            json_path = extractor.save(pattern)
            print(f"   ✅ Saved: {json_path}")
        except Exception as e:
            print(f"   ❌ Save failed: {e}")
            return None
        
        # Step 5: Index in vector DB
        try:
            self.vector_store.add_pattern(pattern)
            print(f"   ✅ Indexed in vector DB")
        except Exception as e:
            print(f"   ⚠️ Indexing failed: {e}")
        
        # Step 6: Update lessons learned
        self._add_lesson(analysis, trade)
        
        return pattern['id']
    
    async def _dexter_analyze(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dexter post-mortem analysis.
        
        Returns:
            {
                'confidence': 0.85,
                'pattern_type': 'box_range',
                'learnings': 'Strong MA50 support with volume confirmation',
                'why_worked': '...',
                'what_could_improve': '...'
            }
        """
        # TODO: Implement Dexter API call
        # For now, placeholder
        raise NotImplementedError("Dexter integration pending")
    
    def _simple_analysis(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple heuristic analysis (fallback without Dexter).
        """
        pnl = trade.get('pnl_pct', 0)
        outcome = trade.get('outcome', 'unknown')
        
        # Simple confidence based on outcome
        if outcome == 'win' and pnl > 2.0:
            confidence = 0.8
            learnings = f"Strong {trade['direction']} setup, +{pnl:.1f}% profit"
        elif outcome == 'win' and pnl > 0.5:
            confidence = 0.6
            learnings = f"Moderate {trade['direction']} setup, +{pnl:.1f}% profit"
        elif outcome == 'loss' and pnl < -1.0:
            confidence = 0.7  # Learn from mistakes!
            learnings = f"Failed {trade['direction']}, {pnl:.1f}% loss - avoid similar"
        else:
            confidence = 0.4
            learnings = "Marginal setup"
        
        return {
            'confidence': confidence,
            'pattern_type': 'box_range',
            'learnings': learnings
        }
    
    def _add_lesson(self, analysis: Dict[str, Any], trade: Dict[str, Any]) -> None:
        """
        Add to lessons learned.
        """
        # TODO: Implement lessons.json updates
        pass
```

### ✅ Success Criteria (Phase 3)
- [ ] Auto-population working
- [ ] Dexter integration (or fallback heuristic)
- [ ] New patterns added after trades
- [ ] RAG grows automatically

### 📊 Expected Improvement
- **Data growth:** +10-20 patterns per week
- **Quality:** Only confident patterns saved
- **Lessons:** Automatic learning from mistakes

---

## 📊 METRICS & VALIDATION

### How to measure success?

#### 1. Retrieval Quality
```python
# scripts/evaluate_rag.py
def evaluate_retrieval_quality():
    """
    Test cases:
    1. Query: "BTC consolidation breakout"
       Expected: patterns with box_range, BTC, long direction
    
    2. Query: "High volume breakout"
       Expected: patterns with volume_zscore > 1.0
    
    3. Query: "Failed short setup"
       Expected: patterns with negative PnL, short direction
    
    Metrics:
    - Precision@5: Are top-5 relevant?
    - MRR (Mean Reciprocal Rank): Position of first relevant result
    - Diversity: How many different symbols/timeframes?
    """
```

#### 2. Council Impact
```python
def measure_council_impact():
    """
    Compare:
    - Council WITH RAG context vs WITHOUT
    
    Metrics:
    - Win rate improvement
    - Confidence calibration
    - Decision time
    """
```

#### 3. Data Growth
```python
def track_data_growth():
    """
    Monitor:
    - Patterns per day
    - Average W_box score
    - Coverage (symbols, timeframes)
    """
```

---

## 📝 SUMMARY

### Timeline
| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **0. Data** | 1-2 days | 50+ patterns in library.json |
| **1. Vector DB** | 3-4 days | Semantic search working |
| **2. Multi-HyDE** | 4-5 days | +11% retrieval accuracy |
| **3. Auto-population** | 3-4 days | Self-learning RAG |
| **Total** | **11-15 days** | Production-ready v2.0 |

### Priority
1. 🔴 **CRITICAL:** Phase 0 (data) - without this, nothing works
2. 🔴 **HIGH:** Phase 1 (vector DB) - enables semantic search
3. 🟡 **MEDIUM:** Phase 2 (Multi-HyDE) - accuracy boost
4. 🟢 **LOW:** Phase 3 (auto-population) - automation

### Expected Results
- **v1.0 (current):** Basic JSON retrieval, 6/10
- **v1.5 (Phase 0+1):** Vector search, 8/10
- **v2.0 (All phases):** Multi-HyDE + auto-learning, 9/10

---

*Last updated: December 8, 2025*
