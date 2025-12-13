"""
Day 5: Build FAISS Vector Index for RAG patterns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime


def check_patterns():
    """Check pattern library status."""
    library_path = Path('rag/patterns/library.json')
    
    if not library_path.exists():
        print("❌ library.json not found!")
        return None
    
    with open(library_path) as f:
        library = json.load(f)
    
    patterns = library.get('patterns', [])
    print(f"Found {len(patterns)} patterns in library")
    
    return patterns


def build_index():
    """Build FAISS index using PatternVectorStore."""
    try:
        from rag.vector_store import PatternVectorStore
    except ImportError as e:
        print(f"❌ Cannot import PatternVectorStore: {e}")
        print("Trying alternative approach...")
        return build_simple_index()
    
    print("\nBuilding FAISS index...")
    
    store = PatternVectorStore(patterns_dir='rag/patterns')
    count = store.build_from_directory(verbose=True)
    
    return count


def build_simple_index():
    """Fallback: Build simple JSON-based index."""
    print("\nBuilding simple index (no FAISS)...")
    
    patterns_dir = Path('rag/patterns')
    patterns = []
    
    # Load all pattern files
    for f in patterns_dir.glob('pattern_*.json'):
        with open(f) as fp:
            patterns.append(json.load(fp))
    
    # Create index
    index = {
        "version": "1.0",
        "type": "simple_json",
        "patterns": len(patterns),
        "by_type": {},
        "by_outcome": {"win": 0, "loss": 0},
        "created_at": datetime.now().isoformat(),
    }
    
    for p in patterns:
        ptype = p.get('pattern_type', 'unknown')
        outcome = p.get('outcome', {}).get('result', 'unknown')
        
        index['by_type'][ptype] = index['by_type'].get(ptype, 0) + 1
        if outcome in index['by_outcome']:
            index['by_outcome'][outcome] += 1
    
    # Save index
    index_path = patterns_dir / 'simple_index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ Simple index created: {index_path}")
    return len(patterns)


def test_search():
    """Test search functionality."""
    print("\n" + "-" * 60)
    print("Testing search...")
    
    try:
        from rag.vector_store import PatternVectorStore
        
        store = PatternVectorStore(patterns_dir='rag/patterns')
        
        # Test query
        results = store.search("momentum breakout bullish", top_k=3)
        
        print(f"\nQuery: 'momentum breakout bullish'")
        print(f"Results: {len(results)}")
        
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.get('pattern_type', 'N/A')} - {r.get('outcome', {}).get('result', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"⚠️ Search test skipped: {e}")
        return False


def main():
    print("=" * 60)
    print("DAY 5: BUILD RAG FAISS INDEX")
    print("=" * 60)
    
    # Check patterns
    patterns = check_patterns()
    if not patterns:
        return
    
    if len(patterns) < 10:
        print(f"⚠️ Only {len(patterns)} patterns. Recommend 10+ for good retrieval.")
    
    # Build index
    try:
        count = build_index()
        print(f"\n✅ Indexed {count} patterns")
    except Exception as e:
        print(f"❌ FAISS build failed: {e}")
        count = build_simple_index()
    
    # Test search
    test_search()
    
    print()
    print("=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()