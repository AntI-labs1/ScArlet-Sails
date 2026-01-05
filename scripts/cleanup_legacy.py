#!/usr/bin/env python3
"""
🧹 LEGACY FILE CLEANUP SCRIPT

Протокол "Чистка" - удаление устаревших файлов.

Этот скрипт удаляет старые дубликаты и конфликтующие файлы.

Usage:
    python cleanup_legacy.py --dry-run  # Preview changes
    python cleanup_legacy.py --execute  # Actually delete
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class LegacyCleanup:
    """
    🧹 Cleanup utility for legacy files
    """
    
    # Files to be removed based on TZ
    LEGACY_FILES = [
        # Duplicates that should use canonical_pipeline.py
        "scripts/old_validate.py",
        "scripts/data_checker.py",
        "scripts/validation_util.py",
        
        # Old/conflicting implementations
        "scripts/legacy_pipeline.py",
        "scripts/deprecated_fetch.py",
        "data/old_cache/*.json",
    ]
    
    # Patterns for cleanup
    CLEANUP_PATTERNS = [
        "**/*_backup.py",
        "**/*_old.py",
        "**/*_deprecated.py",
        "**/.DS_Store",
        "**/__pycache__",
        "**/*.pyc",
    ]
    
    def __init__(self, dry_run: bool = True, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.stats = {
            "files_found": 0,
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": []
        }
    
    def setup_logging(self):
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - [%(levelname)s] - %(message)s'
        )
    
    def find_legacy_files(self) -> List[Path]:
        """
        Find all legacy files to be removed
        """
        self.logger.info("🔍 Scanning for legacy files...")
        
        files_to_remove = []
        
        # Check specific files
        for filepath in self.LEGACY_FILES:
            path = ROOT_DIR / filepath
            
            # Handle wildcards
            if "*" in filepath:
                parent = Path(filepath).parent
                pattern = Path(filepath).name
                if (ROOT_DIR / parent).exists():
                    matches = list((ROOT_DIR / parent).glob(pattern))
                    files_to_remove.extend(matches)
            elif path.exists():
                files_to_remove.append(path)
        
        # Check patterns
        for pattern in self.CLEANUP_PATTERNS:
            matches = list(ROOT_DIR.glob(pattern))
            files_to_remove.extend(matches)
        
        # Remove duplicates
        files_to_remove = list(set(files_to_remove))
        
        self.stats["files_found"] = len(files_to_remove)
        self.logger.info(f"Found {len(files_to_remove)} legacy files")
        
        return files_to_remove
    
    def get_file_info(self, filepath: Path) -> Dict:
        """
        Get file information
        """
        try:
            stat = filepath.stat()
            return {
                "path": str(filepath.relative_to(ROOT_DIR)),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_dir": filepath.is_dir()
            }
        except Exception as e:
            return {
                "path": str(filepath),
                "error": str(e)
            }
    
    def remove_file(self, filepath: Path) -> bool:
        """
        Remove a single file or directory
        """
        try:
            info = self.get_file_info(filepath)
            
            if self.dry_run:
                self.logger.info(f"  [DRY-RUN] Would delete: {info['path']} ({info.get('size', 0)} bytes)")
                return True
            
            if filepath.is_dir():
                # Remove directory and contents
                import shutil
                shutil.rmtree(filepath)
                self.logger.info(f"  ❌ Deleted directory: {info['path']}")
            else:
                # Remove file
                size = filepath.stat().st_size
                filepath.unlink()
                self.stats["bytes_freed"] += size
                self.logger.info(f"  ❌ Deleted: {info['path']} ({size} bytes)")
            
            self.stats["files_deleted"] += 1
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete {filepath}: {e}"
            self.logger.error(f"  ⚠️ {error_msg}")
            self.stats["errors"].append(error_msg)
            return False
    
    def cleanup(self) -> Dict:
        """
        Execute cleanup
        """
        mode = "DRY-RUN" if self.dry_run else "EXECUTE"
        self.logger.info(f"\n🧹 Starting cleanup [{mode}]...\n")
        
        files = self.find_legacy_files()
        
        if not files:
            self.logger.info("✅ No legacy files found!")
            return self.stats
        
        self.logger.info(f"\nFiles to be removed:")
        for filepath in sorted(files):
            self.remove_file(filepath)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 CLEANUP SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Files found: {self.stats['files_found']}")
        self.logger.info(f"Files deleted: {self.stats['files_deleted']}")
        
        if not self.dry_run:
            kb_freed = self.stats['bytes_freed'] / 1024
            self.logger.info(f"Space freed: {kb_freed:.2f} KB")
        
        if self.stats['errors']:
            self.logger.warning(f"\n⚠️ Errors: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                self.logger.warning(f"  - {error}")
        
        if self.dry_run:
            self.logger.info("\n🔵 This was a DRY-RUN. Use --execute to actually delete.")
        else:
            self.logger.info("\n✅ Cleanup completed!")
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="🧹 Legacy File Cleanup Utility"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without deleting (default)"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (use with caution!)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # If --execute is specified, turn off dry-run
    dry_run = not args.execute
    
    if args.execute:
        print("\n⚠️  WARNING: You are about to DELETE files!")
        print("This action CANNOT be undone.")
        response = input("\nType 'YES' to continue: ")
        
        if response != "YES":
            print("❌ Cleanup cancelled.")
            return
    
    cleaner = LegacyCleanup(dry_run=dry_run, verbose=args.verbose)
    cleaner.cleanup()


if __name__ == "__main__":
    main()
