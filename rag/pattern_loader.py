"""
ScArlet-Sails Pattern Loader

Массовая загрузка паттернов в RAG систему.
Поддерживает JSON, CSV, директории.

Philosophy:
    "Batch operations > single operations"
    "Validate first, load second"
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .pattern_validator import (
    BatchValidationResult,
    Pattern,
    PatternValidator,
    ValidationError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LOAD RESULT
# =============================================================================

@dataclass
class LoadResult:
    """Результат загрузки паттернов."""
    total_files: int = 0
    total_patterns: int = 0
    loaded_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    loaded_patterns: List[Pattern] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.loaded_count / self.total_patterns if self.total_patterns > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_files': self.total_files,
            'total_patterns': self.total_patterns,
            'loaded_count': self.loaded_count,
            'skipped_count': self.skipped_count,
            'error_count': self.error_count,
            'success_rate': self.success_rate,
            'errors': self.errors[:10],  # First 10
            'warnings': self.warnings[:10],
        }


# =============================================================================
# PATTERN LOADER
# =============================================================================

class PatternLoader:
    """
    Загрузчик паттернов в RAG систему.
    
    Поддерживает:
    - JSON файлы (один паттерн или массив)
    - CSV файлы
    - Директории с паттернами
    - Валидация перед загрузкой
    - Дедупликация
    
    Usage:
        loader = PatternLoader(patterns_dir="rag/patterns")
        
        # Из файла
        result = loader.load_file("my_patterns.json")
        
        # Из директории
        result = loader.load_directory("patterns/")
        
        # Из списка dict
        result = loader.load_patterns([{...}, {...}])
    """
    
    def __init__(
        self,
        patterns_dir: Optional[Union[str, Path]] = None,
        validate: bool = True,
        strict: bool = False,
        skip_duplicates: bool = True,
    ):
        """
        Args:
            patterns_dir: Директория для сохранения паттернов
            validate: Валидировать перед загрузкой
            strict: Строгая валидация (warnings = errors)
            skip_duplicates: Пропускать дубликаты
        """
        self.patterns_dir = Path(patterns_dir) if patterns_dir else Path("rag/patterns")
        self.validate = validate
        self.strict = strict
        self.skip_duplicates = skip_duplicates
        
        self.validator = PatternValidator(strict=strict)
        self._existing_ids: set = set()
        
        # Загружаем существующие ID
        self._load_existing_ids()
    
    def _load_existing_ids(self) -> None:
        """Загрузить ID существующих паттернов."""
        library_file = self.patterns_dir / "library.json"
        
        if library_file.exists():
            try:
                with open(library_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                patterns = data.get('patterns', [])
                self._existing_ids = {p.get('id', '').lower() for p in patterns}
                logger.info(f"Loaded {len(self._existing_ids)} existing pattern IDs")
                
            except Exception as e:
                logger.warning(f"Failed to load existing patterns: {e}")
    
    def load_file(
        self,
        file_path: Union[str, Path],
        save: bool = True,
    ) -> LoadResult:
        """
        Загрузить паттерны из файла.
        
        Args:
            file_path: Путь к файлу (JSON или CSV)
            save: Сохранить в library.json
            
        Returns:
            LoadResult
        """
        file_path = Path(file_path)
        result = LoadResult(total_files=1)
        
        if not file_path.exists():
            result.errors.append(f"File not found: {file_path}")
            return result
        
        # Определяем формат
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            patterns_data = self._load_json(file_path, result)
        elif suffix == '.csv':
            patterns_data = self._load_csv(file_path, result)
        else:
            result.errors.append(f"Unsupported file format: {suffix}")
            return result
        
        if not patterns_data:
            return result
        
        # Загружаем паттерны
        return self._process_patterns(patterns_data, result, save=save)
    
    def load_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        save: bool = True,
    ) -> LoadResult:
        """
        Загрузить паттерны из директории.
        
        Args:
            dir_path: Путь к директории
            recursive: Рекурсивно обходить поддиректории
            save: Сохранить в library.json
            
        Returns:
            LoadResult
        """
        dir_path = Path(dir_path)
        result = LoadResult()
        
        if not dir_path.exists():
            result.errors.append(f"Directory not found: {dir_path}")
            return result
        
        # Собираем все файлы
        pattern = "**/*.json" if recursive else "*.json"
        json_files = list(dir_path.glob(pattern))
        
        pattern = "**/*.csv" if recursive else "*.csv"
        csv_files = list(dir_path.glob(pattern))
        
        all_files = json_files + csv_files
        result.total_files = len(all_files)
        
        if not all_files:
            result.warnings.append(f"No pattern files found in {dir_path}")
            return result
        
        # Собираем все паттерны
        all_patterns_data = []
        
        for file_path in all_files:
            if file_path.name == 'library.json':
                continue  # Пропускаем library
            
            suffix = file_path.suffix.lower()
            
            try:
                if suffix == '.json':
                    patterns_data = self._load_json(file_path, result)
                elif suffix == '.csv':
                    patterns_data = self._load_csv(file_path, result)
                else:
                    continue
                
                if patterns_data:
                    all_patterns_data.extend(patterns_data)
                    
            except Exception as e:
                result.errors.append(f"Error loading {file_path}: {e}")
        
        # Загружаем паттерны
        return self._process_patterns(all_patterns_data, result, save=save)
    
    def load_patterns(
        self,
        patterns_data: List[Dict[str, Any]],
        save: bool = True,
    ) -> LoadResult:
        """
        Загрузить паттерны из списка словарей.
        
        Args:
            patterns_data: Список словарей с паттернами
            save: Сохранить в library.json
            
        Returns:
            LoadResult
        """
        result = LoadResult(total_files=0)
        return self._process_patterns(patterns_data, result, save=save)
    
    def _load_json(
        self,
        file_path: Path,
        result: LoadResult,
    ) -> List[Dict[str, Any]]:
        """Загрузить паттерны из JSON файла."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Может быть список, объект с 'patterns', или один паттерн
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'patterns' in data:
                    return data['patterns']
                else:
                    return [data]  # Один паттерн
            else:
                result.errors.append(f"Invalid JSON structure in {file_path}")
                return []
                
        except json.JSONDecodeError as e:
            result.errors.append(f"JSON parse error in {file_path}: {e}")
            return []
        except Exception as e:
            result.errors.append(f"Error reading {file_path}: {e}")
            return []
    
    def _load_csv(
        self,
        file_path: Path,
        result: LoadResult,
    ) -> List[Dict[str, Any]]:
        """Загрузить паттерны из CSV файла."""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Конвертируем типы
                    pattern = self._convert_csv_row(row)
                    if pattern:
                        patterns.append(pattern)
                        
        except Exception as e:
            result.errors.append(f"CSV parse error in {file_path}: {e}")
        
        return patterns
    
    def _convert_csv_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Конвертировать строку CSV в паттерн."""
        pattern = {}
        
        # Обязательные поля
        for field in ['id', 'name', 'direction', 'description']:
            if field in row and row[field]:
                pattern[field] = row[field]
        
        # Необязательные строковые поля
        for field in ['outcome', 'category']:
            if field in row and row[field]:
                pattern[field] = row[field]
        
        # Числовые поля
        if 'pnl_pct' in row and row['pnl_pct']:
            try:
                pattern['pnl_pct'] = float(row['pnl_pct'])
            except ValueError:
                pass
        
        # Списки (разделённые запятыми)
        if 'tags' in row and row['tags']:
            pattern['tags'] = [t.strip() for t in row['tags'].split(',')]
        
        if 'lessons' in row and row['lessons']:
            pattern['lessons'] = [l.strip() for l in row['lessons'].split('|')]
        
        return pattern if pattern.get('id') else None
    
    def _process_patterns(
        self,
        patterns_data: List[Dict[str, Any]],
        result: LoadResult,
        save: bool = True,
    ) -> LoadResult:
        """Обработать и загрузить паттерны."""
        result.total_patterns = len(patterns_data)
        
        if not patterns_data:
            return result
        
        # Валидация
        if self.validate:
            validation = self.validator.validate_batch(patterns_data)
            
            result.error_count = validation.invalid_count
            result.warnings.extend(validation.warnings)
            
            for err in validation.errors:
                result.errors.append(f"{err.pattern_id}: {err.message}")
            
            valid_patterns = validation.patterns
        else:
            # Без валидации — просто конвертируем
            valid_patterns = []
            for data in patterns_data:
                try:
                    valid_patterns.append(Pattern(**data))
                except Exception as e:
                    result.errors.append(f"Invalid pattern: {e}")
                    result.error_count += 1
        
        # Фильтрация дубликатов
        new_patterns = []
        for pattern in valid_patterns:
            if self.skip_duplicates and pattern.id in self._existing_ids:
                result.skipped_count += 1
                result.warnings.append(f"Skipped duplicate: {pattern.id}")
            else:
                new_patterns.append(pattern)
                self._existing_ids.add(pattern.id)
        
        result.loaded_count = len(new_patterns)
        result.loaded_patterns = new_patterns
        
        # Сохранение
        if save and new_patterns:
            self._save_to_library(new_patterns, result)
        
        return result
    
    def _save_to_library(
        self,
        new_patterns: List[Pattern],
        result: LoadResult,
    ) -> None:
        """Сохранить паттерны в library.json."""
        library_file = self.patterns_dir / "library.json"
        
        # Загружаем существующую библиотеку
        if library_file.exists():
            try:
                with open(library_file, 'r', encoding='utf-8') as f:
                    library = json.load(f)
            except Exception:
                library = {'patterns': [], 'metadata': {}}
        else:
            library = {'patterns': [], 'metadata': {}}
        
        # Добавляем новые паттерны
        existing_patterns = library.get('patterns', [])
        
        for pattern in new_patterns:
            pattern_dict = pattern.to_dict()
            existing_patterns.append(pattern_dict)
        
        # Обновляем metadata
        library['patterns'] = existing_patterns
        library['metadata'] = {
            'total_patterns': len(existing_patterns),
            'last_updated': datetime.now().isoformat(),
            'version': library.get('metadata', {}).get('version', 1),
        }
        
        # Сохраняем
        try:
            self.patterns_dir.mkdir(parents=True, exist_ok=True)
            
            with open(library_file, 'w', encoding='utf-8') as f:
                json.dump(library, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(new_patterns)} patterns to {library_file}")
            
        except Exception as e:
            result.errors.append(f"Failed to save library: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику библиотеки."""
        library_file = self.patterns_dir / "library.json"
        
        if not library_file.exists():
            return {'total_patterns': 0, 'exists': False}
        
        try:
            with open(library_file, 'r', encoding='utf-8') as f:
                library = json.load(f)
            
            patterns = library.get('patterns', [])
            
            # Статистика по категориям
            categories = {}
            outcomes = {'win': 0, 'loss': 0, 'breakeven': 0, 'unknown': 0}
            directions = {'long': 0, 'short': 0, 'neutral': 0}
            
            for p in patterns:
                cat = p.get('category', 'other')
                categories[cat] = categories.get(cat, 0) + 1
                
                out = p.get('outcome', 'unknown')
                outcomes[out] = outcomes.get(out, 0) + 1
                
                dir_ = p.get('direction', 'neutral')
                directions[dir_] = directions.get(dir_, 0) + 1
            
            return {
                'total_patterns': len(patterns),
                'exists': True,
                'categories': categories,
                'outcomes': outcomes,
                'directions': directions,
                'metadata': library.get('metadata', {}),
            }
            
        except Exception as e:
            return {'total_patterns': 0, 'exists': True, 'error': str(e)}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_patterns_from_file(
    file_path: Union[str, Path],
    patterns_dir: Optional[str] = None,
) -> LoadResult:
    """Удобная функция для загрузки из файла."""
    loader = PatternLoader(patterns_dir=patterns_dir)
    return loader.load_file(file_path)


def load_patterns_from_directory(
    dir_path: Union[str, Path],
    patterns_dir: Optional[str] = None,
) -> LoadResult:
    """Удобная функция для загрузки из директории."""
    loader = PatternLoader(patterns_dir=patterns_dir)
    return loader.load_directory(dir_path)