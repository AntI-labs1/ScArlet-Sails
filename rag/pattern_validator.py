"""
ScArlet-Sails Pattern Validator

Валидация паттернов перед загрузкой в RAG.
Pydantic схемы обеспечивают строгую типизацию.

Philosophy:
    "Мусор на входе → мусор на выходе"
    "Валидация ДО загрузки, не после"
"""

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class Direction(str, Enum):
    """Направление сделки."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Outcome(str, Enum):
    """Результат сделки."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    UNKNOWN = "unknown"


class ExitReason(str, Enum):
    """Причина выхода."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    SIGNAL_REVERSAL = "signal_reversal"
    RISK_LIMIT = "risk_limit"
    OTHER = "other"


class Regime(str, Enum):
    """Режим рынка."""
    NORMAL = "normal"
    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"
    TRENDING = "trending"
    RANGING = "ranging"
    CRISIS = "crisis"


class PatternCategory(str, Enum):
    """Категория паттерна."""
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    FALSE_BREAKOUT = "false_breakout"
    RANGE_TRADE = "range_trade"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    DIVERGENCE = "divergence"
    CONSOLIDATION_BREAK = "consolidation_break"
    OTHER = "other"


# =============================================================================
# SCHEMAS
# =============================================================================

class EntryIndicators(BaseModel):
    """Индикаторы на момент входа."""
    rsi: Optional[float] = Field(None, ge=0, le=100)
    price_to_ma50_pct: Optional[float] = None
    price_to_ema21_pct: Optional[float] = None
    volume_ratio: Optional[float] = Field(None, ge=0)
    atr_pct: Optional[float] = Field(None, ge=0)
    macd_histogram: Optional[float] = None
    bb_position: Optional[float] = Field(None, ge=0, le=1, description="0=lower band, 1=upper band")
    
    class Config:
        extra = "allow"  # Разрешаем дополнительные поля


class EntryDetails(BaseModel):
    """Детали входа в позицию."""
    description: Optional[str] = None
    indicators: Optional[EntryIndicators] = None
    timestamp: Optional[datetime] = None
    price: Optional[float] = Field(None, gt=0)
    
    class Config:
        extra = "allow"


class ExitDetails(BaseModel):
    """Детали выхода из позиции."""
    reason: Optional[ExitReason] = None
    timestamp: Optional[datetime] = None
    price: Optional[float] = Field(None, gt=0)
    
    class Config:
        extra = "allow"


class PatternContext(BaseModel):
    """Контекст паттерна."""
    symbol: Optional[str] = None
    timeframe: Optional[str] = Field(None, pattern=r"^(1m|5m|15m|30m|1h|4h|1d|1w)$")
    regime: Optional[Regime] = None
    trend: Optional[str] = None
    
    class Config:
        extra = "allow"


class PatternMetadata(BaseModel):
    """Метаданные паттерна."""
    source: Optional[str] = None
    analyst: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: int = 1
    
    class Config:
        extra = "allow"


class Pattern(BaseModel):
    """
    Полная схема паттерна.
    
    Обязательные поля:
    - id: уникальный идентификатор
    - name: человекочитаемое название
    - direction: long/short/neutral
    - outcome: win/loss/breakeven
    - description: текстовое описание (для embeddings)
    
    Рекомендуемые поля:
    - pnl_pct: результат в процентах
    - category: категория паттерна
    - entry: детали входа
    - context: контекст рынка
    """
    
    # Required
    id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    direction: Direction
    outcome: Outcome = Outcome.UNKNOWN
    description: str = Field(..., min_length=10, max_length=2000)
    
    # Recommended
    pnl_pct: float = Field(0.0, ge=-100, le=1000)
    category: Optional[PatternCategory] = None
    
    # Entry/Exit
    entry: Optional[EntryDetails] = None
    exit: Optional[ExitDetails] = None
    holding_period_hours: Optional[int] = Field(None, ge=0)
    
    # Context
    context: Optional[PatternContext] = None
    
    # Learning
    lessons: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Optional[PatternMetadata] = None
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """ID должен быть snake_case."""
        if ' ' in v:
            raise ValueError("ID cannot contain spaces, use snake_case")
        return v.lower()
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Теги в нижнем регистре."""
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    @model_validator(mode='after')
    def validate_outcome_pnl(self) -> 'Pattern':
        """Проверить согласованность outcome и pnl."""
        if self.outcome == Outcome.WIN and self.pnl_pct < 0:
            raise ValueError("WIN outcome cannot have negative pnl_pct")
        if self.outcome == Outcome.LOSS and self.pnl_pct > 0:
            raise ValueError("LOSS outcome cannot have positive pnl_pct")
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict для JSON."""
        return self.model_dump(mode='json', exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Создать из dict."""
        return cls(**data)


# =============================================================================
# VALIDATION RESULT
# =============================================================================

class ValidationError(BaseModel):
    """Ошибка валидации."""
    pattern_id: Optional[str] = None
    field: str
    message: str
    value: Optional[Any] = None


class ValidationResult(BaseModel):
    """Результат валидации."""
    valid: bool
    pattern: Optional[Pattern] = None
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)


class BatchValidationResult(BaseModel):
    """Результат валидации пачки паттернов."""
    total: int
    valid_count: int
    invalid_count: int
    patterns: List[Pattern] = Field(default_factory=list)
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.valid_count / self.total if self.total > 0 else 0.0


# =============================================================================
# VALIDATOR
# =============================================================================

class PatternValidator:
    """
    Валидатор паттернов.
    
    Usage:
        validator = PatternValidator()
        
        # Один паттерн
        result = validator.validate(pattern_dict)
        if result.valid:
            pattern = result.pattern
        
        # Пачка паттернов
        result = validator.validate_batch(patterns_list)
        print(f"Valid: {result.valid_count}/{result.total}")
    """
    
    def __init__(self, strict: bool = False):
        """
        Args:
            strict: Если True, любое предупреждение становится ошибкой
        """
        self.strict = strict
        self._seen_ids: set = set()
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Валидировать один паттерн.
        
        Args:
            data: Словарь с данными паттерна
            
        Returns:
            ValidationResult с паттерном или ошибками
        """
        errors = []
        warnings = []
        
        # Проверка обязательных полей
        required = ['id', 'name', 'direction', 'description']
        for field in required:
            if field not in data or not data[field]:
                errors.append(ValidationError(
                    pattern_id=data.get('id'),
                    field=field,
                    message=f"Missing required field: {field}",
                ))
        
        if errors:
            return ValidationResult(valid=False, errors=errors)
        
        # Проверка уникальности ID
        pattern_id = data.get('id', '').lower()
        if pattern_id in self._seen_ids:
            errors.append(ValidationError(
                pattern_id=pattern_id,
                field='id',
                message=f"Duplicate pattern ID: {pattern_id}",
            ))
            return ValidationResult(valid=False, errors=errors)
        
        # Pydantic валидация
        try:
            pattern = Pattern(**data)
            self._seen_ids.add(pattern.id)
            
            # Дополнительные проверки
            warnings.extend(self._check_quality(pattern))
            
            if self.strict and warnings:
                return ValidationResult(
                    valid=False,
                    errors=[ValidationError(
                        pattern_id=pattern.id,
                        field='quality',
                        message=w,
                    ) for w in warnings],
                )
            
            return ValidationResult(
                valid=True,
                pattern=pattern,
                warnings=warnings,
            )
            
        except Exception as e:
            # Парсим Pydantic ошибки
            error_msg = str(e)
            errors.append(ValidationError(
                pattern_id=data.get('id'),
                field='schema',
                message=error_msg,
            ))
            return ValidationResult(valid=False, errors=errors)
    
    def validate_batch(
        self,
        data_list: List[Dict[str, Any]],
        continue_on_error: bool = True,
    ) -> BatchValidationResult:
        """
        Валидировать пачку паттернов.
        
        Args:
            data_list: Список словарей с паттернами
            continue_on_error: Продолжать при ошибках
            
        Returns:
            BatchValidationResult
        """
        self._seen_ids.clear()  # Сброс для пачки
        
        patterns = []
        all_errors = []
        all_warnings = []
        
        for i, data in enumerate(data_list):
            result = self.validate(data)
            
            if result.valid:
                patterns.append(result.pattern)
                all_warnings.extend(result.warnings)
            else:
                all_errors.extend(result.errors)
                if not continue_on_error:
                    break
        
        return BatchValidationResult(
            total=len(data_list),
            valid_count=len(patterns),
            invalid_count=len(data_list) - len(patterns),
            patterns=patterns,
            errors=all_errors,
            warnings=all_warnings,
        )
    
    def _check_quality(self, pattern: Pattern) -> List[str]:
        """Проверить качество паттерна."""
        warnings = []
        
        # Короткое описание
        if len(pattern.description) < 50:
            warnings.append(f"Short description ({len(pattern.description)} chars)")
        
        # Нет тегов
        if not pattern.tags:
            warnings.append("No tags provided")
        
        # Нет категории
        if pattern.category is None:
            warnings.append("No category specified")
        
        # Unknown outcome
        if pattern.outcome == Outcome.UNKNOWN:
            warnings.append("Unknown outcome")
        
        # Нет lessons
        if not pattern.lessons:
            warnings.append("No lessons extracted")
        
        # Нет контекста
        if pattern.context is None:
            warnings.append("No context provided")
        
        return warnings
    
    def reset(self) -> None:
        """Сбросить состояние валидатора."""
        self._seen_ids.clear()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_pattern(data: Dict[str, Any], strict: bool = False) -> ValidationResult:
    """Удобная функция для валидации одного паттерна."""
    validator = PatternValidator(strict=strict)
    return validator.validate(data)


def validate_patterns(
    data_list: List[Dict[str, Any]],
    strict: bool = False,
) -> BatchValidationResult:
    """Удобная функция для валидации пачки паттернов."""
    validator = PatternValidator(strict=strict)
    return validator.validate_batch(data_list)


def validate_pattern_file(
    file_path: Union[str, Path],
    strict: bool = False,
) -> BatchValidationResult:
    """Валидировать файл с паттернами."""
    import json
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return BatchValidationResult(
            total=0,
            valid_count=0,
            invalid_count=0,
            errors=[ValidationError(
                field='file',
                message=f"File not found: {file_path}",
            )],
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Может быть список или один паттерн
        if isinstance(data, list):
            return validate_patterns(data, strict=strict)
        else:
            result = validate_pattern(data, strict=strict)
            return BatchValidationResult(
                total=1,
                valid_count=1 if result.valid else 0,
                invalid_count=0 if result.valid else 1,
                patterns=[result.pattern] if result.valid else [],
                errors=result.errors,
                warnings=result.warnings,
            )
            
    except json.JSONDecodeError as e:
        return BatchValidationResult(
            total=0,
            valid_count=0,
            invalid_count=0,
            errors=[ValidationError(
                field='json',
                message=f"Invalid JSON: {e}",
            )],
        )