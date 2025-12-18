# PROJECT STRUCTURE
Date: 2024-12-18

> Актуальная структура репозитория после cleanup (commit 3dccacc)

## ScArlet-Sails/

### Core Modules (core/)
Основные модули данных, фичей, риск-менеджмента и бектестинга.

- **feature_engine_v2.py** — ACTIVE: основной feature engine
- **feature_engine.py** — LEGACY: старая версия, используется только в main.py
- **data_loader.py** — загрузка OHLCV данных
- **feature_loader.py** — загрузка сгенерированных фичей
- **canonical_state.py** — построение канонического состояния S(t)
- **ood_detector.py** — детектор out-of-distribution (Mahalanobis distance)
- **regime_detector.py** — классификация режимов волатильности (ATR-based)
- **rolling_dispersion.py** — дисперсия предсказаний P_rb, P_ml, P_hyb
- **dynamic_position_sizer.py** — динамическое управление позицией
- **sanitize_features.py** — валидация и очистка фичей
- **backtest_engine.py** — ядро бектестинга
- **metrics_calculator.py** — расчёт метрик производительности
- **position_sizer.py** — базовое управление позицией
- **models.py** — Pydantic схемы и data models
- **trade_logger.py** — логирование сделок

### Components (components/)
Переиспользуемые компоненты высокого уровня.

- **opportunity_scorer.py** — скоринг торговых возможностей (P_rb)
- **advanced_risk_penalty.py** — продвинутые риск-пенальти

### Strategies (strategies/)
Торговые стратегии — три основных агента системы.

**ACTIVE:**
- **rule_based_v2.py** — P_rb(S): rule-based стратегия
- **xgboost_ml_v3.py** — P_ml(S): основная ML-стратегия
- **hybrid_q_learner.py** — P_hyb(S): Q-learning гибрид
- **simple_strategies.py** — baseline / вспомогательные стратегии

**DEPRECATED → archive/:**
- xgboost_ml.py → archive/deprecated_strategies/
- xgboost_ml_v2.py → archive/deprecated_strategies/
- hybrid_v2.py → archive/deprecated_hybrid/

### Council (council/)
LLM-based совет агентов для принятия решений.

- **base_agent.py**
- **pattern_detector.py**
- **recommendation.py**

### RAG (rag/)
RAG-хранилище знаний: паттерны, сделки, уроки.

- **patterns/** — исторические паттерны
- **trades/** — записи сделок
- **lessons/** — извлечённые уроки

### Analysis (analysis/)
Аналитика, визуализация, продвинутые бектесты.

- **walk_forward_validation.py**
- **backtest_dynamic_sizing.py**
- **backtest_real_prb.py**
- **dispersion_analyzer.py**
- **dispersion_visualizer.py**

### Backtesting (backtesting/)
Низкоуровневые сценарии бектестов.

### Scripts (scripts/)
CLI-скрипты: обучение, бектесты, утилиты.

- **train_xgboost_v3.py**
- **run_backtest.py**

### Interface (interface/)
Human-in-the-loop интерфейс (CLI/TUI).

### RL (rl/)
Reinforcement Learning окружения и агенты.

### Tests (tests/)
Unit и integration тесты.

**ACTIVE:**
- test_backtest_engine.py
- test_regime_position.py
- test_rolling_dispersion.py
- test_rule_based_strategy.py
- test_train_xgboost_v3_sanitize.py
- test_hybrid_q_learner.py
- test_xgboost_strategy.py

**DEPRECATED → archive/tests_legacy/:**
- _old_test_integration.py
- _old_test_real_data.py

### Archive (archive/)
Исторические/устаревшие модули.

- **deprecated_strategies/** — xgboost v1/v2
- **deprecated_hybrid/** — hybrid_v2
- **tests_legacy/** — старые тесты
- **core_legacy/** — будущий дом для feature_engine.py

### Data & Artifacts
- **data/** — DVC-трекируемые сырые данные
- **features/** — генерация фич (скрипты/спеки)
- **models/** — обученные модели (JSON/pkl)
- **backtest_results/** — результаты бектестов

### Documentation (docs/)
- **SYSTEM_ARCHITECTURE_DETAILED.md**
- **MATHEMATICAL_FRAMEWORK.md**
- **PHASE3_STATUS.md**

### Other
- **reports/** — отчёты по спринтам
- **inventory/** — автоматические инвентаризации
- **visualization/** — графики и визуализации
- **configs/** — конфигурационные файлы
- **checks/** — pre-commit checks / linting
- **run_backtest.py** — высокоуровневый раннер бектестов
- **orchestrator.py** — оркестрация пайплайна
- **main.py** — entry point / CLI

---

## Version Control

**Active Versions:**
- Feature Engine: v2
- XGBoost Strategy: v3
- Rule-Based: v2
- Hybrid: q_learner

**Deprecated & Archived:**
- См. DEPRECATED_FILES.md для полного списка
