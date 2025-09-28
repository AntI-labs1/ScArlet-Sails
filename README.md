# Scarlet Sails

**Autonomous Trading System - Reborn from the Ashes**

Scarlet Sails is the successor to the KIT_RnD_Layer4 project, built on the lessons learned from a comprehensive failure analysis (Operation Valkyrie). This project embodies a disciplined approach to autonomous trading system development, prioritizing simplicity, reliability, and systematic risk management.

## Project Genesis

This project was born from the systematic analysis and controlled destruction of KIT_RnD_Layer4, a failed autonomous trading system that suffered from architectural complexity, scope creep, and lack of focus. Operation Valkyrie was conducted to extract valuable lessons and viable ideas from the wreckage.

## Core Principles

- **MVP First**: Every feature must work in its simplest form before expansion
- **Data→Signal→Risk→Exec**: All trading decisions follow this strict pipeline
- **No AI Black Boxes**: AI assists but never replaces human control
- **Disciplined Development**: Strict rules, quality gates, and kill criteria

## Architecture

Scarlet Sails follows a clean, modular monolithic architecture that can evolve into microservices when justified by scale (>100k LOC).

```
src/
├── data/          # Market data ingestion and processing
├── signals/       # Trading signal generation
├── risk/          # Risk management and position sizing
├── execution/     # Order execution and portfolio management
├── monitoring/    # System health and performance monitoring
└── strategies/    # Pluggable trading strategies
```

## Documentation

- `docs/valkyrie/` - Operation Valkyrie artifacts and lessons learned
- `docs/adr/` - Architecture Decision Records, including kill decisions
- `docs/api/` - API documentation (generated)
- `docs/strategies/` - Strategy documentation and backtests

## Development Rules

All development follows the strict rules defined in `docs/valkyrie/rules.yml`. Key principles:

- Maximum 3 active features in development
- 80% test coverage for critical path
- All operations must be idempotent and deterministic
- Mandatory backtest→stress test→paper trade→review cycle

## Risk Management

- Maximum 2% portfolio per trade
- Stop-loss mandatory for every position
- Daily drawdown limit 5%
- Kill switch stops all trading in 5 seconds

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Antihrist-star/scarlet-sails.git
cd scarlet-sails

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start in paper trading mode
python -m src.main --mode paper
```

## License

Private project - All rights reserved

## Memorial

This project stands as a memorial to KIT_RnD_Layer4 (2024-2025), whose failure taught us the value of discipline, focus, and systematic approach to complex system development. May its lessons guide us to success.

*"From the ashes of complexity, simplicity rises."*

