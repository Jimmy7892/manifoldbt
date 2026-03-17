"""Exception hierarchy for manifoldbt."""


class BacktesterError(Exception):
    """Base exception for all manifoldbt errors."""


class DataError(BacktesterError):
    """Raised when data loading, versioning, or format issues occur."""


class StrategyError(BacktesterError):
    """Raised when strategy compilation or validation fails."""


class ConfigError(BacktesterError):
    """Raised when backtest configuration is invalid."""


class LicenseError(BacktesterError):
    """Raised when a Pro feature is used without a valid license."""
