"""Exception hierarchy of the simulator.

``MarsEvaError``
    ``ConfigurationError``            a parameter set is inconsistent
    ``ValidationError``               input data failed an integrity check
        ``ProfileValidationError``    astronaut / suit profile
        ``TerrainValidationError``    terrain provider
        ``EnvironmentValidationError`` environment sample
    ``SimulationError``               the model cannot continue
        ``InfeasibleStateError``      a state violates a hard physical limit
        ``RouteNotFoundError``        no route satisfies the constraints
"""

from __future__ import annotations

from typing import Any


class MarsEvaError(Exception):
    """Base class of every error raised by this package."""


class ConfigurationError(MarsEvaError):
    """A configuration value is out of its valid range or contradicts another one."""


class ValidationError(MarsEvaError, ValueError):
    """Input data failed validation."""


class ProfileValidationError(ValidationError):
    """An astronaut or suit profile holds a non-physical value."""


class TerrainValidationError(ValidationError):
    """A terrain provider returned inconsistent or out-of-range data."""


class EnvironmentValidationError(ValidationError):
    """An environment provider returned a non-physical sample."""


class SimulationError(MarsEvaError):
    """The simulation reached a state it cannot continue from."""


class InfeasibleStateError(SimulationError):
    """A state violates a hard limit (for example a start cell that is not walkable)."""


class RouteNotFoundError(SimulationError):
    """No route satisfies the constraints.

    Attributes:
        diagnostics: the search telemetry (``SearchDiagnostics``) explaining which
            constraint rejected the candidates, or ``None`` when unavailable.
    """

    def __init__(self, message: str, diagnostics: Any | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
