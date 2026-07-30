"""Project-specific exception hierarchy."""


class PdpUqError(Exception):
    """Base exception for expected pdp-uq failures."""


class ArtifactError(PdpUqError):
    """An external data or model artifact is missing, invalid, or unsafe."""


class DataValidationError(PdpUqError):
    """Input data does not satisfy the documented schema."""


class ConfigurationError(PdpUqError):
    """A project configuration is missing or invalid."""
