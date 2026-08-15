from __future__ import annotations

import abc
import typing
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


def UserConfigurable(*args, **kwargs):
    extra = kwargs.pop("json_schema_extra", None)
    if extra is None:
        extra = {"user_configurable": True}
    elif isinstance(extra, dict):
        extra = {**extra, "user_configurable": True}
    return Field(*args, **kwargs, json_schema_extra=extra)


class SystemConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    def get_user_config(self) -> dict[str, Any]:
        return _get_user_config_fields(self)


class SystemSettings(BaseModel):
    """A base class for all system settings."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    name: str
    description: str


S = TypeVar("S", bound=SystemSettings)


class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""

    prefix: str = ""
    default_settings: typing.ClassVar[S]

    @classmethod
    def get_user_config(cls) -> dict[str, Any]:
        return _get_user_config_fields(cls.default_settings)

    @classmethod
    def build_agent_configuration(cls, configuration: dict) -> S:
        """Process the configuration for this object."""

        defaults = cls.default_settings.model_dump()
        final_configuration = deep_update(defaults, configuration)

        return cls.default_settings.__class__.model_validate(final_configuration)


def _get_user_config_fields(instance: BaseModel) -> dict[str, Any]:
    """
    Get the user config fields of a Pydantic model instance.

    Args:
        instance: The Pydantic model instance.

    Returns:
        The user config fields of the instance.
    """
    user_config_fields = {}

    for name, value in instance.__dict__.items():
        field_info = instance.model_fields[name]
        extra = field_info.json_schema_extra or {}
        if isinstance(extra, dict) and extra.get("user_configurable"):
            user_config_fields[name] = value
        elif isinstance(value, SystemConfiguration):
            user_config_fields[name] = value.get_user_config()
        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            user_config_fields[name] = [i.get_user_config() for i in value]
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            user_config_fields[name] = {
                k: v.get_user_config() for k, v in value.items()
            }

    return user_config_fields


def deep_update(original_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a dictionary.

    Args:
        original_dict (dict): The dictionary to be updated.
        update_dict (dict): The dictionary to update with.

    Returns:
        dict: The updated dictionary.
    """
    for key, value in update_dict.items():
        if (
            key in original_dict
            and isinstance(original_dict[key], dict)
            and isinstance(value, dict)
        ):
            original_dict[key] = deep_update(original_dict[key], value)
        else:
            original_dict[key] = value
    return original_dict
