from __future__ import annotations

from pathlib import Path
from typing import Iterator

from autogpt.compat import json_lib
from autogpt.config import Config
from autogpt.logs import logger

from ..memory_item import MemoryItem
from .base import VectorMemoryProvider


class JSONFileMemory(VectorMemoryProvider):
    """Memory backend that stores memories in a JSON file"""

    SAVE_OPTIONS = json_lib.OPT_SERIALIZE_NUMPY | json_lib.OPT_SERIALIZE_DATACLASS

    file_path: Path
    memories: list[MemoryItem]

    def __init__(self, config: Config) -> None:
        """Initialize a class instance

        Args:
            config: Config object

        Returns:
            None
        """
        self.file_path = config.workspace_path / f"{config.memory_index}.json"
        logger.debug(
            f"Initialized {__class__.__name__} with index path {self.file_path}"
        )

        self.memories = []
        try:
            self.load_index()
            logger.debug(f"Loaded {len(self.memories)} MemoryItems from file")
        except Exception as e:
            logger.warn(f"Could not load MemoryItems from file: {e}")
            self.memories = []
            self.save_index()
        else:
            # Ensure a valid index exists (empty touch() files used to blow up on load).
            if (not self.file_path.is_file()) or self.file_path.stat().st_size == 0:
                self.save_index()

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(self.memories)

    def __contains__(self, x: MemoryItem) -> bool:
        return x in self.memories

    def __len__(self) -> int:
        return len(self.memories)

    def add(self, item: MemoryItem):
        self.memories.append(item)
        logger.debug(f"Adding item to memory: {item.dump()}")
        self.save_index()
        return len(self.memories)

    def discard(self, item: MemoryItem):
        try:
            self.remove(item)
        except:
            pass

    def clear(self):
        """Clears the data in memory."""
        self.memories.clear()
        self.save_index()

    def load_index(self):
        """Loads all memories from the index file"""
        if not self.file_path.is_file():
            logger.debug(f"Index file '{self.file_path}' does not exist")
            return
        raw = self.file_path.read_text(encoding="utf-8").strip()
        if not raw:
            # touch() / interrupted writes leave an empty file; treat as fresh index.
            logger.debug(f"Index file '{self.file_path}' is empty")
            return
        logger.debug(f"Loading memories from index file '{self.file_path}'")
        json_index = json_lib.loads(raw)
        if not isinstance(json_index, list):
            raise ValueError(
                f"Memory index must be a JSON list, got {type(json_index).__name__}"
            )
        for memory_item_dict in json_index:
            self.memories.append(MemoryItem(**memory_item_dict))

    def save_index(self):
        logger.debug(f"Saving memory index to file {self.file_path}")
        with self.file_path.open("wb") as f:
            return f.write(json_lib.dumps(self.memories, option=self.SAVE_OPTIONS))
