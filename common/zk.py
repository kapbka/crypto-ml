import os
from typing import List, Any
from typing import Optional
from uuid import uuid4

from aiozk import ZKClient

from common.constants import ZK_CONNECTION


class ZooKeeper:
    def __init__(self, name: str):
        self.zk = ZKClient(ZK_CONNECTION)
        self.name = name
        self.id = str(uuid4())
        self._lock = self.zk.recipes.Lock(f"/lock/{self.name}")
        self._path = f"/members/{self.name}"
        self._created: Optional[str] = None

    async def __aenter__(self):
        await self.zk.start()
        await self.zk.ensure_path(self._path)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._created:
            await self.zk.delete(self._created)
        await self.zk.close()

    async def get_shared_value(self, values: List[Any]) -> Any:
        async with self._lock:
            next_index = -1

            children = await self.zk.get_children(self._path)
            children = set(map(int, children))
            for idx in range(len(values)):
                if idx not in children:
                    next_index = idx
                    break

            if next_index == -1:
                raise ValueError(f"No free values left, children: {children}, values: {values}")

            self._created = await self.zk.create(
                os.path.join(self._path, str(next_index)), data=self.id, ephemeral=True)

            return values[next_index]
