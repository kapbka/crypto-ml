import pytest

from common.zk import ZooKeeper


@pytest.mark.asyncio
async def test_zk():
    async with ZooKeeper('test') as zk1:
        assert await zk1.get_shared_value([1, 2]) == 1
        async with ZooKeeper('test') as zk2:
            assert await zk2.get_shared_value([1, 2]) == 2
            async with ZooKeeper('test') as zk3:
                with pytest.raises(ValueError):
                    await zk3.get_shared_value([1, 2])

            async with ZooKeeper('test1') as zk4:
                assert await zk4.get_shared_value([1, 2]) == 1