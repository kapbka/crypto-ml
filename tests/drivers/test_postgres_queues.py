import asyncio

import pytest

from common.interrupt import InterruptionHandler
from common.queues.postgres import Subscriber, Publisher, RPCHandler, RPCClient
from db.api.db import DB


@pytest.mark.asyncio
async def test_postgres_pub_sub():
    async with \
            DB() as db1, \
            DB() as db2, \
            DB() as db3, \
            DB() as db4, \
            InterruptionHandler() as guard, \
            Subscriber(db1, guard, "btc.price", uid="price_1") as sub_price1, \
            Subscriber(db2, guard, "btc.price", uid="price_2") as sub_price2, \
            Subscriber(db3, guard, "btc.input", uid="price_3") as sub_input, \
            Publisher(db4, "btc.price") as pub_btc_price, \
            Publisher(db4, "btc.input") as pub_btc_input, \
            Publisher(db4, "xrp.price") as pub_xrp_price, \
            Publisher(db4, "xrp.input") as pub_xrp_input:

        pub_btc_price.send('pub_btc_price')
        pub_btc_input.send('pub_btc_input')
        pub_xrp_price.send('pub_xrp_price')
        pub_xrp_input.send('pub_xrp_input')

        assert await sub_price1.get() == 'pub_btc_price'
        assert await sub_price2.get() == 'pub_btc_price'
        assert await sub_input.get() == 'pub_btc_input'


@pytest.mark.asyncio
async def test_postgres_rpc():
    test_name = 'test_call'

    def handler(x: int, y: int):
        return x * y

    for i in range(5):
        async with \
                DB() as db1, \
                DB() as db2, \
                InterruptionHandler() as guard, \
                RPCHandler(db=db1, guard=guard, name=test_name, func=handler) as server, \
                RPCClient(db=db2, guard=guard, name=test_name) as multiply:
            task = asyncio.ensure_future(server.run())

            size = 10
            tasks = [multiply(x=i, y=i) for i in range(i, size)]
            assert await asyncio.gather(*tasks) == [i**2 for i in range(i, size)]
            task.cancel()


class CustomException(Exception):
    pass


@pytest.mark.asyncio
async def test_postgres_exception():
    test_name = 'test_call_2'

    def handler():
        raise CustomException('hehe')

    async with \
            DB() as db1, \
            DB() as db2, \
            InterruptionHandler() as guard, \
            RPCHandler(db=db1, guard=guard, name=test_name, func=handler) as server, \
            RPCClient(db=db2, guard=guard, name=test_name) as multiply:
        task = asyncio.ensure_future(server.run())
        with pytest.raises(CustomException):
            await multiply()
        task.cancel()


@pytest.mark.asyncio
async def test_postgres_rpc_multiple_handlers():
    test_name = 'test_call_3'

    async with \
            DB() as db1, \
            DB() as db2, \
            DB() as db3, \
            DB() as db4, \
            InterruptionHandler() as guard, \
            RPCHandler(db=db1, guard=guard, name=test_name, uid='1', func=lambda: 1) as server1, \
            RPCHandler(db=db2, guard=guard, name=test_name, uid='2', func=lambda: 2) as server2, \
            RPCHandler(db=db3, guard=guard, name=test_name, uid='3', func=lambda: 3) as server3, \
            RPCClient(db=db4, guard=guard, name=test_name) as get_id:

        future = asyncio.ensure_future(asyncio.gather(*[get_id() for _ in range(3)]))
        await server1.run_once()
        await server2.run_once()
        await server3.run_once()

        assert await future == [1, 2, 3]

        # next call should time out as there are no jobs to handle
        with pytest.raises(asyncio.exceptions.TimeoutError):
            await asyncio.wait_for(server3.run_once(), 3)
