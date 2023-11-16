import pytest
from asynctest.mock import CoroutineMock, patch, call, MagicMock

from workers.trader import Trader


@pytest.mark.asyncio
@patch('workers.trader.send_message', new=MagicMock())
async def test_shares():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(0, 100)),
                        average_price=100)

    async with Trader(api=api) as trader:

        trader._currency, trader._usd = 0, 100
        crypto = await trader.get_share(price=100)
        assert crypto == 0

        trader._currency, trader._usd = 1, 0
        crypto = await trader.get_share(price=100)
        assert crypto == 1

        trader._currency, trader._usd = 0.5, 50
        crypto = await trader.get_share(price=100)
        assert crypto == 0.5

        trader._currency, trader._usd = 1, 50
        crypto = await trader.get_share(price=100)
        assert crypto == 2 / 3


@pytest.mark.asyncio
@patch('workers.trader.send_message', new=MagicMock())
async def test_sync_buy_all_shares():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(0, 100)),
                        average_price=100,
                        buy=CoroutineMock(side_effect=[1]),
                        cancel_order=CoroutineMock(),
                        get_order=CoroutineMock(return_value=dict(status='FILLED')))

    mock = MagicMock()
    mock.__aiter__.return_value = [(1, 100)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        await trader.sync_state(share=1, price=10)
        api.buy.assert_called_with(amount=100)


@pytest.mark.asyncio
@patch('workers.trader.send_message', new=MagicMock())
async def test_not_sync_shares():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(0.00503704, 752.36408334)),
                        average_price=100,
                        buy=CoroutineMock(side_effect=[1]),
                        cancel_order=CoroutineMock())

    mock = MagicMock()
    mock.__aiter__.return_value = [dict(status='FILLED', id=1)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        await trader.sync_state(share=0.2, price=37602)
        api.buy.assert_not_called()


@pytest.mark.asyncio
@patch('workers.trader.send_message', new=MagicMock())
async def test_sync_buy_1_share():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(0, 100)),
                        average_price=100,
                        buy=CoroutineMock(side_effect=[1]),
                        cancel_order=CoroutineMock(),
                        get_order=CoroutineMock(return_value=dict(status='FILLED')))

    mock = MagicMock()
    mock.__aiter__.return_value = [dict(status='FILLED', id=1)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        await trader.sync_state(share=1/2, price=10)
        api.buy.assert_called_with(amount=100 / 2)


@pytest.mark.asyncio
@patch('workers.trader.send_message', new=MagicMock())
async def test_sync_sell_1_share():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(10, 0)),
                        average_price=100,
                        sell=CoroutineMock(side_effect=[1]),
                        cancel_order=CoroutineMock(),
                        get_order=CoroutineMock(return_value=dict(status='FILLED')))

    mock = MagicMock()
    mock.__aiter__.return_value = [dict(status='FILLED', id=1)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        await trader.sync_state(share=0.5, price=10)
        api.sell.assert_called_with(amount=5)


@pytest.mark.asyncio
@patch('workers.trader.send_message', new=MagicMock())
async def test_limit_orders():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[dict(status='PENDING')]),
                        get_balance=CoroutineMock(return_value=(3, 0)),  # 3 shares in currency, 0 in usd
                        sell_oco=CoroutineMock(side_effect=[1, 2, 3]),
                        average_price=100,
                        cancel_order=CoroutineMock())

    mock = MagicMock()
    mock.__aiter__.return_value = [dict(status='FILLED', id=1)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        await trader.sync_limit_orders(limit_prices={(0, 0, 10, 5): 0.5,
                                                     (0, 0, 11, 4): 0.3,
                                                     (0, 0, 12, 3): 0.2},
                                       price=10)

        assert call(amount=3 * 0.5, lower=5, upper=10) in api.sell_oco.call_args_list
        assert call(amount=3 * 0.3, lower=4, upper=11) in api.sell_oco.call_args_list
        assert call(amount=3 * 0.2, lower=3, upper=12) in api.sell_oco.call_args_list


@pytest.mark.asyncio
async def test_exception_in_send():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(0, 100)),
                        average_price=100,
                        buy=CoroutineMock(side_effect=[1]),
                        cancel_order=CoroutineMock(),
                        get_order=CoroutineMock(return_value=dict(status='FILLED')))

    mock = MagicMock()
    mock.__aiter__.return_value = [dict(status='FILLED', id=1)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        with patch('workers.trader.send_message', new=CoroutineMock(side_effect=Exception('fail'))):
            await trader.sync_state(share=1, price=10)
            api.buy.assert_called_with(amount=100)


@pytest.mark.asyncio
@patch.object(Trader, 'sync_state', new=CoroutineMock())
@patch.object(Trader, 'sync_limit_orders', new=CoroutineMock())
@patch('workers.trader.send_message', new=MagicMock())
async def test_handler():
    api = CoroutineMock(get_orders=CoroutineMock(return_value=[]),
                        get_balance=CoroutineMock(return_value=(0, 10)),
                        average_price=100,
                        buy=CoroutineMock(side_effect=[1]),
                        cancel_order=CoroutineMock())

    mock = MagicMock()
    mock.__aiter__.return_value = [dict(status='FILLED', id=1)]
    api.subscribe_to_user_updates.return_value = mock

    async with Trader(api=api) as trader:
        await trader.handle(limit_prices={(0, 0, 10, 5): 0.5,
                                          (0, 0, 11, 4): 0.3,
                                          (0, 0, 12, 3): 0.2},
                            price=10)
        api.get_balance.assert_called()
