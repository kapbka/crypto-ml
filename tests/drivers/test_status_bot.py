import json
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from asynctest.mock import CoroutineMock, MagicMock

from bots.status import error_handler, send_status, send_graph, pick_bot, enable_bot, disable_bot, handle_bot_md5
from bots.status import get_versions, show_realtime_enabled, send_graph_for_bot, set_version_handler
from common.constants import DealStatus
from common.deal_generator import DealGenerator
from db.api.ann import DBAnn
from db.api.attribute import DBIndividualAttribute
from db.api.deal import DBDeal
from db.api.individual import DBIndividual
from db.api.portfolio import DBPortfolio
from db.api.version import DBVersion
from db.data_cache import DataCache
from db.model import Deal


class FakeBinance:
    def __init__(self, ticker: str):
        pass

    async def __aenter__(self):
        return CoroutineMock(average_price=100,
                             get_balance=CoroutineMock(return_value=(0, 100)),
                             get_orders=CoroutineMock(return_value=[1]),
                             get_order=CoroutineMock(return_value={'price': 10, 'side': 'SELL', 'origQty': 10}))

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


DEAL = Deal(status=DealStatus.Open, buy_price=10)


@pytest.mark.asyncio
@patch('bots.status.Binance', new=FakeBinance)
@patch.object(DBDeal, 'get_last', new=MagicMock(return_value=DEAL))
async def test_status():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    await send_status(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
@patch('bots.status.Binance', new=FakeBinance)
@patch.object(DBIndividual, 'get_realtime', new=MagicMock(return_value=[(MagicMock(), MagicMock())]))
@patch.object(DBDeal, 'get_last', new=MagicMock(return_value=DEAL))
async def test_exception():
    update = MagicMock(message=MagicMock(reply=CoroutineMock()))
    assert await error_handler(update, Exception())
    update.message.reply.assert_called_once()


@pytest.mark.asyncio
@patch.object(DealGenerator, 'process', new=MagicMock(return_value=([], 0, 0)))
@patch.object(DealGenerator, '__init__', new=MagicMock(return_value=None))
@patch.object(DataCache, 'load', new=MagicMock(return_value=pd.DataFrame({'close': [100, 120],
                                                                          'high': [150, 150],
                                                                          'low': [100, 100], },
                                                                         index=[datetime(2021, 1, 1, 0, 1, 1),
                                                                                datetime(2021, 1, 1, 1, 1, 1)])))
@patch.object(DBPortfolio, 'get_realtime', new=MagicMock(return_value=[MagicMock(individual_id=1, md5='12345')]))
@patch.object(DBPortfolio, 'get_batch', new=MagicMock(return_value=[MagicMock()]))
@patch.object(DBIndividualAttribute, 'get', new=MagicMock())
@patch.object(DBIndividual, 'get_db', new=MagicMock())
@patch.object(DBAnn, 'get', new=MagicMock(return_value=MagicMock(indicators=json.dumps({'rsi': [20]}),
                                                                 offsets=json.dumps([10]))))
@patch('bots.status.deals_to_trades', new=MagicMock(return_value=[np.array([0, 1., 500, 1245454])]))
@patch('bots.status.run_portfolio_detailed', new=MagicMock(return_value=[(500, 0.5), (300, 0.5)]))
@patch('bots.status.plot_to_file', new=MagicMock())
async def test_graph():
    msg = MagicMock(chat=MagicMock(id=1), reply_photo=CoroutineMock())
    with open(f"/tmp/{msg.chat.id}_graph.png", "w") as tmp:
        await send_graph(msg)
        msg.reply_photo.assert_called_once()


@pytest.mark.asyncio
@patch.object(DealGenerator, 'process', new=MagicMock(return_value=([], 0, 0)))
@patch.object(DealGenerator, '__init__', new=MagicMock(return_value=None))
@patch.object(DataCache, 'load', new=MagicMock(return_value=[pd.DataFrame()]))
@patch.object(DBIndividual, 'get_realtime', new=MagicMock(return_value=[]))
@patch.object(DBIndividual, 'get', new=MagicMock(return_value=MagicMock()))
@patch('bots.status.plot_to_file', new=MagicMock())
async def test_graph_no_bots():
    msg = MagicMock(chat=MagicMock(id=1), reply_photo=CoroutineMock())
    with pytest.raises(Exception):
        await send_graph(msg)


@pytest.mark.asyncio
@patch.object(DBIndividual, 'search', new=MagicMock(return_value=[]))
async def test_handle_bot_md5_short_no_matches():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    msg.text = 'c1a'
    await handle_bot_md5(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
@patch.object(DBIndividual, 'search', new=MagicMock(return_value=[MagicMock(md5='test')]))
async def test_handle_bot_md5_short():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    msg.text = 'c1a'
    await handle_bot_md5(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
@patch.object(DBIndividual, 'search', new=MagicMock(return_value=[]))
@patch.object(DBIndividual, 'get_realtime',
              new=MagicMock(return_value=[(MagicMock(md5='f6779780f64c85502b797025fd1926ec'), MagicMock(version_id=2))]))
async def test_handle_bot_md5_full_enabled():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    msg.text = 'f6779780f64c85502b797025fd1926ec'
    await handle_bot_md5(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
@patch.object(DBIndividual, 'search', new=MagicMock(return_value=[]))
@patch.object(DBIndividual, 'get_realtime', new=MagicMock(return_value=[]))
async def test_handle_bot_md5_full_disabled():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    msg.text = 'f6779780f64c85502b797025fd1926ec'
    await handle_bot_md5(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
async def test_pick_bot():
    msg = MagicMock(chat=MagicMock(id=1), edit_text=CoroutineMock())
    query = MagicMock(message=msg, answer=CoroutineMock())
    await pick_bot(query, callback_data={'md5': 'test'})
    query.answer.assert_called_once()
    msg.edit_text.assert_called_once()


@pytest.mark.asyncio
@patch.object(DBIndividual, 'get_realtime', new=MagicMock(return_value=[(MagicMock(), 1)]))
@patch.object(DBIndividual, 'get_db', new=MagicMock(return_value=MagicMock()))
@patch.object(DBIndividualAttribute, 'get', new=MagicMock(return_value=MagicMock()))
async def test_enable_bot():

    msg = MagicMock(chat=MagicMock(id=1), edit_text=CoroutineMock())
    query = MagicMock(message=msg, answer=CoroutineMock())
    with patch.object(DBIndividualAttribute, 'set', new=MagicMock()) as save:
        await enable_bot(query, callback_data={'md5': 'test'})

        query.answer.assert_called_once()
        msg.edit_text.assert_called_once()
        save.assert_called_once()


@pytest.mark.asyncio
@patch.object(DBIndividual, 'get_realtime', new=MagicMock(return_value=[(MagicMock(), 1)]))
@patch.object(DBIndividual, 'get_db', new=MagicMock(return_value=MagicMock()))
@patch.object(DBIndividualAttribute, 'get', new=MagicMock(return_value=MagicMock()))
async def test_disable_bot():
    msg = MagicMock(chat=MagicMock(id=1), edit_text=CoroutineMock())
    query = MagicMock(message=msg, answer=CoroutineMock())
    with patch.object(DBIndividualAttribute, 'set', new=MagicMock()) as save:
        await disable_bot(query, callback_data={'md5': 'test'})

        query.answer.assert_called_once()
        msg.edit_text.assert_called_once()
        save.assert_called_once()


@pytest.mark.asyncio
async def test_show_realtime_enabled():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    await show_realtime_enabled(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
async def test_show_graph_for_bot():
    msg = MagicMock(chat=MagicMock(id=1), reply_photo=CoroutineMock())
    query = MagicMock(message=msg, answer=CoroutineMock())

    with patch('bots.status.send_graph_with_days', new=CoroutineMock()) as send:
        await send_graph_for_bot(query, {'md5': 'test', 'days': '7'})
        send.assert_called_once()


@pytest.mark.asyncio
@patch.object(DBVersion, 'get_batch', new=MagicMock(return_value=[MagicMock()]))
async def test_get_versions():
    msg = MagicMock(chat=MagicMock(id=1), reply=CoroutineMock())
    await get_versions(msg)
    msg.reply.assert_called_once()


@pytest.mark.asyncio
async def test_set_version():
    msg = MagicMock(chat=MagicMock(id=1), edit_text=CoroutineMock())
    query = MagicMock(message=msg, answer=CoroutineMock())

    await set_version_handler(query, {'id': '1'})
    msg.edit_text.assert_called_once()
    query.answer.assert_called_once()
