from unittest.mock import patch

import pytest
from asynctest.mock import MagicMock

from common.reporting import send_message


@pytest.mark.asyncio
async def test_simple_notification():
    with patch('requests.get', new=MagicMock()) as send:
        send_message(text='hello from UT', chats=[220583423], bot_id="test_id")
        send.assert_called_with('https://api.telegram.org/bottest_id/sendMessage',
                                params={'chat_id': 220583423, 'text': 'hello from UT'})


@pytest.mark.asyncio
@patch('requests.get', new=MagicMock(side_effect=Exception('fail')))
async def test_failure():
    send_message(text='hello from UT', chats=[220583423])
