import json
import logging
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, List

import numpy as np
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import Update
from aiogram.utils.callback_data import CallbackData

from common.k8s.probes import mark_alive, mark_ready
from common.constants import DEFAULT_CURRENCY, DealStatus, RealtimeStatus, HistoryStatus, LABEL_COLUMNS, \
    TELEGRAM_USERS, CURRENT_VERSION
from common.deal_generator import DealGenerator
from common.exchanges.binance_api import Binance
from common.log import setup_logging
from common.plot_tools import plot_to_file
from common.portfolio import deals_to_trades, run_portfolio_detailed
from db.api import DB
from db.data_cache import DataCache
from db.model import Price, Individual, IndividualAttribute
from models.ann import max_offset

ORDER_FIELDS = ['price', 'side', 'origQty']


@dataclass
class ChatState:
    currency: str = DEFAULT_CURRENCY
    version: int = CURRENT_VERSION


# Initialize bot and dispatcher
bot = Bot(token=os.environ['TELEGRAM_STATUS_BOT_ID'])
dp = Dispatcher(bot)
chats = defaultdict(ChatState)
bots_cb = CallbackData('bot', 'md5', 'action', 'days')
version_cb = CallbackData('version', 'id', 'action')


@dp.message_handler(commands=['version'], user_id=TELEGRAM_USERS)
async def get_versions(message: types.Message):
    markup = types.InlineKeyboardMarkup()
    with DB() as db:
        for version in db.version.get_batch():
            markup.add(
                types.InlineKeyboardButton(
                    f"{version.id} - {version.name}",
                    callback_data=version_cb.new(id=version.id, action='set')),
            )
    await message.reply(f'Select version', reply_markup=markup)


@dp.callback_query_handler(version_cb.filter(action='set'), user_id=TELEGRAM_USERS)
async def set_version_handler(query: types.CallbackQuery, callback_data: Dict[str, str]):
    await query.answer(f"Setting version to {callback_data['id']}")
    chats[query.message.chat.id].version = int(callback_data['id'])
    await query.message.edit_text(f"Successfully set version to {callback_data['id']}")


@dp.message_handler(commands=['status'], user_id=TELEGRAM_USERS)
async def send_status(message: types.Message):
    """
    This handler will be called when user sends `/status` command
    """
    state = chats[message.chat.id]

    async with Binance(ticker=f"{state.currency.upper()}USDT") as exchange, DB() as db:
        portfolio_individuals_with_attrs = db.portfolio.get_realtime_members_with_attrs(
            currency_code=state.currency,
            version_id=state.version
        )

        pieces = []
        for individual, attr in portfolio_individuals_with_attrs:

            deal = db.deal.get_last(currency=state.currency, individual=individual.id, version=attr.version_id)

            pieces.extend([
                str(individual.md5),
                str(deal),
            ])

            if deal and deal.status == DealStatus.Open:
                profit = (exchange.average_price - deal.buy_price) * 100 / deal.buy_price
                pieces.append(f"Last deal profit: {profit:.1f}%")

        for order in await exchange.get_orders():
            order = await exchange.get_order(order)

            reduced = [f"{f}: {order[f]}" for f in ORDER_FIELDS]
            pieces.append(f"Order: {','.join(reduced)}")

        currency, usd = await exchange.get_balance()

        pieces.append(f"Last action: {db.action.get(state.currency)}")
        pieces.append(f"{state.currency.upper()} balance: {currency:.3f}, USD balance: {usd:.1f}$, "
                      f"average price: {exchange.average_price:.1f}$")
        pieces.append(f"Total in USD: {exchange.average_price * currency + usd:.1f}$")

        await message.reply("\n".join(pieces))


async def send_graph_with_days(message: types.Message, days: int, md5: Optional[str]):
    """
    This handler will be called when user sends `/graph` command
    """
    state = chats[message.chat.id]

    with DB() as db:
        portfolio_individuals_with_attrs: List[Tuple[Individual, IndividualAttribute]] = \
            db.portfolio.get_realtime_members_with_attrs(
                currency_code=state.currency,
                version_id=state.version,
                portfolio_md5=md5
        )
        anns = list(map(db.ann.get, map(lambda x: x[0].ann_id, portfolio_individuals_with_attrs)))

        offset = max(map(lambda x: max_offset(json.loads(x.indicators), json.loads(x.offsets)), anns))
        interval = timedelta(days=days)
        start_ts = datetime.now(tz=timezone.utc) - interval

        df_all = DataCache(db, Price, state.currency).load(from_ts=start_ts - timedelta(minutes=offset))[LABEL_COLUMNS]
        trades = []
        for individual, attrs in portfolio_individuals_with_attrs:
            generator = DealGenerator(db=db, attributes=attrs, read_only=True)
            deals, _, _ = generator.process(current_usd=0, df=df_all, ts_from=start_ts)

            individual_trades = deals_to_trades(deals)
            if individual_trades:
                trades.append(individual_trades)

        ts = np.array([np.datetime64(t).astype(int) for t in df_all.index.values])
        data = np.concatenate(trades) if trades else np.array([])
        data = data[data[:, 3].argsort(kind='stable')] if trades else data
        money = run_portfolio_detailed(
            trades=data,
            prices=df_all['close'].values,
            all_ts=ts,
            total_shares=len(portfolio_individuals_with_attrs)
        ) if len(data) else np.zeros((len(ts), 2))

        df_all['money'] = [m[0] for m in money]
        df_all['shares'] = [m[1] for m in money]
        profit = f"{(df_all['money'][-1] - df_all['money'][0]) * 100 / df_all['money'][0]:.1f}%"

        df_graph = df_all.tail(len(money)).copy()
        graph_path = f"/tmp/{message.chat.id}_graph.png"
        plot_to_file(df=df_graph, bots=['money'], graph_path=graph_path, size_x=20, size_y=10, cols=['shares'])

        close = df_graph['close'].values
        buy_and_hold = (close[-1] - close[0]) * 100 / close[0]

        text = f"Profit for {str(interval).replace(', 0:00:00', '')} is {profit}, " \
               f"buy and hold: {buy_and_hold:.1f}%."
        with open(graph_path, 'rb') as photo:
            await message.reply_photo(photo, caption=text)


@dp.message_handler(commands=['graph'], user_id=TELEGRAM_USERS)
async def send_graph(message: types.Message):
    await send_graph_with_days(message, days=int(message.get_args()) if message.get_args() else 1, md5=None)


def get_keyboard(md5: str, chat_id: int):
    state = chats[chat_id]

    with DB() as db:
        portfolio_individuals_with_attrs = db.portfolio.get_realtime_members_with_attrs(
            currency_code=state.currency,
            version_id=state.version
        )
        individuals = {i[0].md5: i[1].version_id for i in portfolio_individuals_with_attrs}

    markup = types.InlineKeyboardMarkup()

    if individuals.get(md5, 0) != state.version:
        markup.add(
            types.InlineKeyboardButton(
                f"enable version {state.version}",
                callback_data=bots_cb.new(md5=md5, action='enable', days=0)),
        )
    if individuals.get(md5, 0) == state.version:
        markup.add(
            types.InlineKeyboardButton(
                f"disable version {state.version}",
                callback_data=bots_cb.new(md5=md5, action='disable', days=0)),
        )
    for day in [1, 7, 30, 90]:
        markup.add(
            types.InlineKeyboardButton(
                f"graph for {day} day{'s' if day > 1 else ''}",
                callback_data=bots_cb.new(md5=md5, action='graph', days=day)),
        )
    return markup


async def set_realtime_attribute(md5: str, currency: str, realtime_enabled: bool, version: int) -> str:
    with DB() as db:
        individual = db.individual.get_db(md5)

        existing = db.individual.attribute.get(currency_code=currency, version_id=version, md5_or_id=md5)
        db.individual.attribute.set(
            currency_code=currency,
            version_id=version,
            individual=individual,
            realtime_enabled=RealtimeStatus.Enabled.value if realtime_enabled else RealtimeStatus.Disabled.value,
            history_enabled=HistoryStatus.Disabled.value if realtime_enabled else HistoryStatus.Enabled.value,
            is_optimized=existing.is_optimized if existing else 0,
            priority=existing.priority if existing else 0,
            share=1 if realtime_enabled else 0,
            scaler_id=existing.scaler_id,
            oco_buy_percent=existing.oco_buy_percent,
            oco_sell_percent=existing.oco_sell_percent,
            oco_rise_percent=existing.oco_rise_percent,
        )

        portfolio_individuals_with_attrs = db.portfolio.get_realtime_members_with_attrs(
            currency_code=currency,
            version_id=version
        )
        res = []
        for individual, attrs in portfolio_individuals_with_attrs:
            res.append(f"{individual}, attrs: {attrs}")

        return f"New real-time individuals: \n{', '.join(res)}"


@dp.message_handler(regexp='^[a-f0-9]{2,}', user_id=TELEGRAM_USERS)
async def handle_bot_md5(message: types.Message):
    state = chats[message.chat.id]
    if len(message.text) == 32:
        await message.reply(f'Select action for {message.text} version {state.version}',
                            reply_markup=get_keyboard(message.text, message.chat.id))
    else:
        with DB() as db:
            bots = db.individual.search(message.text, limit=10)
            if not bots:
                await message.reply(f'Unable to find individuals matching {message.text}')
                return

            markup = types.InlineKeyboardMarkup()
            for individual in bots:
                markup.add(
                    types.InlineKeyboardButton(
                        individual.md5,
                        callback_data=bots_cb.new(md5=individual.md5, action='pick', days=0)),
                )
            await message.reply(f'Pick full md5 matching {message.text}', reply_markup=markup)


@dp.callback_query_handler(bots_cb.filter(action='pick'), user_id=TELEGRAM_USERS)
async def pick_bot(query: types.CallbackQuery, callback_data: Dict[str, str]):
    await query.answer(f"Working with {callback_data['md5']}")
    await query.message.edit_text(f"Select action for {callback_data['md5']}",
                                  reply_markup=get_keyboard(callback_data['md5'], query.message.chat.id))


@dp.callback_query_handler(bots_cb.filter(action='enable'), user_id=TELEGRAM_USERS)
async def enable_bot(query: types.CallbackQuery, callback_data: Dict[str, str]):
    state = chats[query.message.chat.id]

    await query.answer(f"Enabling {callback_data['md5']}")
    result = await set_realtime_attribute(callback_data['md5'],
                                          currency=state.currency,
                                          realtime_enabled=True,
                                          version=state.version)
    await query.message.edit_text(result)


@dp.callback_query_handler(bots_cb.filter(action='disable'), user_id=TELEGRAM_USERS)
async def disable_bot(query: types.CallbackQuery, callback_data: Dict[str, str]):
    state = chats[query.message.chat.id]

    await query.answer(f"Disabling {callback_data['md5']}")
    result = await set_realtime_attribute(callback_data['md5'],
                                          currency=state.currency,
                                          realtime_enabled=False,
                                          version=state.version)
    await query.message.edit_text(result)


@dp.callback_query_handler(bots_cb.filter(action='graph'), user_id=TELEGRAM_USERS)
async def send_graph_for_bot(query: types.CallbackQuery, callback_data: Dict[str, str]):
    await query.answer(f"Processing {callback_data['days']} days graph for {callback_data['md5']}")
    await send_graph_with_days(query.message, days=int(callback_data['days']), md5=callback_data['md5'])


@dp.message_handler(commands=['enabled'], user_id=TELEGRAM_USERS)
async def show_realtime_enabled(message: types.Message):
    state = chats[message.chat.id]

    with DB() as db:
        portfolio_individuals_with_attrs = db.portfolio.get_realtime_members_with_attrs(
            currency_code=state.currency,
            version_id=state.version
        )
        res = []
        for individual, attr in portfolio_individuals_with_attrs:
            res.append(f"{individual}, version: {attr.version_id}")
        await message.reply('\n'.join(res) if res else "Nobody is enabled")


@dp.errors_handler()
async def error_handler(update: Update, error: Exception):
    logging.exception(f"Handler failed")
    if update.message:
        text = '\n'.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__))
        await update.message.reply(f"Unable to handle your query, error: {text}")
    return True


if __name__ == '__main__':
    setup_logging(filename='/tmp/status-bot.log')
    mark_alive()
    mark_ready()
    executor.start_polling(dp, skip_updates=True)
