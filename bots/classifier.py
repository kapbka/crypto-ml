import logging
import re
from functools import partial

from telegram import Update, Message, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
from textblob import TextBlob

from mining.db.api import connect
from mining.db.models import Tweet, TweetClass, TweetSentiment

logger = logging.getLogger(__name__)

ROWS = [[f'{sentiment.name}_{subj.name}' for sentiment in TweetSentiment]
        for subj in TweetClass if subj.value >= 0] + [['Skip']]

CURRENT_TWEETS = dict()


def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w+:\ / \ / \S+)", " ", tweet).split())


def get_next_tweet(chat_id: int):
    tweet = Tweet.objects(classification=None)[0]
    CURRENT_TWEETS[chat_id] = tweet
    return tweet.text


def mark_tweet(callback_data: str, update: Update, context: CallbackContext):
    parts = callback_data.split("_")
    if len(parts) == 2:
        sentiment = TweetSentiment[parts[0]].value
        classification = TweetClass[parts[1]].value
    else:
        assert callback_data == 'Skip'
        sentiment = TweetSentiment.Neutral
        classification = TweetClass.Skip

    Tweet.objects(id=CURRENT_TWEETS[update.callback_query.message.chat_id].id).update_one(
        set__classification=classification, set__sentiment=sentiment)

    show_next_tweet(get_next_tweet(update.callback_query.message.chat_id), update.callback_query.message)


def show_next_tweet(text: str, message: Message):
    keyboard = [[InlineKeyboardButton(" ".join(name.split("_")).capitalize(), callback_data=name)
                 for name in row] for row in ROWS]

    clean = clean_tweet(text)
    analysis = TextBlob(clean)
    message.reply_text(text + f"\npolarity: {analysis.sentiment.polarity}, "
                              f"subjectivity: {analysis.sentiment.subjectivity}",
                       reply_markup=InlineKeyboardMarkup(keyboard))


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext) -> None:
    show_next_tweet(get_next_tweet(update.message.chat_id), update.message)


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def main():
    # Enable logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
    )

    """Start the bot."""
    connect()

    # Create the Updater and pass it your bot's token.
    updater = Updater("1622427122:AAHcCdb1YiplWn4DCqTBJLT_PKEjv2zRDHw")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    for row in ROWS:
        for item in row:
            dispatcher.add_handler(CallbackQueryHandler(partial(mark_tweet, item), pattern=item))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()