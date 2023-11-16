from search_engine_parser import GoogleSearch


def google(query):
    search_args = (query, 1)
    gsearch = GoogleSearch()
    gresults = gsearch.search(*search_args)
    return gresults['links']


def find_twitter_accounts():
    return google('bitcoin twitter accounts to follow') + \
           google('bitcoin news twitter') + \
           google('crypto news twitter') + \
           google('crypto twitter accounts to follow') + \
           google('stock market twitter accounts')
