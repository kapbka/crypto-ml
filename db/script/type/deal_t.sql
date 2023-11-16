create type deal_t as
(
    currency_code varchar(4),
    version_id    integer,
    --
    id            integer,
    --
    individual_id integer,
    md5           varchar(32),
    --
    buy_ts        timestamp,
    buy_price     float,
    --
    sell_ts       timestamp,
    sell_price    float,
    -- running result in crypto and usd
    crypto        float,
    usd           float,
    percent       float
);
grant usage on type deal_t to crypto_view;