create or replace function get_run_sum_deals_tab
-- returns individual deals with running values of crypto and usd
(
    p_currency_code    currency.code%type,          -- currency.code
    p_version_arr      text[],                      -- array of version.id passed as text from grafana
    p_individ_id       individual.id%type,          -- individual.id
    p_start_ts         timestamp with time zone,    -- start of the interval
    p_end_ts           timestamp with time zone,    -- end of the interval
    p_commission       float default 0.1,           -- deal commission percent
    -- system
    p_is_debug         integer default 0            -- 0 - off, 1 - on
)
returns setof deal_t
language plpgsql
as
$$
declare
    vf_init_usd    float;
    vf_crypto      float;
    vf_usd         float;
    vf_percent     float;
    --
    cr_deals    record;
    r           deal_t%rowtype;
begin
    for cr_deals in
    (
        select d.currency,
               d.version,
               --
               d.id,
               --
               d.individual,
               i.md5,
               --
               d.buy_ts,
               d.buy_price,
               --
               d.sell_ts,
               case
                   -- if a deal is open at the end of the interval
                   -- we get the price for the end of the interval
                   when d.sell_price is null then
                       (
                           select close
                             from price p
                            where p.currency = p_currency_code
                              and p.ts = date_trunc('minute', p_end_ts)
                       )
                   else
                       d.sell_price
               end as sell_price,
               row_number() over(partition by d.currency, d.version order by d.buy_ts) as row_num
          from deal d,
               individual i
         where 1 = 1
           and d.currency = p_currency_code
           and d.version = any(p_version_arr::int[])
           and d.individual = p_individ_id
           and (
                   (d.buy_ts between p_start_ts and p_end_ts
                    or
                    d.buy_ts < p_end_ts and coalesce(d.sell_ts,p_end_ts) between p_start_ts and p_end_ts
                   )
                   or
                   (p_start_ts between d.buy_ts and coalesce(d.sell_ts,p_end_ts) and p_end_ts between d.buy_ts and coalesce(d.sell_ts,p_end_ts) )
               )
           and d.individual = i.id
    )
    loop
        r.currency_code := cr_deals.currency;
        r.version_id    := cr_deals.version;
        r.id            := cr_deals.id;
        r.individual_id := cr_deals.individual;
        r.md5           := cr_deals.md5;
        r.buy_ts        := cr_deals.buy_ts;
        r.buy_price     := cr_deals.buy_price;
        r.sell_ts       := cr_deals.sell_ts;
        r.sell_price    := cr_deals.sell_price;

        -- first deal in the interval
        if cr_deals.row_num = 1 then
            vf_init_usd := get_init_usd
                           (
                               p_currency_code := cr_deals.currency,
                               p_start_ts      := p_start_ts
                           );
            vf_usd := vf_init_usd;
            if p_is_debug = 1 then
                raise notice 'version %, init_usd %', cr_deals.version, vf_init_usd;
            end if;
        end if;

        -- running values of crypto, usd and percent
        vf_crypto := vf_usd * (1 - p_commission/100) / cr_deals.buy_price;
        vf_usd := vf_crypto * cr_deals.sell_price * (1 - p_commission/100);
        vf_percent := (vf_usd - vf_init_usd) / vf_init_usd * 100;
        --
        r.crypto := vf_crypto;
        r.usd := vf_usd;
        r.percent := vf_percent;

        if p_is_debug = 1 then
            raise notice 'deal_id %, crypto %, perc %, usd %', cr_deals.id, vf_crypto, vf_percent, vf_usd;
        end if;

        -- return this record to the queue
        return next r;
    end loop;
    --
    return;
end;
$$;