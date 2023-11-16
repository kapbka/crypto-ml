create or replace function get_run_sum_deals_by_minute_tab
-- returns individual deals with running values of crypto and usd
(
    p_currency_code    currency.code%type,          -- currency.code
    p_version_arr      text[],                      -- array of version.id passed as text from grafana
    p_md5_arr          text[],                      -- array of individual.md5
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
    vt_start_ts          timestamp with time zone;
    vt_end_ts            timestamp with time zone;
    --
    vi_grafana_row_limit constant integer := 1000000;
    vi_count             integer := 0;
    vi_slice_interval    integer;
    vb_skip_record       boolean;
    --
    va_versions          int[] := p_version_arr::int[];
    vi_version           integer;
    vt_md5               text;
    vf_init_usd          float;
    vf_crypto            float;
    vf_usd               float;
    vf_percent           float;
    vf_run_crypto        float;
    vf_run_usd           float;
    vf_run_percent       float;
    vf_last_buy_ts       timestamp with time zone;
    vf_last_sell_ts      timestamp with time zone;
    --
    cr_deals             record;
    r                    deal_t;
begin
    select least(min(buy_ts), p_start_ts),
           greatest(max(coalesce(sell_ts, clock_timestamp())), p_end_ts)
      into vt_start_ts,
           vt_end_ts
      from deal d
     where d.currency = p_currency_code
       and d.version = any(p_version_arr::int[])
       and d.individual in
           (
               select id
                 from individual i
                where i.md5 = any(p_md5_arr)
           )
       and (
               (d.buy_ts between p_start_ts and p_end_ts
                or
                d.buy_ts < p_end_ts and coalesce(d.sell_ts,p_end_ts) between p_start_ts and p_end_ts
               )
               or
               (p_start_ts between d.buy_ts and coalesce(d.sell_ts,p_end_ts) and p_end_ts between d.buy_ts and coalesce(d.sell_ts,p_end_ts) )
           );

    vi_slice_interval := ceiling(extract(epoch from (vt_end_ts - vt_start_ts))/60 * array_length(p_version_arr, 1) * array_length(p_md5_arr, 1) / vi_grafana_row_limit);

    foreach vi_version in array va_versions
    loop
        foreach vt_md5 in array p_md5_arr
        loop
            vf_last_buy_ts := null;
            vf_last_sell_ts := null;

            for cr_deals in
            (
                with prices as
                (
                    select *
                      from price p
                     where 1 = 1
                       and p.currency = p_currency_code
                       and p.ts between vt_start_ts and vt_end_ts
                ),
                deals as
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
                           case
                               -- if a deal is open at the end of the interval
                               -- we set sell_ts as the end of the interval
                               when d.sell_price is null or d.sell_ts > vt_end_ts then
                                   date_trunc('minute', vt_end_ts)
                               else
                                   d.sell_ts
                           end as sell_ts,
                           case
                               -- if a deal is open at the end of the interval
                               -- we get the price for the end of the interval
                               when d.sell_price is null or d.sell_ts > vt_end_ts then
                                   (
                                       select close
                                         from price p
                                        where p.currency = p_currency_code
                                          and p.ts = date_trunc('minute', vt_end_ts)
                                   )
                               else
                                   d.sell_price
                           end as sell_price,
                           row_number() over(order by d.buy_ts) as row_num
                      from deal d,
                           individual i
                     where 1 = 1
                       and d.currency = p_currency_code
                       and d.version = vi_version
                       and i.md5 = vt_md5
                       and d.individual = i.id
                       and (
                               d.buy_ts between vt_start_ts and vt_end_ts
                               or
                               d.buy_ts < vt_end_ts and coalesce(d.sell_ts,vt_end_ts) between vt_start_ts and vt_end_ts
                           )
                )
                select p_currency_code as currency,
                       coalesce(d.version, vi_version) as version,
                       d.individual as individual,
                       d.id,
                       coalesce(d.md5, vt_md5) as md5,
                       p.ts,
                       p.close,
                       p.ts as buy_ts,
                       p.close as buy_price,
                       p.ts as sell_ts,
                       p.close as sell_price,
                       --
                       d.buy_ts as last_buy_ts,
                       d.buy_price as last_buy_price,
                       --
                       d.sell_ts as last_sell_ts,
                       d.sell_price as last_sell_price,
                       row_number() over(order by p.ts) as row_num
                  from prices p
                  left join deals d
                    on p.ts = d.buy_ts
            )
            loop
                vi_count := vi_count + 1;

                vb_skip_record := (mod(vi_count, vi_slice_interval) != 0);

                -- first deal in the interval
                if cr_deals.row_num = 1 then
                    vf_init_usd := get_init_usd
                                    (
                                        p_currency_code := cr_deals.currency,
                                        p_start_ts      := vt_start_ts
                                    );
                    vf_usd := vf_init_usd;
                    vf_run_usd := vf_init_usd;

                    if p_is_debug = 1 then
                        raise notice 'version %, init_usd %', cr_deals.version, vf_init_usd;
                    end if;
                end if;

                -- running values of crypto, usd and percent
                -- before any deals were made at all
                -- or
                -- there were previous deals and we are OUTSIDE the deal
                if (cr_deals.row_num = 1 and cr_deals.id is null) or cr_deals.ts > coalesce(cr_deals.last_sell_ts, vf_last_sell_ts) then
                    vf_crypto := coalesce(vf_crypto, 0.999);
                    vf_usd := coalesce(vf_usd, vf_init_usd);
                    vf_percent := coalesce(vf_percent, 0);
                -- there were previous deals and we are at buying point the deal
                elsif cr_deals.ts = cr_deals.last_buy_ts then
                    vf_crypto := vf_run_usd  * (1 - p_commission/100) / cr_deals.last_buy_price;
                    vf_usd := vf_crypto * cr_deals.sell_price * (1 - p_commission/100);
                    vf_percent := (vf_usd - vf_init_usd) / vf_init_usd * 100;
                    -- we don't skip buy and sell points, will skip the next minute
                    if vb_skip_record then
                        vb_skip_record := false;
                    end if;
                -- there were previous deals and we are INSIDE the deal
                elsif cr_deals.ts > vf_last_buy_ts and cr_deals.ts < vf_last_sell_ts then
                    -- the amount of the crypto remains the same, we didn't sell yet
                    vf_usd := vf_crypto * cr_deals.sell_price * (1 - p_commission/100);
                    vf_percent := (vf_usd - vf_init_usd) / vf_init_usd * 100;
                -- there were previous deals and we are at selling point
                elsif cr_deals.ts = vf_last_sell_ts then
                    vf_crypto := vf_run_crypto;
                    vf_usd := vf_run_usd;
                    vf_percent := vf_run_percent;
                    -- we don't skip buy and sell points, will skip the next minute
                    if vb_skip_record then
                        vb_skip_record := false;
                    end if;
                end if;

                if cr_deals.id is not null then
                    vf_last_buy_ts := cr_deals.last_buy_ts;
                    vf_last_sell_ts := cr_deals.last_sell_ts;
                    --
                    if p_is_debug = 1 then
                        raise notice 'before calc vf_run_usd %', vf_run_usd;
                        raise notice 'before calc cr_deals.last_buy_price %', cr_deals.last_buy_price;
                        raise notice 'before calc cr_deals.last_sell_price %', cr_deals.last_sell_price;
                    end if;
                    --
                    vf_run_crypto := vf_run_usd * (1 - p_commission/100) / cr_deals.last_buy_price;
                    vf_run_usd := vf_run_crypto * cr_deals.last_sell_price * (1 - p_commission/100);
                    vf_run_percent := (vf_run_usd - vf_init_usd) / vf_init_usd * 100;
                    --
                    if p_is_debug = 1 then
                        raise notice 'vf_run_crypto %, vf_run_usd %', vf_run_crypto, vf_run_usd;
                    end if;
                end if;

                if p_is_debug = 1 then
                    raise notice 'deal_id %, crypto %, perc %, usd %', cr_deals.id, vf_crypto, vf_percent, vf_usd;
                end if;

                -- return this record to the queue
                if not vb_skip_record then
                    r.currency_code := cr_deals.currency;
                    r.version_id    := cr_deals.version;
                    r.id            := cr_deals.id;
                    r.individual_id := cr_deals.individual;
                    r.md5           := cr_deals.md5;
                    r.buy_ts        := cr_deals.buy_ts;
                    r.buy_price     := cr_deals.buy_price;
                    r.sell_ts       := cr_deals.sell_ts;
                    r.sell_price    := cr_deals.sell_price;
                    --
                    r.crypto        := vf_crypto;
                    r.usd           := vf_usd;
                    r.percent       := vf_percent;
                    --
                    return next r;
                end if;
            end loop; -- deals
       end loop; -- md5
    end loop; -- versions
    --
    return;
end;
$$;