create or replace procedure calc_run_sum
(
    p_currency_code         currency.code%type,               -- currency.code
    p_version_id            version.id%type,                  -- version.id
    p_interval_id           run_sum_interval.id%type,         -- run_sum_interval.id
    --
    p_current_batch         integer,                          -- number of batch out of total number that needs to be run, starts with 0
    p_number_of_batches     integer             default 3,    -- total number of batches (should be no more than 3 so far, optimal value 2)
    --
    p_percent_commission    float               default 0.1,  -- percent of commission that will be taken for every order (buy/sell), defaulted to 0.1% (Binance exchange)
    --
    p_md5_arr               character varying[] default null, -- array of md5
    -- system
    p_process_mode          integer             default 0     -- 0 - write to a table, 1 - write to output using raise notice structure
)
language plpgsql
as
$$
declare
    vt_start_ts          timestamp;
    vt_end_ts            timestamp;
    vi_header_id         integer;
    vf_init_usd          float;
    vt_last_price_ts     timestamp;
    -- records for cursors
    cr_individ           record;
    cr_deals             record;
    -- cursors
    c_individ cursor
    (
        cp_number_of_batches    integer,
        cp_current_batch        integer,
        cp_md5_arr              character varying[]
    ) for
    select i.id,
           i.md5,
           count(*) over() as cnt
      from individual i
     where 1 = 1
       and mod(i.id, cp_number_of_batches) = cp_current_batch
       and (cp_md5_arr is null or i.md5 = any(cp_md5_arr))
     order by id;
begin
    if p_currency_code is null then
        raise 'Currency code is null!';
    end if;

    if p_version_id is null then
        raise 'Version ID is null!';
    end if;

    if p_interval_id is null then
        raise 'Interval ID is null!';
    end if;

    if p_current_batch >= p_number_of_batches then
        raise 'Current batch % should be consistent with total number of batches % and not exceed this value, starts with 0', p_current_batch, p_number_of_batches;
    end if;

    if p_percent_commission not between 0.0 and 100.0 then
        raise 'p_percent_commission should be between 0 and 100 values, passed value %', p_percent_commission;
    end if;

    if p_process_mode not in (0,1) then
        raise 'p_process_mode has to be 0 (table) or 1 (output)';
    end if;

    -- get header_id
    select rsh.start_ts,
           rsh.end_ts,
           rsh.id,
           rsh.init_usd
      into vt_start_ts,
           vt_end_ts,
           vi_header_id,
           vf_init_usd
      from run_sum_header rsh
     where 1 = 1
       and rsh.currency_code = p_currency_code
       and rsh.version_id = p_version_id
       and rsh.interval_id = p_interval_id;

    -- end date
    select max(ts) into vt_last_price_ts from price;
    vt_end_ts := least(date_trunc('minute', vt_end_ts), vt_last_price_ts);

    for cr_individ in c_individ
    (
        cp_number_of_batches := p_number_of_batches,
        cp_current_batch     := p_current_batch,
        cp_md5_arr           := p_md5_arr
    )
    loop
        if p_process_mode = 1 then
            raise notice '---- INDIVIDUAL % ---- ', cr_individ.md5;
        end if;

        for cr_deals in
        (
            select d.id,
                   d.buy_price,
                   d.sell_price,
                   d.crypto,
                   d.usd,
                   d.percent
              from get_run_sum_deals_tab
                   (
                       p_currency_code := p_currency_code,
                       p_version_arr   := array[p_version_id::text],
                       p_individ_id    := cr_individ.id,
                       p_start_ts      := vt_start_ts,
                       p_end_ts        := vt_end_ts,
                       p_is_debug      := p_process_mode
                   ) d
             order by d.buy_ts desc
             limit 1
        )
        loop
            if p_process_mode = 0 then
                insert into run_sum(header_id, individual_id, md5, usd, percent, crypto)
                values(vi_header_id, cr_individ.id, cr_individ.md5, cr_deals.usd, cr_deals.percent, cr_deals.crypto);
            elsif p_process_mode = 1 then
                raise notice '---- TOTAL % -----', cr_individ.md5;
                raise notice 'crypto %, perc %, usd %, init_usd %', cr_deals.crypto, cr_deals.percent, cr_deals.usd, vf_init_usd;
                raise notice ' ';
            end if;
        end loop; -- deal
    end loop; -- individual

end;
$$;