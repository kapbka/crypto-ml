create or replace function get_init_usd
(
    p_currency_code    currency.code%type,      -- currency.code
    p_start_ts         timestamp with time zone -- the start date of the interval
)
returns float
language plpgsql
as
$$
declare
    vf_usd    float;
begin
    select close
      into vf_usd
      from price
     where currency = p_currency_code
       and ts = date_trunc('minute', p_start_ts);

   return vf_usd;

end;
$$;