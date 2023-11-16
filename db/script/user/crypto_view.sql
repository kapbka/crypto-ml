create user crypto_view with password 'F&qbW)=`d3h>8@)r';
grant connect on database DB_NAME to crypto_view;
alter role crypto_view in database DB_NAME set search_path = public;
-- tables
grant select on all tables in schema public to crypto_view;
alter default privileges in schema public grant select on tables to crypto_view;
-- functions
grant execute on all functions in schema public to crypto_view;
alter default privileges in schema public grant execute on functions to crypto_view;
-- procedures
grant execute on all procedures in schema public to crypto_view;