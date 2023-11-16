import logging
import os
import sys

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy import text

from common.constants import DB_CONNECTION
from common.log import setup_logging
from db.api.db import DB
from db.model.data.insert_all import insert_all

SCHEMA_FILES = [
    # view
    'db/script/user/crypto_view.sql',
    # type
    'db/script/type/deal_t.sql',
    # function
    'db/script/function/array_reverse.sql',
    'db/script/function/get_init_usd.sql',
    'db/script/function/get_run_sum_deals_tab.sql',
    'db/script/function/get_run_sum_deals_by_minute_tab.sql',
    'db/script/function/generate_pivot_table.sql',
    # procedure
    'db/script/procedure/calc_run_sum.sql'
]


def create_schema(folder_path: str):
    with DB() as db:
        for file in SCHEMA_FILES:
            file_name = file.split('/')[-1]
            object_name = file_name.split('.')[0]
            object_type = file.split('/')[-2]

            if object_type == 'user':
                res = db.session.execute(text("select * from pg_roles p where rolname = :object_name limit 1;")
                                         .bindparams(object_name=object_name))
            elif object_type == 'type':
                res = db.session.execute(text("select * from pg_type p where typname = :object_name limit 1;")
                                         .bindparams(object_name=object_name))
            else:
                res = db.session.execute(text("select * from pg_proc p where proname = :object_name limit 1;")
                                         .bindparams(object_name=object_name))

            if object_type != 'user' and res.rowcount > 0:
                logging.info(f'Drop {object_name}')
                if object_type in ['function', 'procedure']:
                    db.session.execute(text(f"drop {object_type} if exists {object_name};"))
                else:
                    db.session.execute(text(f"drop {object_type} {object_name} cascade;"))

            if object_type != 'user' or res.rowcount == 0:
                logging.info(f'Executing {file_name}')
                with open(os.path.join(folder_path, file), 'r') as file_open:
                    db.session.execute(file_open.read().replace('DB_NAME', os.getenv('POSTGRES_DB')))


if __name__ == '__main__':
    setup_logging()

    script_dir = os.path.join(sys.argv[1], 'db/alembic')

    engine = create_engine(DB_CONNECTION)
    connection = engine.connect()
    context = MigrationContext.configure(connection)
    alembic_cfg = Config()
    alembic_cfg.set_main_option('script_location', script_dir)
    alembic_cfg.set_main_option('sqlalchemy.url', DB_CONNECTION)

    is_clean_db = not engine.dialect.has_table(connection, 'individual')

    if is_clean_db:
        script = ScriptDirectory.from_config(alembic_cfg)
        head = script.get_current_head()
        context.stamp(script, head)

    create_schema(sys.argv[1] if len(sys.argv) > 1 else '.')

    command.upgrade(alembic_cfg, 'head')

    # one more time if there were table drops in alembic version scripts
    create_schema(sys.argv[1] if len(sys.argv) > 1 else '.')

    if is_clean_db:
        insert_all()
