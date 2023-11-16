"""update portfolio members

Revision ID: 2dae444a023d
Revises: f8006bc4cf51
Create Date: 2022-04-26 16:33:26.769117

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2dae444a023d'
down_revision = 'f8006bc4cf51'
branch_labels = None
depends_on = None


def upgrade():
    upd = """
    update individual_attribute ia
       set realtime_enabled = 0,
           history_enabled = 1
     where ia.md5 in
           (
               select p.md5
                 from portfolio p
                where p.portfolio_md5 = '2466d8fd515a7c6037a9b658b5426e11'
           )
       and ia.currency_code = 'btc'
    """
    op.execute(upd)
