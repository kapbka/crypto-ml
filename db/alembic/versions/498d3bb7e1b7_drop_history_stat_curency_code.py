"""drop history_stat.curency_code

Revision ID: 498d3bb7e1b7
Revises: 205644baa644
Create Date: 2022-01-11 12:21:53.400584

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '498d3bb7e1b7'
down_revision = '205644baa644'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column('history_stat', 'currency_code')
