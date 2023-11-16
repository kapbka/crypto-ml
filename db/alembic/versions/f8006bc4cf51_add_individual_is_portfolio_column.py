"""add individual is portfolio column

Revision ID: f8006bc4cf51
Revises: 498d3bb7e1b7
Create Date: 2022-01-13 16:50:28.593257

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer


# revision identifiers, used by Alembic.
revision = 'f8006bc4cf51'
down_revision = '498d3bb7e1b7'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("individual") as batch_op:
        batch_op.add_column(Column('is_portfolio', Integer))
