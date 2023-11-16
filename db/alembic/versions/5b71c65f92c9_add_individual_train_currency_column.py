"""add individual.train_currency column

Revision ID: 5b71c65f92c9
Revises: 
Create Date: 2021-12-30 15:18:53.856834

"""
from alembic import op
from sqlalchemy import Column
from sqlalchemy import String


# revision identifiers, used by Alembic.
revision = '5b71c65f92c9'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("individual") as batch_op:
        batch_op.add_column(Column('train_currency', String(6)))
