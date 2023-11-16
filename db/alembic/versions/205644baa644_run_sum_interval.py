"""run sum interval

Revision ID: 205644baa644
Revises: 5b71c65f92c9
Create Date: 2022-01-06 16:44:56.347872

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '205644baa644'
down_revision = '5b71c65f92c9'
branch_labels = None
depends_on = None


def upgrade():
    # drop
    op.drop_table("individual_run_sum")
    op.drop_table("individual_run_sum_header")
