"""add share access

Revision ID: 7d3bb29ce674
Revises: 6d2bb29ce673
Create Date: 2024-03-01 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = '7d3bb29ce674'
down_revision = '6d2bb29ce673'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Check if table exists
    connection = op.get_bind()
    table_exists = connection.execute(
        text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'share_access')")
    ).scalar()
    
    if not table_exists:
        # Create share_access table
        op.create_table(
            'share_access',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('document_id', sa.Integer(), nullable=False),
            sa.Column('shared_by_id', sa.Integer(), nullable=False),
            sa.Column('shared_with_email', sa.String(), nullable=False),
            sa.Column('access_level', sa.Enum('VIEW', 'EDIT', name='accesslevel', create_type=False), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['shared_by_id'], ['users.id']),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('idx_share_access_email', 'share_access', ['shared_with_email'])

def downgrade() -> None:
    # Don't drop anything in downgrade since table exists in Supabase
    pass