"""recreate documents table

Revision ID: recreate_documents
Revises: xxxx
Create Date: 2024-03-19
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic
revision = 'recreate_documents'
down_revision = None  # Replace with your previous migration ID
branch_labels = None
depends_on = None
def upgrade() -> None:
    # Drop dependent table first
    op.drop_table('document_embeddings')

    # Drop existing documents table
    op.drop_table('documents')

    # Recreate documents table with correct schema
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('file_type', sa.String(), nullable=False),
        sa.Column('metadata_col', postgresql.JSON(), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)

    # Recreate document_embeddings table with the correct reference
    op.create_table(
        'document_embeddings',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE')),
        sa.Column('embedding_data', postgresql.ARRAY(sa.Float()), nullable=False)
    )

def downgrade() -> None:
    op.drop_table('document_embeddings')
    op.drop_table('documents')
