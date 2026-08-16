-- ============================================================================
-- LocateMe — Supabase PostgreSQL + pgvector Schema (Phase 4)
-- Real-Time Missing Person Detection & Candidate Vector Screening
-- ============================================================================

-- 1. Enable the pgvector extension for 512-D facial feature embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the missing_persons registry table
CREATE TABLE IF NOT EXISTS public.missing_persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    image_url TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

-- 3. Create an HNSW Index for sub-millisecond Cosine Similarity Search
-- (Uses vector_cosine_ops matching the InceptionResnetV1 unit-normalized vector space)
CREATE INDEX IF NOT EXISTS missing_persons_embedding_hnsw_idx
ON public.missing_persons
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Create an RPC Stored Procedure for Vectorized Similarity Matching
-- Calculates Cosine Similarity: 1 - Cosine Distance (<=> operator)
CREATE OR REPLACE FUNCTION match_missing_persons(
    query_embedding vector(512),
    match_threshold double precision DEFAULT 0.68,
    match_count integer DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    person_id TEXT,
    name TEXT,
    image_url TEXT,
    similarity_score double precision,
    metadata JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        mp.id,
        mp.person_id,
        mp.name,
        mp.image_url,
        (1 - (mp.embedding <=> query_embedding))::double precision AS similarity_score,
        mp.metadata,
        mp.created_at
    FROM public.missing_persons mp
    WHERE (1 - (mp.embedding <=> query_embedding)) >= match_threshold
    ORDER BY mp.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- 5. Set up Storage Bucket for Reference Photos
INSERT INTO storage.buckets (id, name, public)
VALUES ('missing-persons-photos', 'missing-persons-photos', true)
ON CONFLICT (id) DO NOTHING;

-- 6. Storage Security Policies (Public Read, Authenticated/Service-Role Write)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'objects' AND policyname = 'Public Read Missing Person Reference Photos'
    ) THEN
        CREATE POLICY "Public Read Missing Person Reference Photos"
        ON storage.objects FOR SELECT
        USING (bucket_id = 'missing-persons-photos');
    END IF;
END $$;
