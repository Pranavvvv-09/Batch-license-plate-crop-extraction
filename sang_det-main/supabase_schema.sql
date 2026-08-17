-- =========================================================================
--  sang_det - Supabase PostgreSQL Schema & Storage Setup
--  Paste this entire file into your Supabase SQL Editor and click RUN.
-- =========================================================================

-- 1. Create Videos Table
CREATE TABLE IF NOT EXISTS videos (
    id                BIGSERIAL PRIMARY KEY,
    url               TEXT NOT NULL,
    title             TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    error             TEXT,
    frames_processed  BIGINT NOT NULL DEFAULT 0,
    plates_saved      BIGINT NOT NULL DEFAULT 0,
    detections_seen   BIGINT NOT NULL DEFAULT 0,
    rejected_conf     BIGINT NOT NULL DEFAULT 0,
    rejected_size     BIGINT NOT NULL DEFAULT 0,
    rejected_blur     BIGINT NOT NULL DEFAULT 0,
    rejected_dupe     BIGINT NOT NULL DEFAULT 0,
    duration_s        DOUBLE PRECISION,
    position_s        DOUBLE PRECISION NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 0,
    added_at          DOUBLE PRECISION NOT NULL,
    started_at        DOUBLE PRECISION,
    finished_at       DOUBLE PRECISION,
    heartbeat_at      DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_added_at ON videos(added_at);

-- 2. Create Plates Table (Holds 10,000+ Plate Crops and OCR Labels)
CREATE TABLE IF NOT EXISTS plates (
    id           BIGSERIAL PRIMARY KEY,
    video_id     BIGINT REFERENCES videos(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    frame_index  BIGINT NOT NULL DEFAULT 0,
    timestamp_s  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    confidence   DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    box          TEXT NOT NULL DEFAULT '[0,0,0,0]',
    width        INTEGER NOT NULL DEFAULT 0,
    height       INTEGER NOT NULL DEFAULT 0,
    blur_score   DOUBLE PRECISION,
    phash        TEXT,
    bytes        BIGINT,
    saved_at     DOUBLE PRECISION NOT NULL,
    plate_text   TEXT,
    ocr_status   TEXT NOT NULL DEFAULT 'unlabeled',
    labeled_at   DOUBLE PRECISION,
    storage_url  TEXT
);

CREATE INDEX IF NOT EXISTS idx_plates_video ON plates(video_id);
CREATE INDEX IF NOT EXISTS idx_plates_saved ON plates(saved_at);
CREATE INDEX IF NOT EXISTS idx_plates_status ON plates(ocr_status);
CREATE INDEX IF NOT EXISTS idx_plates_filename ON plates(filename);
CREATE INDEX IF NOT EXISTS idx_plates_plate_text ON plates(plate_text);

-- 3. Create Events Log Table
CREATE TABLE IF NOT EXISTS events (
    id        BIGSERIAL PRIMARY KEY,
    video_id  BIGINT,
    level     TEXT NOT NULL,
    message   TEXT NOT NULL,
    at        DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);

-- 4. Create Key-Value State Table
CREATE TABLE IF NOT EXISTS state (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

-- 5. Set Up Supabase Storage Bucket for Images / Plate Crops
-- Creates the 'plates' public bucket if not already present
INSERT INTO storage.buckets (id, name, public)
VALUES ('plates', 'plates', true)
ON CONFLICT (id) DO UPDATE SET public = true;

-- Allow public read access to the plates bucket
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'objects' AND schemaname = 'storage' AND policyname = 'Public Access for Plates'
    ) THEN
        CREATE POLICY "Public Access for Plates" 
        ON storage.objects FOR SELECT 
        USING (bucket_id = 'plates');
    END IF;
END $$;

-- Allow authenticated / service role insert & update
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'objects' AND schemaname = 'storage' AND policyname = 'Allow Uploads for Plates'
    ) THEN
        CREATE POLICY "Allow Uploads for Plates" 
        ON storage.objects FOR INSERT 
        WITH CHECK (bucket_id = 'plates');
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'objects' AND schemaname = 'storage' AND policyname = 'Allow Updates for Plates'
    ) THEN
        CREATE POLICY "Allow Updates for Plates" 
        ON storage.objects FOR UPDATE 
        USING (bucket_id = 'plates');
    END IF;
END $$;
