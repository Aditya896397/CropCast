# ============================================================
# CROP PRICE FORECASTING SYSTEM — CONFIG
# ============================================================

import os

# ── Supabase ──────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://odmxuhtjpgomzwzwejoy.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbXh1aHRqcGdvbXp3endlam95Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIxNjc0NTIsImV4cCI6MjA4Nzc0MzQ1Mn0.Qv_swXi8zbH-4xz_ujGw8QUja7uHH3FYhWzGti0nM3Y")

# ── Crops & Files ─────────────────────────────────────────
CROPS = {
   'Wheat':       'data/wheat_gujarat.csv',
    'Tomato':      'data/tomato_gujarat.csv',
    'Onion':       'data/onion_gujarat.csv',
    'Wheat_MH':    'data/wheat_maharashtra.csv',
    'Tomato_MH':   'data/tomato_maharashtra.csv',
    'Onion_MH':    'data/onion_maharashtra.csv',
}

# ── Model Settings ────────────────────────────────────────
TRAIN_RATIO      = 0.80
TARGET_COL       = "Modal_Price"
DATE_COL         = "Price_Date"
DATE_FORMAT      = "%d/%m/%Y"

# ARIMAX orders per crop (p, d, q)
ARIMAX_ORDERS = {
    "Wheat":      (2, 1, 2),
    "Tomato":     (3, 1, 2),
    "Onion":      (2, 1, 2),
    "Wheat_MH":   (2, 1, 2),
    "Tomato_MH":  (3, 1, 2),
    "Onion_MH":   (2, 1, 2),
}

# ── Gujarat Climate Normals ───────────────────────────────
GUJARAT_CLIMATE = {
    1:(2,28,12),   2:(2,31,14),   3:(4,36,19),
    4:(5,40,24),   5:(15,42,27),  6:(100,38,28),
    7:(280,33,26), 8:(250,32,25), 9:(120,33,25),
    10:(25,35,22), 11:(8,32,17),  12:(3,29,13)
}

# ── Maharashtra Climate Normals ───────────────────────────
MAHARASHTRA_CLIMATE = {
    1:(1,31,14),   2:(1,33,16),   3:(3,36,20),
    4:(12,38,24),  5:(40,37,26),  6:(280,32,25),
    7:(460,29,24), 8:(380,29,24), 9:(200,30,23),
    10:(80,32,21), 11:(30,32,17), 12:(8,31,14)
}

# ── Climate map ───────────────────────────────────────────
CLIMATE_MAP = {
    'Wheat':      GUJARAT_CLIMATE,
    'Tomato':     GUJARAT_CLIMATE,
    'Onion':      GUJARAT_CLIMATE,
    'Wheat_MH':   MAHARASHTRA_CLIMATE,
    'Tomato_MH':  MAHARASHTRA_CLIMATE,
    'Onion_MH':   MAHARASHTRA_CLIMATE,
}


# Neural Network (MLPRegressor) settings
NN_HIDDEN_LAYERS  = (128, 64, 32)
NN_MAX_ITER       = 500
NN_LEARNING_RATE  = 0.001
NN_RANDOM_STATE   = 42

# ── Feature Engineering ───────────────────────────────────
LAG_DAYS          = [1, 3, 7, 14, 30]
ROLLING_WINDOWS   = [7, 14, 30]

# ── Paths ─────────────────────────────────────────────────
OUTPUT_DIR        = "outputs"
MODEL_DIR         = "models"
