import os


# SAVE_MODEL_DIR = "../../saved_models"
# SAVE_VIZ_DIR = "../../Visualizations"

# NUM_TEST = 60
# Data FOLDER
DATA = "../processed_data"
# RITE
RITE = os.path.join(DATA, "RITE")
RITE_IMG = os.path.join(RITE, "image")
RITE_ART = os.path.join(RITE, "artery")
RITE_VEIN = os.path.join(RITE, "vein")
RITE_VES = os.path.join(RITE, "vessel")

# FIVES
FIVES = os.path.join(DATA, "FIVES")
FIVES_IMG = os.path.join(FIVES, "image")
FIVES_ART = os.path.join(FIVES, "artery")
FIVES_VEIN = os.path.join(FIVES, "vein")
FIVES_VES = os.path.join(FIVES, "vessel")

# RMHAS
RMHAS = os.path.join(DATA, "RMHAS")
RMHAS_IMG = os.path.join(RMHAS, "image")
RMHAS_ART = os.path.join(RMHAS, "artery")
RMHAS_VEIN = os.path.join(RMHAS, "vein")
RMHAS_VES = os.path.join(RMHAS, "vessel")

# training

EPOCHS = 100
BUFFER_SIZE = 100
BATCH_SIZE = 2
VALIDATION_RATIO = 0.2
