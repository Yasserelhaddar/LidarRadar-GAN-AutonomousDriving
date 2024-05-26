import torch
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\Train_dataset"
VAL_DIR = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Autoencoder\\Test_dataset"
BATCH_SIZE = 1
LEARNING_RATE = 0.001
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_S = "model.pth.tar"


#transforms = ToTensorV2()
    
