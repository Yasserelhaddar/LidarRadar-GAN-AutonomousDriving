import torch
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\CycleGan_train_dataset_2^14"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
NUM_WORKERS = 0
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "critic.pth.tar"


#transforms = ToTensorV2()
    
