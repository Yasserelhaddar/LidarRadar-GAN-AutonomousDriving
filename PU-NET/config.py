import torch
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\data_8192\\Dataset"
#VAL_DIR = "C:\\Users\\ElHaddar\\OneDrive\\Desktop\\Cycle GAN\\CycleGan_test_dataset_2^15"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_S = "genS.pth.tar"
CHECKPOINT_GEN_R = "genR.pth.tar"
CHECKPOINT_CRITIC_S = "criticS.pth.tar"
CHECKPOINT_CRITIC_R = "criticR.pth.tar"

#transforms = ToTensorV2()
    
