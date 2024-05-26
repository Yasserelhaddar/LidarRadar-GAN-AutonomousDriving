
# LIDAR and RADAR Sensor Modeling and Data Augmentation with GANs for Autonomous Driving

## Project Overview

This project aims to enhance the perception module of autonomous vehicles using deep learning techniques. It addresses data scarcity by using simulators to generate point cloud (PCL) data and applying deep neural networks to translate simulated data to real-world scenarios. The main goals include modeling LIDAR and RADAR sensors, data augmentation with Generative Adversarial Networks (GANs), and implementing various neural network architectures for point cloud processing.


## Repository Structure

```plaintext
      .
      ├── CycleGAN
      │   ├── chamfer_loss.py
      │   ├── config.py
      │   ├── Data_process.py
      │   ├── data_read.py
      │   ├── dataset.py
      │   ├── discriminator_model.py
      │   ├── discriminator_model2.py
      │   ├── Discriminator.py
      │   ├── evaluate.py
      │   ├── functions.py
      │   ├── generator_model.py
      │   ├── generator_model2.py
      │   ├── Generator.py
      │   ├── loss.py
      │   ├── model1.py
      │   ├── npz.py
      │   ├── train.py
      │   ├── utils.py
      │
      ├── PU-NET
      │   ├── chamfer_loss.py
      │   ├── config.py
      │   ├── dataset.py
      │   ├── evaluate.py
      │   ├── functions.py
      │   ├── loss.py
      │   ├── model.py
      │   ├── Repulsion_loss.py
      │   ├── train.py
      │   ├── utils.py
      │
      ├── Autoencoder
      │   ├── PointNet_autoencoder
      │   │   ├── chamfer_loss.py
      │   │   ├── config.py
      │   │   ├── dataset.py
      │   │   ├── evaluate.py
      │   │   ├── functions.py
      │   │   ├── loss.py
      │   │   ├── model.py
      │   │   ├── model1.py
      │   │   ├── old_model.py
      │   │   ├── train.py
      │   │   ├── utils.py
      │   ├── VoxelNet_autoencoder
      │       ├── convolutional_autoencoder_32.py
      │       ├── voxelnet_based_autoencoder.py
      │       ├── pointnet_based_autoencoder_old_model.py
      │
      ├── Carla
      │   ├── Test_lidar_semantic.py
      │   ├── Test_lidar.py
      │   ├── Test_radar.py
      │   ├── test.py
      │
      ├── Conditional_GAN
      │   ├── chamfer_loss.py
      │   ├── config_CGAN.py
      │   ├── config.py
      │   ├── dataset_CGAN.py
      │   ├── dataset.py
      │   ├── discriminator_CGAN.py
      │   ├── discriminator.py
      │   ├── evaluate.py
      │   ├── functions.py
      │   ├── generator_CGAN.py
      │   ├── loss.py
      │   ├── train_CGAN.py
      │   ├── train.py
      │   ├── utils.py
      │
      └── Readme.txt

```

## Detailed Description

### CycleGAN

**Objective: Translate point cloud data between simulated and real-world domains.**
**Key Files:**
   1. chamfer_loss.py: Implements Chamfer loss function.
   2. config.py: Configuration of hyperparameters and model checkpoints.
   3. data_read.py: Reads and processes the dataset.
   4. discriminator_model.py, discriminator_model2.py: Defines the discriminator models.
   5. Discriminator.py: Main discriminator implementation.
   6. evaluate.py: Evaluates the trained model.
   7. functions.py: Utility functions for point cloud data processing.
   8. generator_model.py, generator_model2.py: Defines the generator models.
   9. Generator.py: Main generator implementation.
   10. train.py: Training script for CycleGAN.

### Conditional GAN

**Objective: Generate realistic point cloud data conditioned on specific inputs.**
**Key Files:**
   1. chamfer_loss.py: Implements Chamfer loss function.
   2. config_CGAN.py, config.py: Configuration of hyperparameters and model checkpoints.
   3. dataset_CGAN.py, dataset.py: Processes the dataset for training.
   4. discriminator_CGAN.py, discriminator.py: Defines the discriminator models.
   5. evaluate.py: Evaluates the trained Conditional GAN model.
   6. functions.py: Utility functions for data processing.
   7. generator_CGAN.py: Defines the generator model.
   8. loss.py: Implements loss functions.
   9. train_CGAN.py, train.py: Training scripts for Conditional GAN.


### PU-NET

**Objective: Upsample point cloud data to improve resolution.**
**Key Files:**
   1. chamfer_loss.py: Implements Chamfer loss function.
   2. config.py: Configuration of hyperparameters and model checkpoints.
   3. dataset.py: Processes the dataset for training.
   4. evaluate.py: Evaluates the trained PU-NET model.
   5. functions.py: Utility functions for data processing.
   6. loss.py: Implements loss functions including Chamfer and K-mean loss.
   7. model.py: Defines the PU-NET model.
   8. Repulsion_loss.py: Implements the repulsion loss function.
   9. train.py: Training script for PU-NET.

### Autoencoder

**Objective: Encode and decode point cloud data using neural networks.**
**Submodules:**

***PointNet Autoencoder:***
   1. chamfer_loss.py: Implements Chamfer loss function.
   2. config.py: Configuration of hyperparameters and model checkpoints.
   3. dataset.py: Processes the dataset for training.
   4. evaluate.py: Evaluates the trained PointNet autoencoder.
   5. functions.py: Utility functions for data processing.
   6. loss.py: Implements loss functions.
   7. model.py, model1.py, old_model.py: Defines various versions of the PointNet model.
   8. train.py: Training script for PointNet autoencoder.

***VoxelNet Autoencoder:***
   1. convolutional_autoencoder_32.py, voxelnet_based_autoencoder.py, pointnet_based_autoencoder_old_model.py: Implementations of VoxelNet autoencoder models.



## Installation and Usage

### Prerequisites

   1. Python 3.8 or higher
   2. Required libraries: torch, numpy, scipy, matplotlib, plyfile
   3. Additional libraries as specified in each submodule's requirements.

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up the environment**
   - Using `pip`:
     ```bash
     pip install -r torch, numpy, scipy, matplotlib, plyfile
     ```
     And install any extra required libraries in the submodules



## Running the Models

Each submodule contains its own set of instructions for running the models. For example:

1. **Training CycleGAN**
   ```bash
      cd CycleGAN
      python train.py
   ```

2. **Evaluating PU-NET**
   - Using `pip`:
     ```bash
      cd PU-NET
      python evaluate.py
     ```


## Contribution

Feel free to contribute by opening issues, submitting pull requests, or providing feedback.

## Acknowledgments

**Yasser El Haddar**
**Prof. Thomas Limbrunner**
**Mr. Konstantin Raab**
**Technical Institue of Deggendorf**

## License

This project is licensed under the MIT License.
