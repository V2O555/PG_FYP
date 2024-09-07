# Dataset
Download the MaSTr1325 dataset from https://box.vicos.si/borja/viamaro/index.html

The image files should be put into the folder `data/images`, and the annotation files should be put into the folder `data/masks`.

# The Weight of the Trained Model
The trained model ShorelineNet_Std and ShorelineNet_Adv can be downloaded from https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabuvx_ucl_ac_uk/Esw7fHBR00RFhOI8xf2MyOQBcZ06TGEulRsWafHGxrOGYA?e=XklNGn

Download the trained model weights from the above webpage and place both files in the project folder `result`. File `model.pt` corresponds to ShorelineNet_Std as mentioned above while `model_attacked_optimised.pt` corresponds to ShorelineNet_Adv.


# Adversarial Attack Testing

This project demonstrates performing adversarial attacks on a neural network using different perturbation parameters. To test and visualize the effect of attacks, you can modify the parameters listed in the code snippet below and then run the `test.py` script.

## Parameters for Adversarial Attack

- **`epsilon`**: This parameter controls the magnitude of the perturbation. It represents how much noise will be added to the image for the adversarial attack. Increase or decrease this value to control the strength of the attack.
  
- **`attack_method`**: This parameter defines the type of attack. The values can be selected from `FGSM`, `BIM`, `PGD`, `Occlusion`, `Rotation`.

- **`alpha`**: This is only for BIM and PGD. The step size for each iteration of the attack. A smaller alpha means finer, more precise steps in the attack process.

- **`iteration`**: This is only for BIM and PGD. This parameter controls how many iterations the attack will perform. More iterations mean a stronger attack.

- **`delta`**: This is only for the perturbation magnitude of black-box attacks.

## How to Run

1. Open the Python script `test.py` in any code editor.
2. Modify the parameters shown in the code snippet above to test different configurations.
3. Save the changes.
4. Run the `test.py` script:

   ```bash
   python test.py

## Training Your Own Model

If you want to train the model from scratch, follow the instructions below. There are two options: training the base model or training a defense model using adversarial examples.

### Training the Base Model

To train the base model, you simply need to run the `train.py` script. You can modify hyperparameters like the number of epochs and other settings directly in the script.

1. Open `train.py` in your preferred code editor.
2. Modify the `epoch` parameter and any other hyperparameters like learning rate, batch size, etc.
3. Save the changes.
4. Run the following command to start the training:

   ```bash
   python train.py

### Training the Defense Model

To train a defense model that is robust to adversarial attacks, follow these steps:

1. First, generate adversarial examples by running the `generate.py` script. This will create a dataset of adversarial samples used for defense training.
   
   ```bash
   python generate.py
2. After the adversarial examples are generated, run the `train_attack.py` script to train the model using these examples. This will improve the model's robustness against attacks.

   ```bash
   python train_attack.py
