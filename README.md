# Low Light Photo Enhancement Using U-Net

This project implements a deep learning-based solution to enhance low-light images using a **U-Net architecture**. The U-Net model is designed to improve the quality of images taken in poorly lit environments by increasing brightness and preserving detail without overexposure.

## Features

- **U-Net Architecture**: A convolutional neural network specifically tailored for image enhancement tasks.
- **Image Preprocessing**: Efficient data augmentation techniques to prepare low-light images for training.
- **Real-Time Enhancement**: Enhance low-light images with high precision in real-time.
- **Preservation of Details**: Balances brightness and contrast, ensuring that image details are preserved.
- **Custom Loss Function**: Optimized loss function to minimize artifacts and enhance image clarity.

## Model Architecture

The U-Net model is widely used for image segmentation tasks but can be adapted for image enhancement. In this project, the U-Net consists of:

- **Encoder Path**: Extracts low-level features using convolutional and pooling layers.
- **Bottleneck**: Captures the context of the image at the deepest level of the network.
- **Decoder Path**: Reconstructs the image from the extracted features, using up-sampling and skip connections for detailed restoration.

## Datasets

- **Low-Light Images**: The model is trained on a dataset of low-light images that have corresponding ground truth images for comparison and model optimization.
- **Image Augmentation**: Data augmentation techniques such as flipping, rotation, and brightness adjustments are applied to increase the diversity of the training data.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Low-Light-Enhancement.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd Low-Light-Enhancement
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset:**
   - Place your low-light image dataset and corresponding ground truth images in the `data/` directory.
   - The directory structure should look like this:
     ```
     data/
       ├── low_light/
       └── ground_truth/
     ```

5. **Train the U-Net model:**
   ```bash
   python train.py
   ```

6. **Test the model on new low-light images:**
   ```bash
   python test.py --input path_to_low_light_image --output path_to_output_image
   ```

## Usage

- **Training the Model**: The training script will read images from the `data/low_light/` and `data/ground_truth/` directories. The U-Net model will be trained using a custom loss function designed to enhance the brightness while maintaining details.
- **Testing the Model**: You can pass any low-light image through the trained model to generate an enhanced image.
- **Real-Time Enhancement**: The model can process images in real time, making it suitable for applications requiring on-the-fly image enhancement.

## Results

The enhanced images produced by the U-Net model will have:

- Improved brightness and clarity.
- Minimal noise or artifacts.
- Enhanced details without overexposure.

## Dependencies

- Python 3.x
- TensorFlow or PyTorch (choose based on your implementation)
- OpenCV
- NumPy
- Matplotlib

## Future Improvements

- Explore other deep learning architectures such as GANs for further improvement.
- Implement real-time video enhancement.
- Fine-tune the model for various lighting conditions.

