# **Low-Dose Chest X-ray Super-Resolution Using SRGAN**

## **Overview**

This project leverages **Super-Resolution Generative Adversarial Networks (SRGANs)** to enhance low-resolution **chest X-ray images**, improving their clarity and usability for diagnostic and research purposes.

The model is trained and fine-tuned on medical imaging datasets to improve spatial resolution while preserving anatomical details. The web-based interface allows users to upload low-resolution X-ray images and retrieve super-resolved outputs in real time.

## **Project Architecture**

The following diagram outlines the end-to-end pipeline of this project:

```
+---------------------------+       +---------------------------+       +---------------------------+
|  User Uploads X-ray       | ----> |  SRGAN Model Enhances    | ----> |  Download High-Res X-ray  |
|  (Low-Resolution)         |       |  Resolution & Structure  |       |  for Diagnosis/Research   |
+---------------------------+       +---------------------------+       +---------------------------+
```

### **Workflow**

1. **Input Processing**: The low-resolution X-ray is uploaded via the web interface.
2. **Super-Resolution Generation**: The SRGAN model processes the image and generates a high-resolution output.
3. **Output Visualization & Download**: The user receives an enhanced image for further analysis.

## **Technical Details**

The model was fine-tuned to make it compatible with grayscale medical image datasets, ensuring better performance on chest X-ray images.

- **Model**: Super-Resolution GAN (SRGAN)
- **Dataset**: NIH Chest X-ray Dataset
- **Frameworks**: PyTorch, Flask
- **Frontend**: HTML, CSS (for basic UI)
- **Deployment**: Render / Local Deployment

### **Model Performance Metrics**

| Metric                                 | Score    |
| -------------------------------------- | -------- |
| **Peak Signal-to-Noise Ratio (PSNR)**  | 42.29 dB |
| **Structural Similarity Index (SSIM)** | 0.9810   |
| **Mean Squared Error (MSE)**           | 0.0002   |
| **L1 Loss**                            | 0.0099   |
| **Perceptual Loss**                    | 0.0254   |

## **Dataset & Preprocessing**

- **Source**: [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Preprocessing Steps**:
  - Convert images to grayscale
  - Resize to **64×64 (low-resolution)** and **256×256 (high-resolution)**
  - Normalize images to **[-1,1]** for model input
  - Train/test split: **80% training, 20% validation**

## **Results & Visualization**

### **Super-Resolution Comparison**

Below are sample results from the model:

| **Low-Resolution Input** | **Super-Resolved Output** | **Ground Truth (High-Resolution)** |
| ------------------------ | ------------------------- | ---------------------------------- |
| ![Low-Res](../mnt/data/low_res_image.png) | ![Super-Resolved](../mnt/data/super_res_image.png) | ![High-Res](../mnt/data/high_res_image.png) |
| ------------------------ | ------------------------- | ---------------------------------- |
|                          |                           |                                    |

The model successfully restores fine details and anatomical structures, making low-quality X-rays more interpretable for medical professionals.

## **Deployment & Web Interface**

The application is deployed as a **Flask-based web app**, allowing easy access to the model’s inference capabilities.

### **Setup & Running Locally**

To run this project locally, follow these steps:

#### **1. Clone the Repository**

```bash
git clone https://github.com/SphurtiMW/Image_SuperResolution.git
cd ImageSuperResolution
```

#### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **3. Run the Flask Application**

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to access the web app.

## **Deployment Instructions**

The application can be deployed using **Render** or any cloud service.

### **Steps for Deployment on Render**

1. Push the repository to GitHub.
2. Create a **New Web Service** on [Render](https://render.com).
3. Set the **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the **Start Command**:
   ```bash
   python app.py
   ```
5. Deploy and obtain a public **URL**.

## **Future Improvements**

- **Fine-tuning on domain-specific datasets** to further improve clinical relevance.
- **Integration with cloud-based PACS (Picture Archiving and Communication Systems)** for hospital deployment.
- **Automated batch processing** to enhance multiple images simultaneously.




 

