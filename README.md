# DLM Capstone Repository

This repository contains the project materials for our capstone project.

## Clone the Repository
Run the following command to clone the repository:
```bash
git clone https://github.com/MouadMelouk/dlm-capstone.git
```

## Collaboration Guidelines

- Pull the latest changes:
```bash
git pull origin main
```

- Add changes and push:
```bash
git add .
git commit -m "[YourName] Update Description"
git push origin main
```

### Ruilin Changes

#### Running Inference with Old Preprocessing

To perform inference using the old preprocessing method, use the following function call:

```python
Results = run_inference_on_images_with_old_preprocess(
    detector_path="path to training/config/detector/xception.yaml",
    weights_path="path to training/weights/xception_best.pth",
    image_paths=test_paths,
    cuda=True,
    manual_seed=42
)
```

#### **Results Structure**
`Results` is a **list of tuples**, where each tuple contains:

- **Path to the saved Grad-CAM output** (`.png` format)
- **Probability score**  
  **Example:**
  ```python
  tensor([0.0317], device='cuda:0', grad_fn=<SelectBackward0>)



#### **How to Change the Grad-CAM Save Location**

To modify the folder where Grad-CAM outputs are stored, follow these steps:

##### **1. Open the File**
Navigate to the following file: detectors/xception_detector.py

##### **2. Locate the Grad-CAM Function**
Inside `xception_detector.py`, find the function responsible for generating the Grad-CAM images.

##### **3. Modify the Save Path**
Change the variable `datasets_base_path` to the **desired folder path** where you want to store the Grad-CAM images.

```python
datasets_base_path = "/your/desired/folder/path"
```


### Chaimae Changes



### Mouad Changes


