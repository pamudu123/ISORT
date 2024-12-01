import cv2
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from waste_classifier import WasteClassifier

class WastePredictor:
    def __init__(self, model_path, num_classes, is_torchscript=False, fp16=False, device=None):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the saved model checkpoint or TorchScript model
            num_classes (int): Number of classes in the model
            is_torchscript (bool): Whether the model is a TorchScript model
            fp16 (bool): Whether the model uses FP16 precision
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.is_torchscript = is_torchscript
        self.fp16 = fp16
        
        # Load the model
        if is_torchscript:
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            self.model = WasteClassifier(num_classes=num_classes)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device and configure precision
        self.model = self.model.to(self.device)
        if fp16:
            self.model = self.model.half()  # Convert weights to FP16 if applicable
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: np.ndarray):
        """ Preprocess a single image using OpenCV """        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to match the model's input size
        image = cv2.resize(image, (224, 224))
        
        # Apply transformations (convert to Tensor and normalize)
        image = self.transform(image)
        
        # Convert to FP16 if required
        if self.fp16:
            image = image.half()
        
        # Add batch dimension
        return image.unsqueeze(0)
    
    def predict_single(self, image, return_confidence=False):
        """ Make prediction for a single image """
        # Preprocess image
        image = self.preprocess_image(image)
        image = image.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        if return_confidence:
            return predicted.item(), confidence.item()
        return predicted.item()
    
    def predict_batch(self, batch_np_images, batch_size=32):
        """ Make predictions for a batch of images using OpenCV """
        predictions = []
        confidences = []
        
        # Process images in batches
        for i in tqdm(range(0, len(batch_np_images), batch_size)):
            batch_np_images = batch_np_images[i:i + batch_size]
            
            # Preprocess batch images
            batch_torch_images = [self.preprocess_image(img).squeeze(0) for img in batch_np_images]
            batch_torch_images = torch.stack(batch_torch_images).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_torch_images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
        
        return np.array(predictions), np.array(confidences)