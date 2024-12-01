import cv2
import time
import os
import numpy as np
from waste_predictor import WastePredictor

CLASS_NAMES = ['BEER_CANS', 'OTHER', 'PET_BOTTLES', 'SHAMPOO_BOTTLES', 'YOGHURT_CUPS']
TORCHSCRIPT_MODEL_PATH = 'model_scripted.pt'

predictor = WastePredictor(
    model_path = TORCHSCRIPT_MODEL_PATH,
    num_classes = len(CLASS_NAMES),                    
    is_torchscript = True,
    fp16 = True
)

image_dir = r'datasets/images'
cv_images = []
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    cv_image = cv2.imread(image_path)
    cv_images.append(cv_image)

t_start = time.time()
pred_ids, confidences = predictor.predict_batch(cv_images)
t_end = time.time()

print(f'Inference Time per Image: {((t_end - t_start)/len(cv_images)) * 1000:.2f} ms')
print(f'Prediction Labels: {pred_ids}')
print(f'Prediction Names: {[CLASS_NAMES[id] for id in pred_ids]}')
print(f'Confidence Scores: {confidences}')

# Show Results
for i, cv_image in enumerate(cv_images):
    label = CLASS_NAMES[pred_ids[i]]
    confidence = confidences[i]

    # Resize image for display
    cv_image = cv2.resize(cv_image, (640, 480)) 

    # Create a black bar at the bottom
    bar_height = 50
    black_bar = np.zeros((bar_height, cv_image.shape[1], 3), dtype=np.uint8)

    # Add text to the black bar
    text = f'{label} ({confidence:.2%})'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (black_bar.shape[1] - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2

    cv2.putText(black_bar, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    combined_image = np.vstack((cv_image, black_bar))

    cv2.imshow('Predictions', combined_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()