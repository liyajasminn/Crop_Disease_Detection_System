from tensorflow.keras.models import load_model

try:
    model = load_model(r"C:\Users\ASUS\OneDrive\Desktop\Crop_Disease_Detection_System\mobilenet_final.keras")
    print("Model loaded successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()