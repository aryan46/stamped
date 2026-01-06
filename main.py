from ultralytics import YOLO

# Load your PyTorch model
print("Loading yolov8n.pt model...")
model = YOLO('yolov8n.pt')

# Export the model to ONNX format
# This will create a new file: 'yolov8n.onnx'
print("Exporting to ONNX format...")
model.export(format='onnx')

print("\n✅ Successfully exported model to 'yolov8n.onnx'")