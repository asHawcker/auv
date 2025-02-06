from depthai_sdk import OakCamera
import cv2

with OakCamera() as oak:
    # Configure camera (lower resolution for stability)
    cam = oak.create_camera('color', resolution='1080p', fps=15)
    
    # Load YOLOv5s model
    nn = oak.create_nn('./result/best.json', cam, nn_type='yolo', spatial=True)
    
    # Initialize OpenCV window
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    
    def detect_gate(packet):
        frame = packet.frame  # Get RGB frame
        
        # Process detections
        for detection in packet.detections:
            # Raw detection data
            bbox = detection.bbox
            confidence = detection.confidence
            x_mm = detection.spatialCoordinates.x
            y_mm = detection.spatialCoordinates.y
            z_mm = detection.spatialCoordinates.z
            
            # Print coordinates (for your AUV's control system)
            print(f"Gate @ X:{x_mm}mm Y:{y_mm}mm Z:{z_mm}mm | Confidence: {confidence:.2f}")
            
            # Simple visualization (optional)
            h, w = frame.shape[:2]
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int(bbox.xmax * w)
            y2 = int(bbox.ymax * h)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Preview", frame)
        cv2.waitKey(1)
    
    # Attach callback
    oak.callback(nn.out.main, detect_gate)
    
    # Start pipeline
    oak.start(blocking=True)
