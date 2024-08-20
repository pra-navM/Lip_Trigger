import cv2
import numpy as np
import time
import argparse
import RPi.GPIO as GPIO
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Replace with the appropriate GPIO pin

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Live Mouth Detection with GPIO Pulse")
parser.add_argument('--pulse_length', type=float, default=0.1, help='Length of GPIO pulse in seconds')
parser.add_argument('--delay_length', type=float, default=0, help='Delay before next GPIO pulse when key is pressed (seconds)')
args = parser.parse_args()

# Load the TFLite model and allocate tensors
interpreter = make_interpreter('ENEW_mouth_state_model_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Get input tensor details
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

# Initialize camera
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

# Initialize variables to track the state and delay
previous_label = None
apply_delay = False

while True:
    start_time = time.time()
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    cam_time = time.time() - start_time
    
    # Preprocess the image to match model input requirements
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    # Run the model
    inference_start = time.time()
    common.set_input(interpreter, img)
    interpreter.invoke()
    result = classify.get_classes(interpreter, top_k=1)
    inference_time = time.time() - inference_start

    # Get the result
    label = result[0].id  # The label (0 for closed, 1 for open)
    score = result[0].score  # Confidence score

    # Check for state change from closed to open
    if previous_label == 0 and label == 1:
        if apply_delay:
            print(f"Applying delay of {args.delay_length} seconds before GPIO pulse")
            time.sleep(args.delay_length)
            apply_delay = False
        
        GPIO.output(18, GPIO.HIGH)
        time.sleep(args.pulse_length)
        GPIO.output(18, GPIO.LOW)

    previous_label = label

    # Calculate and print delays
    total_time = time.time() - start_time
    print(f'Camera Delay: {cam_time:.3f} s, Inference Delay: {inference_time:.3f} s, Total Delay: {total_time:.3f} s')

    # Display the result on the frame
    text = "Mouth Open" if label == 1 else "Mouth Closed"
    cv2.putText(frame, f'{text} ({score:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Live Mouth Detection', frame)

    # Check for key press to apply delay
    if cv2.waitKey(1) & 0xFF == ord('d'):
        apply_delay = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
