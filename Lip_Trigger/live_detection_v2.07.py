import random
import threading
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import argparse

# ------------------- GPIO Setup -------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

# ------------------- Command-line Arguments -------------------
parser = argparse.ArgumentParser(description="Live Mouth Detection with Delay Ramp, Initial Trials & Keypress Adjustment")
parser.add_argument('--pulse_length', type=float, default=0.13, help='Length of GPIO pulse in seconds')
parser.add_argument('--target_delay', type=float, default=0.060, help='Target total delay (seconds) including natural system delay (after initial trials)')
parser.add_argument('--ramp_step', type=float, default=0.005, help='Step size (seconds) to increase artificial delay per detection during ramping')
parser.add_argument('--keypress_adjust', type=float, default=0.030, help='Extra delay (seconds) to add/subtract after keypress for next event')
parser.add_argument('--initial_trials', type=int, default=3, help='Number of initial mouth-open events to use initial_delay before ramping')
parser.add_argument('--initial_delay', type=float, default=0.030, help='Initial total delay (seconds) including natural system delay for the initial trials')
args = parser.parse_args()

# ------------------- Load Model -------------------
interpreter = make_interpreter('ENEW_mouth_state_model_quant_edgetpu.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

# ------------------- Camera Setup -------------------
cap = cv2.VideoCapture(0)  # Change index if multiple cameras

# ------------------- State Tracking -------------------
last_state = None

# artificial_delay is the extra delay (on top of measured natural_delay) that we actually wait before triggering the GPIO
artificial_delay = 0.0

# initial-phase bookkeeping
initial_trials_remaining = max(0, args.initial_trials)
initial_phase = initial_trials_remaining > 0
initial_artificial = None  # computed once (relative to natural delay) on the first initial mouth-open event

# keypress one-time adjustment flag
keypress_pending = False

# lock for safety (not strictly necessary here, but safe if you later modify from threads)
delay_lock = threading.Lock()

# ------------------- GPIO Trigger Function -------------------
def trigger_gpio(delay):
    """delay: artificial delay in seconds (only the artificial portion)."""
    # artificial delay only (natural delay is already incurred by main loop)
    if delay > 0:
        time.sleep(delay)
    GPIO.output(18, GPIO.HIGH)
    time.sleep(args.pulse_length)
    GPIO.output(18, GPIO.LOW)

# ------------------- Main loop -------------------
print("Starting live detection. Press 'k' to apply one-time keypress adjustment to next event, 'q' to quit.")
while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image for model
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    # Run inference
    common.set_input(interpreter, img)
    interpreter.invoke()
    result = classify.get_classes(interpreter, top_k=1)

    label = result[0].id
    score = result[0].score

    # Natural delay = time from frame capture (loop_start) to decision time
    natural_delay = time.time() - loop_start  # seconds

    # Determine the total desired delay for this event depending on phase and keypress
    if initial_phase:
        base_total_delay = args.initial_delay
    else:
        base_total_delay = args.target_delay

    # If a keypress adjustment is pending, apply it to the total for this next event
    if keypress_pending:
        adjusted_total_delay_for_event = base_total_delay + args.keypress_adjust
    else:
        adjusted_total_delay_for_event = base_total_delay

    # artificial target for this event (non-negative)
    artificial_delay_target_for_event = max(0.0, adjusted_total_delay_for_event - natural_delay)

    # Handle mouth-open detection (state change)
    if last_state is not None and label != last_state:
        if label == 1:  # Mouth opened
            # 1. Determine the baseline artificial delay for this event
            if initial_phase:
                # In initial phase, we use the fixed initial_artificial calculated on first run
                if initial_artificial is None:
                    initial_artificial = max(0.0, args.initial_delay - natural_delay)
                    with delay_lock:
                        artificial_delay = initial_artificial
                current_baseline = artificial_delay
            else:
                # In ramping phase, update the global baseline toward the target
                # Note: target_delay is used here, NOT the keypress-adjusted one
                target_art = max(0.0, args.target_delay - natural_delay)
                with delay_lock:
                    if artificial_delay < target_art:
                        artificial_delay = min(artificial_delay + args.ramp_step, target_art)
                    else:
                        artificial_delay = target_art
                    current_baseline = artificial_delay

            # 2. Apply the one-time keypress adjustment (Randomly + or -)
            trigger_delay = current_baseline
            if keypress_pending:
                # Randomly choose between +1 and -1
                direction = random.choice([1, -1])
                actual_adjustment = args.keypress_adjust * direction
                
                trigger_delay += actual_adjustment
                
                # Ensure we don't try to sleep for a negative amount of time
                trigger_delay = max(0.0, trigger_delay)
                
                sign_str = "+" if direction > 0 else "-"
                print(f"[Keypress Applied] Baseline: {current_baseline*1000:.1f}ms {sign_str} {args.keypress_adjust*1000:.1f}ms = Total Art: {trigger_delay*1000:.1f}ms")
                
                keypress_pending = False # Reset flag
            else:
                print(f"[Event] Natural: {natural_delay*1000:.1f}ms, Artificial: {trigger_delay*1000:.1f}ms")

            # 3. Trigger the GPIO in a background thread
            threading.Thread(target=trigger_gpio, args=(max(0.0, trigger_delay),), daemon=True).start()

            # 4. Handle trial counter for initial phase
            if initial_phase:
                initial_trials_remaining -= 1
                if initial_trials_remaining <= 0:
                    initial_phase = False
                    print("[System] Initial trials complete. Entering ramping phase.")
    last_state = label

    # ------------------- Display -------------------
    text = "Mouth Open" if label == 1 else "Mouth Closed"
    cv2.putText(frame, f'{text} ({score:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f'Natural Delay: {natural_delay*1000:.1f} ms', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    with delay_lock:
        cv2.putText(frame, f'Artificial Delay (current): {artificial_delay*1000:.1f} ms', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    if initial_phase:
        cv2.putText(frame, f'Initial phase: {initial_trials_remaining} trials left', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1)
        cv2.putText(frame, f'Initial total desired: {args.initial_delay*1000:.1f} ms', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1)
    else:
        cv2.putText(frame, f'Target Total: {args.target_delay*1000:.1f} ms', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1)

    if keypress_pending:
        cv2.putText(frame, f'Pending Adjustment: {args.keypress_adjust*1000:.1f} ms (applies next event)', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow('Live Mouth Detection', frame)

    # ------------------- Key Handling -------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('k'):
        keypress_pending = True
        print(f"[key] keypress pending: next event will use total delay adjustment {args.keypress_adjust*1000:.1f} ms")

# ------------------- Cleanup -------------------
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
