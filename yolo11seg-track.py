import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
import io

import time

def send_email(receiver_email, frame, max_retries=3, delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            # Set up the SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)  # Update if using a different service
            server.starttls()  # Enable security
            server.login('freedomtech85@gmail.com', 'ozwt mqfz zxja ebwh')  # Update with your credentials(sender_mail,password) 

            # Create the email
            msg = MIMEMultipart()
            msg['From'] = 'freedomtech85@gmail.com'  # Update with your email
            msg['To'] = receiver_email
            msg['Subject'] = 'Fire Detected'

            # Encode the frame as an image
            _, buffer = cv2.imencode('.jpg', frame)
            img_data = buffer.tobytes()  # Convert the buffer to bytes

            # Attach the image with the name "noname.jpg"
            img = MIMEImage(img_data, name="Fire.jpg")
            img.add_header('Content-Disposition', 'attachment', filename="noname.jpg")
            msg.attach(img)

            # Send the email
            server.send_message(msg)
            print("Email sent successfully.")
            break  # Exit the loop if the email is sent successfully

        except smtplib.SMTPException as e:
            attempt += 1
            print(f"Failed to send email: {e}. Retrying {attempt}/{max_retries}...")
            time.sleep(delay)  # Wait before retrying

        finally:
            try:
                server.quit()  # Terminate the SMTP session
            except:
                pass  # If server was not initialized, just pass

    if attempt == max_retries:
        print("Max retries reached. Could not send email.")


# Load the YOLOv8 model
model = YOLO("best.pt")
names = model.model.names

# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('vid.mp4')
count = 0
frame_tot = 0

email_threads = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Ensure boxes exist in the results
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Check if tracking IDs exist before attempting to retrieve them
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)  # Use -1 for objects without IDs

        masks = results[0].masks
        if masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = masks.xy
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                c = names[class_id]
                x1, y1, x2, y2 = box

                # Check if mask is not empty
                if mask.size > 0:
                    mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))  # Reshape mask to correct format

                    # Draw the bounding box and mask
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.fillPoly(overlay, [mask], color=(0, 0, 255))

                    # Draw the track ID and class label
                    cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                    if 'fire' in c:
                        # Create a thread to send an email with the current frame
                        receiver_email = "truckersfan66@gmail.com"  # Update with the actual receiver email
                        email_thread = threading.Thread(target=send_email, args=(receiver_email, frame.copy()))  # Use frame.copy() to avoid threading issues
                        email_threads.append(email_thread)  # Track the thread
                        email_thread.start()  # Start the email thread

            alpha = 0.5  # Transparency factor (0 = invisible, 1 = fully visible)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show the frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_tot += 1  # Increment frame counter
# Wait for all email threads to finish after the loop
for thread in email_threads:
    thread.join()  # Ensure all emails are sent before closing

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
