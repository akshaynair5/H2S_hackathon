import cv2
import os
import io
from google.cloud import vision

def detect_fake_video(video_path, frame_interval=30):
    client = vision.ImageAnnotatorClient.from_service_account_file('key.json')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detection_summary = {
        'full_matching_images': set(),
        'visually_similar_images': set(),
        'pages_with_matching_images': set()
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:  # Sample every Nth frame
            _, buffer = cv2.imencode('.jpg', frame)
            content = buffer.tobytes()

            frame_results = detect_fake_image_from_content(content, client)

            detection_summary['full_matching_images'].update(frame_results['full_matching_images'])
            detection_summary['visually_similar_images'].update(frame_results['visually_similar_images'])
            for page in frame_results['pages_with_matching_images']:
                detection_summary['pages_with_matching_images'].add((page['title'], page['url']))

        frame_count += 1

    cap.release()

    return detection_summary


if __name__ == "__main__":
    video_to_check = "  "       #File
    detection_results = detect_fake_video(video_to_check, frame_interval=60)

    print("Result:")

    if detection_results['full_matching_images']:
        print("\n Found exact matches (from video frames)!")
        for url in detection_results['full_matching_images']:
            print(f"  - {url}")

    if detection_results['pages_with_matching_images']:
        print("\n Found matching pages!")
        for title, url in detection_results['pages_with_matching_images']:
            print(f"  - Title: '{title}'")
            print(f"    URL: {url}")

    if detection_results['visually_similar_images']:
        print("\n Found visually similar images!")
        for url in detection_results['visually_similar_images']:
            print(f"  - {url}")

    if not any(detection_results.values()):
        print(" No matches found for sampled video frames.")
