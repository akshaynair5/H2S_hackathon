import io
from google.cloud import vision

def detect_fake_image(image_path):


    client = vision.ImageAnnotatorClient.from_service_account_file('key.json')

    #image file read
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    #web detection
    response = client.web_detection(image=image)
    annotations = response.web_detection

    results = {
        'full_matching_images': [],
        'visually_similar_images': [],
        'pages_with_matching_images': []
    }

    if annotations.full_matching_images:
        for image in annotations.full_matching_images:
            results['full_matching_images'].append(image.url)

    if annotations.visually_similar_images:
        for image in annotations.visually_similar_images:
            results['visually_similar_images'].append(image.url)

    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            results['pages_with_matching_images'].append({
                'url': page.url,
                'title': page.page_title
            })

    return results



image_to_check = 'crash.jpeg'
detection_results = detect_fake_image(image_to_check)

print("---- Fake Image Detection Results ----")

if detection_results['full_matching_images']:
    print("\n Found exact matches!")
    print("These images have been used elsewhere on the web.")
    for url in detection_results['full_matching_images']:
        print(f"  - {url}")

if detection_results['pages_with_matching_images']:
    print("\n Found matching pages!")
    print("These pages contain your image. Check the context and dates.")
    for page in detection_results['pages_with_matching_images']:
        print(f"  - Title: '{page['title']}'")
        print(f"    URL: {page['url']}")

if detection_results['visually_similar_images']:
    print("\n Found visually similar images!")
    print("These might be cropped or edited versions.")
    for url in detection_results['visually_similar_images']:
        print(f"  - {url}")

if not any(detection_results.values()):
    print(" No matching or similar images found on the web.")
    