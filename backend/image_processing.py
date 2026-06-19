import cv2
import os

def enhance_image(input_path, output_path):

    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return False

    enhanced = cv2.equalizeHist(image)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        enhanced
    )

    return True