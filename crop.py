import os
import cv2
import numpy as np
from PIL import Image

show_debug = False
margin = 64

def show_debug_image(image, title = 'debug'):
    if show_debug and image is not None:
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(title, 600, 800)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_image(image_path):
    try:
        pil_img = Image.open(image_path)
        if pil_img.mode == 'I;16':
            # Convert 16-bit to 8-bit
            array = np.array(pil_img).astype(np.float32)
            array = 255 * (array - array.min()) / (array.max() - array.min())
            array = array.astype(np.uint8)
        else:
            array = np.array(pil_img.convert('RGB'))
        
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def perspective_transform(image, pts):
    (tl, bl, tr, br) = pts
    
    # Calculate dimensions
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [0, max_height - 1],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1]
        ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
   
    return warped

def crop_page(image_path):
    image = load_image(image_path)
    if image is None:
        return None
    show_debug_image(image, 'Original')
    # Preprocessing
    cropped = image[margin:-margin, margin:-margin]
    show_debug_image(cropped, 'Cropped')

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    show_debug_image(gray, 'Grayscale')

    blurred = cv2.medianBlur(gray, 21)
    show_debug_image(blurred, 'Blurred')

    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    show_debug_image(normalized, 'Normalized')

    _,thr = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
    show_debug_image(thr, 'Threshold')
    
    padded = cv2.copyMakeBorder(thr, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    show_debug_image(padded, 'Padded')

    # Find contours
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    if len(contours) == 1:
        contour = contours[0]
    else:
        print("Page not found")
        return image

    min_rect = cv2.minAreaRect(contour)
    min_rect = (min_rect[0], (min_rect[1][0] + margin * 2, min_rect[1][1] + margin * 2), min_rect[2])
    page_points = cv2.boxPoints(min_rect)
    page_points = np.array(sorted(page_points, key=lambda x: (x[0], x[1])))

    return perspective_transform(image, page_points)

if __name__ == "__main__":
    current_dir = os.getcwd()
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(current_dir):
        if filename.endswith('.tif') and filename != 'output.tif':
            print(filename)
            output = crop_page(filename)
    
            if output is not None:
                show_debug_image(output, 'Output')
                cv2.imwrite("output\\" + filename, output)
