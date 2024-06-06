import cv2
import numpy as np
import matplotlib.pyplot as plt

source_image_path = '/home/nikolay/It_Jim/darts_game/IMG_20240510_172748.jpg'
image_without_background_path = '/home/nikolay/It_Jim/darts_game/masked_image.png'

def find_largest_contour(contours):
    # Функция cv2.arcLength() вычисляет длину контура, а cv2.approxPolyDP() аппроксимирует контур до замкнутого 
    # многоугольника с помощью дискретного метода Рамера-Дугласа-Пекера. Это позволяет упростить форму контура, 
    # сохраняя его общую структуру.
    contours_circul = []
    for contour in contours:
        Perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * Perimeter, True)
        if len(approx) >= 8:
            area = cv2.contourArea(contour)
            circularity = (4 * np.pi * area) / (Perimeter ** 2)
            
            if circularity > 0.85:
                contours_circul.append(contour)
    return contours_circul


def detect_radius(contours, center_x, center_y, ring_distance=20):
    detected_radii = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        detected_radii.append(int(radius))
    
    detected_radii = sorted(detected_radii)
    
    differences = np.diff(detected_radii)
    
    matching_radii = [detected_radii[i+1] for i, diff in enumerate(differences) if diff >= ring_distance]
    
    return matching_radii

def mask_space_out_of_darts(largest_contour, image, gray):
    mask = np.zeros_like(gray)
    mask_image = cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask_image)
    return masked_image

def pre_process_image(image_path):
    image = cv2.imread(image_path)
    if image is not None and len(image.shape) == 3:
        print(image.shape)
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        edges = cv2.Canny(blurred, 100, 200)
        
        return image, gray, blurred, edges
    else:
        return None, None, None, None
    
    
def find_colors(image, color_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_masks = {}
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        color_masks[color_name] = mask
    return color_masks

def define_colors(image):
    color_ranges = {
        'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
        'green': (np.array([40, 70, 50]), np.array([80, 255, 255])),
        'gray': (np.array([0, 0, 0]), np.array([179, 255, 100])),
    }
    
    color_masks = find_colors(image, color_ranges)
    
    for color_name, mask in color_masks.items():
        cv2.imwrite(f'{color_name}_mask.png', mask)

    # Combine the masks
    combined_mask = cv2.bitwise_or(color_masks['red'], color_masks['green'])
    
    # Apply the combined mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(combined_mask))
    cv2.imwrite('masked_image.png', masked_image)
    
    # Convert the masked image to grayscale
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the masked grayscale image
    contours_masked_image, _ = cv2.findContours(gray_masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours_masked_image, -1, (0, 255, 0), 3)
    
    # Save the contoured image
    cv2.imwrite('mask_contoured_image.png', contoured_image)
    
    return color_masks


def find_scores(image, cx, cy, max_radius):
    points = []
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    angle = angles[0]
    for i in range(max_radius):
        x = int(cx + i * np.cos(angle))
        y = int(cy + i * np.sin(angle))
        if cx <= x < cx + max_radius and cy <= y < cy + max_radius:
            points.append((image[x, y]))
    
    scores = []
    window_size = 8
    
    for i in range(0, len(points), window_size):
        window = points[i:i+window_size]
    
def all_colors_on_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    unique_colors = np.unique(image_rgb.reshape(-1, image_rgb.shape[2]), axis=0)
    
    print(f'Number Unique colors: {len(unique_colors)}')
    
    print('Unique colors:')
    for color in unique_colors:
        print(color)
        
def KNN(pixel, palette):
    distances = np.sqrt(np.sum(pixel - palette, axis=1))
    nearest_color = palette[np.argmin(distances)]
    return nearest_color


def colorizing_image_in_5_shades(image, palette):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    approximated_pixels = np.array([KNN(pixel, palette) for pixel in pixels])
    approximated_image = approximated_pixels.reshape(image_rgb.shape)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Approximated Image')
    plt.imshow(approximated_image)
    plt.axis('off')

    plt.show()
    
    

if __name__ == '__main__':
    image_source, gray_source, blurred_source, edges_source = pre_process_image(source_image_path)
    contours_source, _ = cv2.findContours(edges_source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours_source)
    largest_contour = max(largest_contour, key=cv2.contourArea)
    
    masked_image = mask_space_out_of_darts(largest_contour, image_source, gray_source)
    cv2.imwrite('masked_image.png', masked_image)
    print(f'Image without background saved as {image_without_background_path}')
    
    
    
    new_image, new_gray, new_blurred, new_edges = pre_process_image(image_without_background_path)
    # finding contours
    contours, _ = cv2.findContours(new_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length_of_contours = [len(contour) for contour in contours]
    print(sorted(length_of_contours))
    top_contours = []
    i = 0
    for contour in contours:
        if len(contour) > 500:
            i += 1
            top_contours.append(contour)
    print(f'Number of contours: {i}')
    new_contour_image = new_image.copy()
    cv2.drawContours(new_contour_image, top_contours, -1, (0, 255, 0), 3)
    
    masks_for_green_and_yelloq = define_colors(new_image)
    
    
    gray_mask = define_colors(new_image)['gray']
    contours_for_gray, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours_for_gray))
    sorted_gray_contours = sorted(contours_for_gray, key=cv2.contourArea, reverse=True)
    nine_main_contours = sorted_gray_contours[:9]
    print(len(nine_main_contour) for nine_main_contour in nine_main_contours)
    new_contour_image = new_image.copy()
    cv2.drawContours(new_contour_image, nine_main_contours, -1, (0, 255, 0), 3)
    cv2.imwrite('gray_contours.png', new_contour_image)
    
    
    
    
    # Save the result
    # cv2.imwrite('detected_rings.png', new_contour_image)
    # cv2.imwrite('edges.png', new_edges)
    # cv2.imwrite('blurred.png', new_blurred)
    # cv2.imwrite('gray.png', new_gray)
    
    
    # M = cv2.moments(largest_contour)
    # center_x = int(M['m10'] / M['m00'])
    # center_y = int(M['m01'] / M['m00'])
    
    # radius = detect_radius(circul_contours, center_x, center_y)
    # print(radius)
    
    # # Draw the detected circles
    # for r in radius:
    #     cv2.circle(image, (center_x, center_y), r, (0, 255, 0), 2)
    #     cv2.putText(image, str(r), (center_x + r, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the results using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Rings and Labels')
    # plt.show()

    # Save the result
    # cv2.imwrite('detected_rings.png', image)
    # cv2.imwrite('edges.png', edges)
    # cv2.imwrite('blurred.png', blurred)
    # cv2.imwrite('gray.png', gray)
    # cv2.imwrite('detected_circles.png', image)
        
                
        # print('Detecting circles...')
        # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=200, param2=100, minRadius=100, maxRadius=1000)

                
        # max_circles = 150
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     circles = circles[:max_circles]  # Select top N circles
            
        #     for (x, y, r) in circles:
        #         cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        
        # Display the results using matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(edges, cmap='gray')
    # plt.title('Canny Edge Detection')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Circles')
    
    # plt.show()
    
    # # Save the results
    # cv2.imwrite('edges.png', edges)
    # cv2.imwrite('blurred.png', blurred)
    # cv2.imwrite('gray.png', gray)
    # cv2.imwrite('detected_circles.png', image)
    # else:
    #     print("Error: Image not loaded correctly.")
