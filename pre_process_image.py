import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Pylette import extract_colors
from PIL import Image

source_image_path = '/home/nikolay/ML/It_Jim/Darts_game/images/IMG_20240510_172748.jpg'
image_without_background_path = '/home/nikolay/ML/It_Jim/Darts_game/masked_image.png'

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
            'blue': (np.array([90, 70, 50]), np.array([130, 255, 255])),
            'gold': (np.array([26, 100, 50]), np.array([30, 255, 200])),
            'gold_2': (np.array([32, 100, 100]), np.array([36, 255, 255]))
        }
        
    color_masks = find_colors(image, color_ranges)
    
    # Display the masks
    for color_name, mask in color_masks.items():
        cv2.imwrite(f'{color_name}_mask.png', mask)
        
def KNN(pixel, palette):
    distances = np.sqrt(np.sum((pixel - palette)**2, axis=1))
    nearest_color = palette[np.argmin(distances)]
    return nearest_color


def colorizing_image_in_5_shades(image, palette):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    approximated_pixels = np.array([KNN(pixel, palette) for pixel in pixels], dtype = 'uint8')
    approximated_image = approximated_pixels.reshape(image_rgb.shape)
    
    gold_shades = palette[7]
    mask = np.all(approximated_image == gold_shades, axis = -1)
    
    gold_mask_image = np.zeros_like(image_rgb, dtype = 'uint8')
    gold_mask_image[mask] = gold_shades
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Approximated Image')
    plt.imshow(approximated_image)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Gold Masked Image')
    plt.imshow(gold_mask_image)
    plt.axis('off')

    plt.show()
    
    gold_mask_image = cv2.cvtColor(gold_mask_image, cv2.COLOR_RGB2BGR)
    approximated_image = cv2.cvtColor(approximated_image, cv2.COLOR_RGB2BGR)

    
    cv2.imwrite('approximated_image.png', approximated_image)
    cv2.imwrite('masked_image_gold.png', gold_mask_image)
    
    return approximated_image, gold_mask_image, mask


def find_center_mass(approximated_image, mask, n_clusters = 6):
    gold_pixels_coordinates = np.argwhere(mask)
    
    if len(gold_pixels_coordinates) > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(gold_pixels_coordinates)
        centroids = kmeans.cluster_centers_
        
        for centroid in centroids:
            cv2.circle(approximated_image, (int(centroid[1]), int(centroid[0])), 10, (0, 0, 255), -1)
        cv2.imwrite('image_with_centroids.png', approximated_image)
        
        return centroids
        

def denoiser(image):
    # Разделение цветовых каналов
    b, g, r = cv2.split(image)

    # Применение медианного фильтра к каждому каналу
    b_median = cv2.medianBlur(b, 5)
    g_median = cv2.medianBlur(g, 5)
    r_median = cv2.medianBlur(r, 5)

    # Применение морфологических операций к каждому каналу
    kernel = np.ones((5,5), np.uint8)
    b_morph = cv2.morphologyEx(b_median, cv2.MORPH_CLOSE, kernel)
    g_morph = cv2.morphologyEx(g_median, cv2.MORPH_CLOSE, kernel)
    r_morph = cv2.morphologyEx(r_median, cv2.MORPH_CLOSE, kernel)

    # Объединение каналов обратно
    result = cv2.merge((b_morph, g_morph, r_morph))

    # Сохранение результата в файл
    output_path = 'denoised_image.png'
    cv2.imwrite(output_path, result)
    
    return result

if __name__ == '__main__':
    image_source, gray_source, blurred_source, edges_source = pre_process_image(source_image_path)
    contours_source, _ = cv2.findContours(edges_source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours_source)
    largest_contour = max(largest_contour, key=cv2.contourArea)
    
    masked_image = mask_space_out_of_darts(largest_contour, image_source, gray_source)
    cv2.imwrite(image_without_background_path, masked_image)
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
    
    # Extracting colors
    # TODO Denoiser
    
    filtered_image  = denoiser(new_image)
    filtered_image_pil = Image.fromarray(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    palette = extract_colors(image=filtered_image_pil, palette_size=10, resize = True)
    palette.display(save_to_file=False)
    main_colors = [color.rgb for color in palette]
    
    # Create masks for each color
    define_colors(new_image)
    
    main_colors = [[*color] for color in main_colors]
    print(main_colors)
    approximated_image, gold_mask_image, mask = colorizing_image_in_5_shades(new_image, main_colors)
    
    centroids = find_center_mass(approximated_image, mask)
    print(f'Centroids: {centroids}')
    
    
    # Save the result
    cv2.imwrite('detected_rings.png', new_contour_image)
    cv2.imwrite('edges.png', new_edges)
    cv2.imwrite('blurred.png', new_blurred)
    cv2.imwrite('gray.png', new_gray)
    
    
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
