import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Pylette import extract_colors
from PIL import Image
from scipy import stats
from  scipy.ndimage import center_of_mass

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
    corresponding_contours_radii = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        corresponding_contours_radii.append((distance_to_center, int(radius), contour))
    
    # Sort based on the distance to the center
    detected_radii = sorted(corresponding_contours_radii, key=lambda x: x[0])
    
    # Extract radii and contours after sorting
    sorted_radii = [radius for _, radius, _ in detected_radii]
    sorted_contours = [contour for _, _, contour in detected_radii]
    
    # Calculate differences between successive radii
    differences = np.diff(sorted_radii)
    
    # Find matching radii and corresponding contours based on ring distance
    matching_radii = []
    matching_contours = []
    
    for i, diff in enumerate(differences):
        if diff >= ring_distance:
            matching_radii.append(sorted_radii[i+1])
            matching_contours.append(sorted_contours[i+1])
    
    return matching_radii, matching_contours

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

def define_colors(image, kernel_size, iterations):
    rgb_colors = [
        [148, 135, 77],
        [103, 80, 56]
    ]
    
    target_rgb = np.uint8([[[87, 77, 56]]])
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0]
    
    hsv_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0] for color in rgb_colors]
    
    color_ranges = {
            'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
            'green': (np.array([40, 70, 50]), np.array([80, 255, 255])),
            'brown': (np.array([10, 100, 50]), np.array([20, 200, 150])),
            'target': (np.array([max(0, target_hsv[0] - 10), max(0, target_hsv[1] - 40), max(0, target_hsv[2] - 40)]), np.array([min(179, target_hsv[0] + 10), min(255, target_hsv[1] + 40), min(255, target_hsv[2] + 40)]))
        }
        
    color_masks = find_colors(image, color_ranges)
    
    # Display the masks
    print('Cleaning golden mask')
    for color_name, mask in color_masks.items():
        if color_name == 'brown' or color_name == 'target':
            mask = mask_denoiser(mask, iterations, kernel_size)
        cv2.imwrite(f'{color_name}_mask.png', mask)
        
    return color_masks

        
def KNN(pixel, palette):
    distances = np.sqrt(np.sum((pixel - palette)**2, axis=1))
    nearest_color = palette[np.argmin(distances)]
    return nearest_color


def colorizing_image_in_10_shades(image, palette):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    approximated_pixels = np.array([KNN(pixel, palette) for pixel in pixels], dtype = 'uint8')
    approximated_image = approximated_pixels.reshape(image_rgb.shape)
    
    gold_shades = palette[-1]
    mask = np.all(approximated_image == gold_shades, axis = -1)
    
    gold_mask_image = np.zeros_like(image_rgb, dtype = 'uint8')
    gold_mask_image[mask] = gold_shades
    
    gold_mask_image = cv2.cvtColor(gold_mask_image, cv2.COLOR_RGB2BGR)
    approximated_image = cv2.cvtColor(approximated_image, cv2.COLOR_RGB2BGR)

    
    cv2.imwrite('approximated_image.png', approximated_image)
    cv2.imwrite('masked_image_gold.png', gold_mask_image)
    
    return approximated_image, gold_mask_image, mask


# def find_centers_of_clusters(new_image, target_mask, n_clusters=6, threshold = 0.8):
#     delta = 0.01
#     gold_pixels_coordinates = np.argwhere(target_mask)
#     print(gold_pixels_coordinates)
    
#     block_size = 8
#     block_size_for_centers = 4
    
#     block = np.zeros((block_size, block_size), dtype='uint8')
#     print("Block Shape:", block.shape[0])
    
#     block_for_centers = np.zeros((block_size_for_centers, block_size_for_centers, 3), dtype='uint8')
    
#     centers = []
#     if len(gold_pixels_coordinates) > 0:
#         for i in range(0, target_mask.shape[0], block_size):
#             for j in range(0, target_mask.shape[1], block_size):
#                 block_mask = target_mask[i:i+block_size, j:j+block_size]
#                 total_pixels = block_mask.size
#                 white_pixels = np.sum(block_mask > 0)
#                 if white_pixels / total_pixels >= threshold:
#                     center = center_of_mass(block_mask)
#                     centers.append(center)
#                     cv2.circle(new_image, (int(center[1]) + j, int(center[0]) + i), 5, (0, 255, 0), -1)
#     cv2.imwrite('image_with_centroids.png', new_image)
#     return gold_pixels_coordinates


def find_centers_of_clusters(new_image, target_mask, threshold=0.8, delta=50):
    gold_pixels_coordinates = np.argwhere(target_mask)
    print(gold_pixels_coordinates)
    
    block_size = 8
    block_size_for_centers = 4
    
    block = np.zeros((block_size, block_size), dtype='uint8')
    print("Block Shape:", block.shape[0])
    
    block_for_centers = np.zeros((block_size_for_centers, block_size_for_centers, 3), dtype='uint8')
    
    centers = []
    
    if len(gold_pixels_coordinates) > 0:
        for i in range(0, target_mask.shape[0], block_size):
            for j in range(0, target_mask.shape[1], block_size):
                block_mask = target_mask[i:i+block_size, j:j+block_size]
                total_pixels = block_mask.size
                white_pixels = np.sum(block_mask > 0)
                if white_pixels / total_pixels >= threshold:
                    center = center_of_mass(block_mask)
                    global_center = (center[0] + i, center[1] + j)
                    if not any(np.linalg.norm(np.array(global_center) - np.array(existing_center)) < delta for existing_center in centers):
                        centers.append(global_center)
                        cv2.circle(new_image, (int(global_center[1]), int(global_center[0])), 5, (0, 255, 0), -1)
                
    cv2.imwrite('image_with_centroids.png', new_image)
    return centers
    
def mask_denoiser(golden_mask, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(golden_mask, kernel=kernel, iterations=iterations)
    cleaned_mask = cv2.dilate(eroded_mask, kernel=kernel, iterations=iterations * 2)
    
    cv2.imwrite('cleaned_mask.png', cleaned_mask)
    return cleaned_mask
    
        

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
    
    # cleaned_bgr = replace_outliers_with_mode(result, count_of_neighbors = 5, threshold = 100)
    
    return result


def find_mode(neighbors):
    if len(neighbors) == 0:
        return None
    neighbors_array = np.array(neighbors)
    mode_result = stats.mode(neighbors_array, axis=0)
    mode_color = mode_result.mode[0]
    count = mode_result.count[0]
    if np.all(count > 1):
        return mode_color.astype(int)
    return None


def replace_outliers_with_mode(img, count_of_neighbors=8, threshold=200):
    output_img = img.copy()
    rows, cols = img.shape[:2]
    # Padding the image to avoid boundary issues
    padded_img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            current_pixel = padded_img[y, x][:3]
            neighbors = padded_img[y-1:y+2, x-1:x+2, :3].reshape(-1, 3)
            neighbors = np.delete(neighbors, 4, axis=0)  # Remove the center pixel
            diff = np.mean(np.abs(neighbors - current_pixel), axis=1)
            flag = np.sum(diff > threshold)
            if flag >= count_of_neighbors:
                mode_color = find_mode(neighbors)
                if mode_color is not None:
                    output_img[y-1, x-1][:3] = mode_color  # Adjust indices due to padding
    
    return output_img

def find_target_center(new_image, largest_contour):
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"] + 40)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if 0.7 < circularity < 1.3:
                # Draw the contour and center of the circle
                cv2.drawContours(new_image, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(new_image, (cX, cY), 5, (255, 0, 0), -1)
                cv2.putText(new_image, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the result image
                cv2.imwrite('image_with_center.png', new_image)
                
                return cX, cY
            
            
def detect_circles(image_path):
    # Загрузить изображение
    image = cv2.imread(image_path)
    output = image.copy()
    
    # Преобразовать изображение в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применить размытие для снижения шума
    gray_blurred = cv2.medianBlur(gray, 5)
    
    # Найти круги с использованием алгоритма Хафа
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
        param1=50, param2=30, minRadius=0, maxRadius=0
    )
    
    # Если круги найдены, обработать их
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Нарисовать внешний круг
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Нарисовать центр круга
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    # Сохранить результат
    cv2.imwrite('detected_circles.png', output)
    
    
def calculate_distance(cx, cy, radius): 


if __name__ == '__main__':
    image_source, gray_source, blurred_source, edges_source = pre_process_image(source_image_path)
    contours_source, _ = cv2.findContours(edges_source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours_source)
    largest_contour = max(largest_contour, key=cv2.contourArea)
    
    masked_image = mask_space_out_of_darts(largest_contour, image_source, gray_source)
    cv2.imwrite(image_without_background_path, masked_image)
    print(f'Image without background saved as {image_without_background_path}')

    new_image, new_gray, new_blurred, new_edges = pre_process_image(image_without_background_path)
    # find center of board
    cx, cy = find_target_center(new_image, largest_contour)

    # finding contours
    contours, _ = cv2.findContours(new_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # detect_circles(image_without_background_path)
   
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
    print(f'number of top contours: {len(top_contours)}')
    cv2.drawContours(new_contour_image, top_contours, -1, (0, 255, 0), 3)
    cv2.imwrite('select_contour_image.png', new_contour_image)
    
    # Extracting colors
    # TODO Denoiser
    # filtered_image = denoiser(new_image)
    
    # Create masks for each color
    color_masks = define_colors(new_image, kernel_size = 5, iterations = 1)
    print('Masks ready !!!')
    
    for color_name, mask in color_masks.items():
        if color_name == 'target':
            target_mask = mask
        if color_name == 'red':
            red_mask = mask
        if color_name == 'green':
            green_mask = mask
    
    palette = extract_colors(image=image_without_background_path, palette_size=10, resize = True)
    palette.display(save_to_file=False)
    main_colors = [color.rgb for color in palette]
    
    # main_colors = [[*color] for color in main_colors]
    # print(main_colors)
    # approximated_image, gold_mask_image, mask = colorizing_image_in_10_shades(new_image, main_colors)
    
    centroids = find_centers_of_clusters(new_image, target_mask, threshold=0.7)
    print(f'Centroids: {centroids}')
    
    
    # # Save the result
    # cv2.imwrite('detected_rings.png', new_contour_image)
    # cv2.imwrite('edges.png', new_edges)
    # cv2.imwrite('blurred.png', new_blurred)
    # cv2.imwrite('gray.png', new_gray)
    
    # ------------------------------------------------------------------------
    # M = cv2.moments(largest_contour)
    # center_x = int(M['m10'] / M['m00'])
    # center_y = int(M['m01'] / M['m00'])
    
    matching_radii, matching_contours = detect_radius(contours, cx, cy, ring_distance=20)
    print(f'Matching radii: {matching_radii}, Matching contours: {len(matching_contours)}')

    sorted_radius_contours = []
    for r, contour in zip(matching_radii, matching_contours):
        if r < 50 or any(abs(r - radius) < 50 for radius, _ in sorted_radius_contours):
            continue
        sorted_radius_contours.append((r, contour))

    sorted_radius_contours = sorted(sorted_radius_contours, key=lambda x: x[0])

    if sorted_radius_contours:
        for radius, contour in sorted_radius_contours:
            print(f'Radius: {radius}')
            print(f'Corresponding contour length: {len(contour)}')
    else:
        print('No valid radii found')
        
        
    



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
