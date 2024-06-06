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

def is_circle(contour, tolerance=0.1):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return abs(circularity - 1) >= tolerance

# def detect_radius(contours, center_x, center_y, ring_distance=1):
#     tolerance=0.1
#     corresponding_contours_radii = []
#     circle_contours = []
#     # for contour in contours:
#     #     if is_circle(contour, tolerance):
#     #         circle_contours.append(contour)
    
#     for contour in contours:
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
#         corresponding_contours_radii.append((distance_to_center, int(radius), contour))
    
#     # Sort based on the distance to the center
#     detected_radii = sorted(corresponding_contours_radii, key=lambda x: x[0])
    
#     # Extract radii and contours after sorting
#     sorted_radii = [radius for _, radius, _ in detected_radii]
#     sorted_contours = [contour for _, _, contour in detected_radii]
    
#     # Calculate differences between successive radii
#     differences = np.diff(sorted_radii)
    
#     # Find matching radii and corresponding contours based on ring distance
#     matching_radii = []
#     matching_contours = []
    
#     # for i, diff in enumerate(differences):
#     #     if diff >= ring_distance:
#     #         matching_radii.append(sorted_radii[i+1])
#     #         matching_contours.append(sorted_contours[i+1])
    
#     return sorted_radii, sorted_contours


def calculate_average_radius(contour, center_x, center_y):
    distances = []
    for point in contour:
        distance = np.sqrt((point[0][0] - center_x)**2 + (point[0][1] - center_y)**2)
        distances.append(distance)
    return np.mean(distances)

def detect_radius(contours, center_x, center_y, ring_distance=1):
    corresponding_contours_radii = []

    for contour in contours:
        average_radius = calculate_average_radius(contour, center_x, center_y)
        distance_to_center = np.sqrt((center_x - center_x)**2 + (center_y - center_y)**2)
        corresponding_contours_radii.append((distance_to_center, int(average_radius), contour))
    
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

    print("Sorted radii:", sorted_radii)
    print("Differences between radii:", differences)
    
    for i, diff in enumerate(differences):
        if diff >= ring_distance:
            matching_radii.append(sorted_radii[i+1])
            matching_contours.append(sorted_contours[i+1])
    
    return sorted_radii, sorted_contours, matching_radii, matching_contours


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


def find_colors(image, color_ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_masks = {}
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        color_masks[color_name] = mask
    return color_masks

def define_contours_for_radius(image):
    color_ranges = {
        'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
        'green': (np.array([40, 70, 50]), np.array([80, 255, 255])),
        # 'gray': (np.array([0, 0, 0]), np.array([179, 255, 100])),
        'gray': (np.array([0, 0, 0]), np.array([179, 255, 150]))

        
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
                    # global_center = (center[0] + i, center[1] + j)
                    global_center = (int(center[1] + j), int(center[0] + i))
                    if not any(np.linalg.norm(np.array(global_center) - np.array(existing_center)) < delta for existing_center in centers):
                        centers.append(global_center)
                        # cv2.circle(new_image, (int(global_center[1]), int(global_center[0])), 5, (0, 255, 0), -1)
                        cv2.circle(new_image, global_center, radius=5, color=(0, 255, 0), thickness=-1)
    # for center in centers:                 
    #     if center == (1460.1632653061224, 2404.0):
    #         cv2.circle(new_image, (int(global_center[1]), int(global_center[0])), 5, (0, 0, 255), -1)
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
    
    
def calculate_distance(cx, cy, radii, centroids):
    points = {}
    for i, centroid in enumerate(centroids):
        distance = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
        if distance >= radii[2]:
            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 10
        elif distance >= radii[3] and distance < radii[2]:

            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 20
        elif distance >= radii[4] and distance < radii[3]:

            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 30
        elif distance >= radii[5] and distance < radii[4]:

            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 40
        elif distance >= radii[6] and distance < radii[5]:

            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 50
        elif distance >= radii[7] and distance < radii[6]:

            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 60
        elif distance < radii[7]:

            print(distance)
            print(f'centroid: {centroid[0]} {centroid[1]}')
            points[f'point_{i}'] = 80
    return points 


# def calculate_distance(cx, cy, radii, centroids):
#     if len(radii) < 8:
#         raise ValueError("Список radii должен содержать как минимум 8 элементов.")

#     points = {}
#     scores = [10, 20, 30, 40, 50, 60, 80]  # Инвертированный список значений очков
    
#     for i, centroid in enumerate(centroids):
#         distance = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
#         print(f"distance: {distance}, centroid: {centroid}")

#         if distance < radii[2]:
#             print(f"Using radii[2]: {radii[2]}")
#             points[f'point_{i}'] = scores[6]
#         elif distance < radii[3]:
#             print(f"Using radii[3]: {radii[3]}")
#             points[f'point_{i}'] = scores[5]
#         elif distance < radii[4]:
#             print(f"Using radii[4]: {radii[4]}")
#             points[f'point_{i}'] = scores[4]
#         elif distance < radii[5]:
#             print(f"Using radii[5]: {radii[5]}")
#             points[f'point_{i}'] = scores[3]
#         elif distance < radii[6]:
#             print(f"Using radii[6]: {radii[6]}")
#             points[f'point_{i}'] = scores[2]
#         elif distance < radii[7]:
#             print(f"Using radii[7]: {radii[7]}")
#             points[f'point_{i}'] = scores[1]
#         else:
#             print(f"Distance is greater than radii[7]: {radii[7]}")
#             points[f'point_{i}'] = scores[0]

#     return points



def classificating_points(image, centroids, points):
    window_size = 100
    half_window = window_size // 2
    classified_points = {'red': 0, 'green': 0, 'other': 0}

    # Определение цветовых диапазонов в BGR
    red_color = [np.array([0, 0, 100]), np.array([100, 100, 255])]
    green_color = [np.array([0, 100, 0]), np.array([100, 255, 100])]

    for idx, (x, y) in enumerate(centroids):
        score = points[f'point_{idx}']
        
        red_count = 0
        green_count = 0
        
        for i in range(max(0, y - half_window), min(image.shape[0], y + half_window + 1)):
            for j in range(max(0, x - half_window), min(image.shape[1], x + half_window + 1)):
                point_value = image[i, j]
                print(f"Pixel value at ({i},{j}): {point_value}")

                # Проверка на зеленый цвет
                if (point_value >= green_color[0]).all() and (point_value <= green_color[1]).all():
                    green_count += 1

                # Проверка на красный цвет
                if (point_value >= red_color[0]).all() and (point_value <= red_color[1]).all():
                    red_count += 1
        
        print(f"Point {idx} ({x},{y}): Red count = {red_count}, Green count = {green_count}, Score = {score}")
        
        if red_count > green_count:
            classified_points['red'] += score
        elif green_count > red_count:
            classified_points['green'] += score
        else:
            classified_points['other'] += score
            
    print(f"Total red score: {classified_points['red']}")
    print(f"Total green score: {classified_points['green']}")

    if classified_points['red'] > classified_points['green']:
        return 'Red player WIN !!!!!!!!!!!!!!!!!!!!'
    elif classified_points['green'] > classified_points['red']:
        return 'Green player WIN !!!!!!!!!!!!!!!!!!!!'
    else:
        return 'It is a TIE !!!!!!!!!!!!!!!!!!!!'



if __name__ == '__main__':
    image_source, gray_source, blurred_source, edges_source = pre_process_image(source_image_path)
    contours_source, _ = cv2.findContours(edges_source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours_source)
    largest_contour = max(largest_contour, key=cv2.contourArea)
    
    masked_image = mask_space_out_of_darts(largest_contour, image_source, gray_source)
    cv2.imwrite(image_without_background_path, masked_image)
    print(f'Image without background saved as {image_without_background_path}')

    new_image, new_gray, new_blurred, new_edges = pre_process_image(image_without_background_path)
    new_2_image = new_image.copy()
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
    gray_mask = define_contours_for_radius(new_image)['gray']
    contours_for_gray, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours_for_gray))
    sorted_gray_contours = sorted(contours_for_gray, key=cv2.contourArea, reverse=True)
    nine_main_contours = sorted_gray_contours[1:9]
    print(f'Number of main contours: {len(nine_main_contours)}')

    new_contour_image = new_image.copy()
    cv2.drawContours(new_contour_image, nine_main_contours[-1], -1, (0, 255, 0), 3)
    cv2.imwrite('gray_contours.png', new_contour_image)
    # print('Masks ready !!!')
    # compoition_of_contours = []
    # for i in range(len(nine_main_contours)):
    #     compoition_of_contours.extend(separate_contours(new_image, nine_main_contours[i]))
    # renew_contour_image = new_image.copy()
    # cv2.drawContours(renew_contour_image, compoition_of_contours, -1, (0, 255, 0), 3)
    # cv2.imwrite('renew_contours.png', renew_contour_image)
    # devided_contour = separate_contours(gray_mask, nine_main_contours[5])
    
    
    for color_name, mask in color_masks.items():
        if color_name == 'target':
            target_mask = mask
        if color_name == 'red':
            red_mask = mask
        if color_name == 'green':
            green_mask = mask
    
    # palette = extract_colors(image=image_without_background_path, palette_size=10, resize = True)
    # palette.display(save_to_file=False)
    # main_colors = [color.rgb for color in palette]
    
    # main_colors = [[*color] for color in main_colors]
    # print(main_colors)
    # approximated_image, gold_mask_image, mask = colorizing_image_in_10_shades(new_image, main_colors)
    
    centroids = find_centers_of_clusters(new_image, target_mask, threshold=0.7)
    print(f'centr: {cx}, {cy}')
    print(f'Centroids: {centroids}')
    
    
    # # Save the result
    # cv2.imwrite('detected_rings.png', new_contour_image)
    # cv2.imwrite('edges.png', new_edges)
    # cv2.imwrite('blurred.png', new_blurred)
    # cv2.imwrite('gray.png', new_gray)
    
    
    matching_radii, matching_contours, _, _ = detect_radius(nine_main_contours, cx, cy, ring_distance=1)
    print(f'Matching radii: {matching_radii}, Matching contours: {len(matching_contours)}')
    new_contour_imagee = new_image.copy()
    cv2.drawContours(new_contour_imagee, matching_contours, -1, (0, 0, 255), 3)
    cv2.imwrite('select_circle_contour_image.png', new_contour_imagee)
    
    sorted_radius_contours = []
    for r, contour in zip(matching_radii, matching_contours):
        # if r < 50 or any(abs(r - radius) < 50 for radius, _ in sorted_radius_contours):
        #     continue
        sorted_radius_contours.append((r, contour))

    sorted_radius_contours = sorted(sorted_radius_contours, key=lambda x: x[0], reverse=True)
    
    if sorted_radius_contours:
        for radius, contour in sorted_radius_contours:
            print(f'Radius: {radius}')
            print(f'Corresponding contour length: {len(contour)}')
    else:
        print('No valid radii found')
        
    radii = [r for r, _ in sorted_radius_contours]
    print(len(radii))
    print(f'Sorted radius contours: {radii}')
    # calculate distance
    points = calculate_distance(cx, cy, radii, centroids)
    print(f'Points: {points}')

    
    # color_ranges = {
    # 'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
    # 'green': (np.array([35, 50, 50]), np.array([85, 255, 255])),
    # }
    
    res = classificating_points(new_2_image, centroids, points)
    print(res)