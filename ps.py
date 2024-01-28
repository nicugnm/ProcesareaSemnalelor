def nearest_neighbor_downsample(image, new_width, new_height):
    """
    Redimensionează o imagine folosind interpolarea celor mai apropiați vecini (downsampling).

    :param image: Imaginea originală reprezentată ca o listă 2D de valori ale pixelilor.
    :param new_width: Lățimea imaginii redimensionate.
    :param new_height: Înălțimea imaginii redimensionate.
    :return: Imaginea redimensionată ca o listă 2D de valori ale pixelilor.
    """
    original_height = len(image)
    original_width = len(image[0])

    # Calcularea raportului dintre dimensiunile noi și cele originale
    width_ratio = original_width / new_width
    height_ratio = original_height / new_height

    # Crearea noii imagini cu dimensiunile specificate
    downsampled_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    for i in range(new_height):
        for j in range(new_width):
            # Găsirea celui mai apropiat vecin din imaginea originală
            orig_i = int(i * height_ratio)
            orig_j = int(j * width_ratio)
            downsampled_image[i][j] = image[orig_i][orig_j]

    return downsampled_image


def nearest_neighbor_upsample(image, new_width, new_height):

    # Mărește dimensiunea unei imagini folosind interpolarea celor mai apropiați vecini (upsampling).
    original_height = len(image)
    original_width = len(image[0])

    # Calcularea raportului dintre dimensiunile noi și cele originale
    width_ratio = original_width / new_width
    height_ratio = original_height / new_height

    # Crearea noii imagini cu dimensiunile specificate
    upsampled_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    for i in range(new_height):
        for j in range(new_width):
            # Găsirea celui mai apropiat vecin din imaginea originală
            orig_i = int(i * height_ratio)
            orig_j = int(j * width_ratio)
            upsampled_image[i][j] = image[orig_i][orig_j]

    return upsampled_image


def mean_squared_error(original_image, recovered_image):
    rows = len(original_image)
    cols = len(original_image[0])
    mse = sum(sum((original_image[i][j] - recovered_image[i][j]) ** 2 for j in range(cols)) for i in range(rows))
    return mse / (rows * cols)


def print_image(image):

    # Afișează imaginea reprezentată ca o listă 2D de valori ale pixelilor.
    for row in image:
        print(" ".join(str(pixel) for pixel in row))


if __name__ == '__main__':
    example_image = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    print("Imaginea originala:")
    print_image(example_image)

    downsampled_image = nearest_neighbor_downsample(example_image, 4, 4)
    print("\nImaginea downsampled:")
    print_image(downsampled_image)

    upsampled_image = nearest_neighbor_upsample(downsampled_image, 5, 5)
    print("\nImaginea upsampled:")
    print_image(upsampled_image)

    # Calcularea MSE
    mse_error = mean_squared_error(example_image, upsampled_image)
    print("\nEroarea MSE este:", mse_error)