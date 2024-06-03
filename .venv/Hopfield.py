import numpy as np
import matplotlib.pyplot as plt


# Definir una clase para la red de Hopfield
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, pattern):
        pattern = pattern.reshape(-1, 1)
        self.weights += pattern @ pattern.T
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, steps=8):
        pattern = pattern.copy().reshape(-1)
        for _ in range(steps):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if net_input > 0 else -1
        return pattern.reshape((int(np.sqrt(self.size)), int(np.sqrt(self.size))))


# Función para generar un patrón de círculo
def generate_circle_pattern(radius):
    size = 2 * radius + 1
    pattern = np.zeros((size, size))
    center = (radius, radius)

    for i in range(size):
        for j in range(size):
            if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                pattern[i, j] = 1

    return pattern


# Función para binarizar una imagen en escala de grises
def binarize_image(image, threshold):
    binary_image = (image > threshold).astype(int)
    return binary_image


# Función para detectar posibles círculos en la imagen utilizando la red de Hopfield
def detect_circles(image, hopfield_net, pattern_size):
    rows, cols = image.shape
    detected_positions = []

    for i in range(0, rows - pattern_size + 1):
        for j in range(0, cols - pattern_size + 1):
            sub_image = image[i:i + pattern_size, j:j + pattern_size]
            pattern = sub_image.flatten()
            recovered_pattern = hopfield_net.predict(pattern)

            if np.array_equal(sub_image, recovered_pattern):
                detected_positions.append((i, j))

    return detected_positions


# Cargar la imagen y convertirla a escala de grises
image = plt.imread('C:/Users/manci/Documents/Python/prototipoHopfield/.venv/Image/ImagenMotor1.png')  # Reemplaza 'tu_imagen.png' con la ruta de tu imagen
if image is None:
    print("No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit()
gray_image = np.mean(image, axis=2)  # Convertir a escala de grises

# Binarizar la imagen
threshold = 0.5  # Umbral para binarizar la imagen
binary_image = binarize_image(gray_image, threshold)

# Generar un patrón de círculo y entrenar la red de Hopfield
circle_radius = 5# Radio del círculo
circle_pattern = generate_circle_pattern(circle_radius)
hopfield_net = HopfieldNetwork(size=circle_pattern.size)
hopfield_net.train(circle_pattern)

# Detectar posibles círculos en la imagen
detected_circles = detect_circles(binary_image, hopfield_net, pattern_size=circle_pattern.shape[0])

# Mostrar la imagen y los círculos detectados
plt.imshow(binary_image, cmap='gray')
for circle_pos in detected_circles:
    plt.plot(circle_pos[1] + circle_radius, circle_pos[0] + circle_radius, 'ro', markersize=2)
plt.title('Círculos Detectados')
plt.show()
