import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------
# CELL CLASS — each cell evaluates a location in the image
# ---------------------------------------------------------
class Cell:
    def __init__(self, x, y, window_size):
        self.x = x
        self.y = y
        self.window = window_size
        self.fitness = 0

    def evaluate(self, img):
        # extract window around (x,y)
        h, w = img.shape[:2]
        x1 = max(0, self.x - self.window)
        x2 = min(w, self.x + self.window)

        y1 = max(0, self.y - self.window)
        y2 = min(h, self.y + self.window)

        region = img[y1:y2, x1:x2]
        
        # compute fitness = gradient energy (edge strength)
        gx = cv2.Sobel(region, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(region, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(gx**2 + gy**2)

        self.fitness = grad_mag.mean()
        return self.fitness

# ---------------------------------------------------------
# PROCESSING FUNCTION FOR MULTIPROCESSING
# ---------------------------------------------------------
def evaluate_cell(args):
    cell, img = args
    cell.evaluate(img)
    return cell

# ---------------------------------------------------------
# PARALLEL CELLULAR ALGORITHM
# ---------------------------------------------------------
def parallel_cellular_detection(img, n_cells=500, window=5, generations=10):

    h, w = img.shape[:2]

    # initialize cells in random positions
    cells = [
        Cell(
            x=np.random.randint(0, w),
            y=np.random.randint(0, h),
            window_size=window
        )
        for _ in range(n_cells)
    ]

    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")

        # parallel evaluation
        with Pool(cpu_count()) as p:
            cells = p.map(evaluate_cell, [(c, img) for c in cells])

        # select top 50%
        cells.sort(key=lambda c: c.fitness, reverse=True)
        survivors = cells[: n_cells // 2]

        # reproduction (mutate positions)
        new_cells = []
        for c in survivors:
            for _ in range(2):  # two children
                child = Cell(
                    x=min(w-1, max(0, c.x + np.random.randint(-3, 3))),
                    y=min(h-1, max(0, c.y + np.random.randint(-3, 3))),
                    window_size=window
                )
                new_cells.append(child)

        cells = new_cells

    # final best cells = detected features
    cells.sort(key=lambda c: c.fitness, reverse=True)
    return cells[:100]   # top 100 “detected” pixels

# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
def draw_detections(img, detections):
    out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for c in detections:
        cv2.circle(out, (c.x, c.y), 2, (0,0,255), -1)

    return out

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # load image
    img = cv2.imread("input.jpg", 0)

    best_cells = parallel_cellular_detection(
        img,
        n_cells=500,
        window=5,
        generations=8
    )

    result = draw_detections(img, best_cells)

    cv2.imwrite("detected.png", result)
    print("Detection completed → saved as detected.png")
