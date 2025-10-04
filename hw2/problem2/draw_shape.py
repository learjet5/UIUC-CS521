import matplotlib.pyplot as plt

points = [(7, 6), (5, 2), (1, 0), (3, 4)]

x = [p[0] for p in points]
y = [p[1] for p in points]

x.append(points[0][0])
y.append(points[0][1])

plt.scatter([p[0] for p in points], [p[1] for p in points], color='red', s=80, label='Points')

plt.plot(x, y, color='blue', linestyle='-', linewidth=2, label='Polygon')

for (px, py) in points:
    plt.text(px + 0.1, py + 0.1, f"({px}, {py})", fontsize=10)

plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend()

plt.show()