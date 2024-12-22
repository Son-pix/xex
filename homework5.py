class Solution:
    def __init__(self, filename):
        self.otrezki = []
        self.read_otrezki(filename)

    def read_otrezki(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                x1, y1, x2, y2, nomer = map(float, line.split())
                self.otrezki.append(((x1, y1), (x2, y2), int(nomer)))

    def area_sootnoshenie(self):
        sootnoshenies = []
        for (x1, y1), (x2, y2), nomer in self.otrezki:
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            area_left = x1 * y1
            area_right = 1 - x2 * y2
            area_top = (1 - y2) * (x2 - x1)
            area_bottom = y1 * (x2 - x1)

            total_area = 1
            sootnoshenie = (area_left + area_right + area_top + area_bottom) / total_area
            sootnoshenies.append((nomer, sootnoshenie))

        return sootnoshenies

    def peresechenie(self, p1, p2, p3, p4):
        znamenat = (p4[0] - p3[0]) * (p2[1] - p1[1]) - (p2[0] - p1[0]) * (p4[1] - p3[1])
        if znamenat == 0:
            return None  # параллельные линии

        ua = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) / znamenat
        ub = ((p4[0] - p3[0]) * (p3[1] - p1[1]) - (p4[1] - p3[1]) * (p3[0] - p1[0])) / znamenat

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = p1[0] + ua * (p2[0] - p1[0])
            y = p1[1] + ua * (p2[1] - p1[1])
            return (x, y)
        return None

    def find_peresechenies(self):
        peresechenies_points = {}
        for i in range(len(self.otrezki)):
            for j in range(i + 1, len(self.otrezki)):
                p1, p2, nomerx1 = self.otrezki[i]
                p3, p4, nomerx2 = self.otrezki[j]
                point = self.peresechenie(p1, p2, p3, p4)
                if point and 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                    if point in peresechenies_points:
                        peresechenies_points[point] += 1
                    else:
                        peresechenies_points[point] = 1

        return peresechenies_points

    def find_triple_peresechenies(self):
        peresechenies_points = self.find_peresechenies()
        triple_peresechenies = [point for point, count in peresechenies_points.items() if count >= 3]

        if triple_peresechenies:
            return triple_peresechenies
        else:
            print("Таких точек не найдено")
            return []

def write_otrezki_to_file(filename):
    otrezki = [
        (0.0, 0.0, 0.5, 0.5, 1), 
        (0.0, 0.5, 0.5, 0.0, 2),
        (0.25, 0.0, 0.25, 0.5, 3),
        (0.4, 0.5, 0.8, 0.2, 4),
        (0.7, 0.6, 0.9, 0.9, 5)
    ]

    with open(filename, 'w') as file:
        for otrezok in otrezki:
            line = ' '.join(map(str, otrezok)) + '\n'
            file.write(line)

filename = 'otrezki.txt'
write_otrezki_to_file(filename)

solution = Solution(filename)

area_sootnoshenies = solution.area_sootnoshenie()
print("Соотношения площадей для каждого отрезка:")
for nomer, sootnoshenie in area_sootnoshenies:
    print(f"Отрезок {nomer}: Соотношение площадей = {sootnoshenie:.4f}")

triple_peresechenies = solution.find_triple_peresechenies()
if triple_peresechenies:
    print("Точки пересечения, где пересекаются минимум три отрезка:")
    for point in triple_peresechenies:
        print(point)
