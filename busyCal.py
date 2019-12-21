import math

# ********************************* 一些常数设置************************
NEIGHBOUR_COUNT = 5  # 最近的多少个点算是邻居


# ***********************************************************************

def Cal_tan(p1, p2):
    if p1[1] == p2[1]:
        return 999999.
    return (p1[0] - p2[0]) / (p1[1] - p2[1])


def disance(p1, p2):
    return math.sqrt(sum((p1 - p2) ** 2))


class Line:
    def __init__(self, allPoints, stIndex, anIndex, threashold):
        '''

        :param allPoints:
        :param stIndex: 开始点下标
        :param anIndex: 另一个点下标 ， 要求开始点下标要在另一点之前
        :param threashold: 折线角度偏差阈值
        '''
        point_pool = allPoints.copy()
        startPoint = point_pool[stIndex]
        anotherPoint = point_pool[anIndex]

        del point_pool[anIndex]
        del point_pool[stIndex]

        self.point_pool = []
        for p in point_pool:
            min_dist = min(disance(p, startPoint), disance(p, anotherPoint))
            self.point_pool.append([p, min_dist])
        self.point_pool.sort(key=lambda x: x[1])

        self.points = [startPoint, anotherPoint]
        self.threashold = threashold  # 阈值用于寻找下一个最优点时作限制

        self.tan = Cal_tan(startPoint, anotherPoint)
        self.angle = math.atan(self.tan)

        self.max_error = 0

    def nextBestPoint(self):
        minerror = self.threashold
        minIndex = 0
        failPoint = []  # 失败临近点
        for i, p in enumerate(self.point_pool[:NEIGHBOUR_COUNT]):  # 候选5个邻近点
            error_ = self.max_error_angle(p[0])
            if error_ < minerror:
                minIndex = i
                minerror = error_
            elif error_ > self.threashold:
                failPoint.append(i)
        failPoint.append(minIndex)  # 顺便删除要加入的这个点
        failPoint.sort(reverse=True)

        if minerror < self.threashold:
            self.addPoint(self.point_pool[minIndex][0], failPoint)

            return True
        else:
            return False

    def addPoint(self, newPoint, deletePoint):
        # 更新所有的邻近
        self.points.append(newPoint)
        for fail in deletePoint:
            del self.point_pool[fail]

        for p in self.point_pool:
            newDist = disance(newPoint, p[0])
            if newDist < p[1]:
                p[1] = newDist

        self.point_pool.sort(key=lambda x: x[1])

    def freshProperty(self):
        p = self.points[-1]
        error_angle = self.max_error_angle(p)
        if self.max_error < error_angle:
            self.max_error = error_angle

    def max_error_angle(self, point):
        # 计算最大误差角
        if self.tan is None:
            return 0
        max_diff = 0
        for p in self.points:
            t_tan = Cal_tan(p, point)
            t_angle = math.atan(t_tan)
            diff = abs(t_angle - self.angle)
            if max_diff < diff:
                max_diff = diff
        return max_diff



def findLines(points, threashold):
    lines = []
    for i in range(len(points)):
        candidate = points.copy()
        _tcan = []
        for m, candi in enumerate(candidate):
            _tcan.append((m, candi, disance(points[i], candi)))
        _tcan.sort(key=lambda x: x[2])
        for m in _tcan[1:NEIGHBOUR_COUNT+1]: # 因为有一个自己
            if m[0]>i:
                lines.append(Line(points, i, m[0], threashold))

    result = []

    # 逐步加入最优点，除非无法加入最优点（阈值限制）
    haveNext = True
    while haveNext:
        haveNext = False
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].nextBestPoint():
                haveNext = True
            else:
                result.append(lines[i])
                del lines[i]
    return result
