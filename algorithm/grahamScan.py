"""
see also : https://fr.wikipedia.org/wiki/Calcul_de_l%27enveloppe_convexe

Graham’s Scan computes the convex hull for a collection of Cartesian points. It
locates the lowest point, low, in the input set P and sorts the remaining points { P –
low } in reverse polar angle with respect to the lowest point. With this order in place,
the algorithm can traverse P clockwise from its lowest point. Every left turn of the
last three points in the hull being constructed reveals that the last hull point was
incorrectly chosen so it can be removed.

:param: A convex hull problem instance is defined by a collection of points, P.
:return: The output will be a sequence of (x, y) points representing a clockwise traversal of
         the convex hull. It shouldn’t matter which point is first.

This algorithm is suitable for Cartesian points. If the points, for example, use a different
coordinate system where increasing y values reflect lower points in the plane,
then the algorithm should compute low accordingly. Sorting the points by polar
angle requires trigonometric calculations.
"""


def convex_hull_graham_scan(points):
    '''
    Adapted from Tom Switzer
    '''
    from functools import reduce

    TURN_LEFT, TURN_RIGHT, TURN_NONE = 1, -1, 0

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        vectorial_product = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
        return cmp(vectorial_product, 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)  # find pivot
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hull = [[2, 4],
            [3, 5],
            [3, 3],
            [4, 6],
            [4, 4],
            [4, 2],
            [5, 5],
            [5, 3],
            [6, 4],
            [6, 6],
            [-2, 3]]

    x = [_x[0] for _x in hull]
    y = [_y[1] for _y in hull]
    plt.scatter(x, y)
    plt.show()

    convex_hull = convex_hull_graham_scan(hull)
    x = [_x[0] for _x in convex_hull]
    y = [_y[1] for _y in convex_hull]
    plt.scatter(x, y)
    plt.show()
