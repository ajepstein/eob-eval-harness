"""Best integer fit of Z = a*X^2 + b*Y^2 under Mean Absolute Error."""

LOW, HIGH = -100, 100


def solution(X, Y, Z):
    """Return [a, b] minimizing MAE of Z_i vs a*X_i^2 + b*Y_i^2.

    a and b are searched over the integers in [-100, 100]. Ties are broken by
    the smallest a, then the smallest b.
    """
    xs = [x * x for x in X]
    ys = [y * y for y in Y]
    n = len(Z)
    if n == 0:
        return [0, 0]

    best = None
    best_err = float("inf")
    for a in range(LOW, HIGH + 1):
        # Residual left for b*Y^2 to explain.
        res = [z - a * x2 for z, x2 in zip(Z, xs)]
        for b in range(LOW, HIGH + 1):
            err = 0
            for r, y2 in zip(res, ys):
                err += abs(r - b * y2)
                if err >= best_err:      # can only grow; bail out early
                    break
            else:
                best_err = err
                best = [a, b]
    return best


if __name__ == "__main__":
    print(solution([1, 2], [2, 2], [9, 12]))                    # [1, 2]
    print(solution([1, -2, 3], [-1, 2, -100], [2, 8, 10008]))   # [1, 1]
