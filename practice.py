for n in range(1, 10):
    for m in range(1, 10):
        if m <= n:
            print(n, "*", m, "=", n * m, end="\t")
    print()
