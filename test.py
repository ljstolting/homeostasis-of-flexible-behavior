def sum_digits(n):
    place = 1
    if n < 10:
        num = n
    else:
        while n>10:
            num = n//10
            place += 1
    return num + sum_digits(n-(10**place)+num)

sum_digits(1204)