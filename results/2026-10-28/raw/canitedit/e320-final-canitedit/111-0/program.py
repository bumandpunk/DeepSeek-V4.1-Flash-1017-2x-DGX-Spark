import math

def gcd(a : int, b : int) -> int:
    """Compute the Greatest Common Divisor (GCD) of a and b."""
    assert a > 0 and b > 0
    while b != 0:
        a, b = b, a % b
    return a

def euler_totient(n : int) -> int:
    """Compute the Euler's Totient function of n."""
    assert n > 0 
    if n == 1 : return 1
    count = 0
    for i in range(1, n):
        if gcd(i, n) == 1:
            count += 1
    return count

def powermod(base : int, exp : int, mod : int) -> int:
    """Compute modular exponentiation, base^exp mod mod, via successive squaring."""
    assert exp >= 0 and mod > 0
    if exp == 0:
        return 1
    half = powermod(base, exp // 2, mod)
    result = (half * half) % mod
    if exp % 2 == 1:
        result = (result * base) % mod
    return result

def check_coprime_euler(a : int, b : int):
    assert a > 0 and b > 0
    return powermod(a, euler_totient(b), b) == 1

### START TESTS ###
if True: # pragma: no cover
    assert gcd(1,1) == 1
    assert gcd(1,2) == 1
    assert gcd(3,7) == 1
    assert gcd(4,2) == 2
    assert gcd(3123,312) == 3
    assert gcd(25,45) == 5
    assert gcd(987, 987) == 987

    for i in range(1,50):
        for j in range(1,50):
            assert gcd(i,j) == math.gcd(i,j)

    assert euler_totient(18) == 6
    assert euler_totient(5913) == 3888
    assert euler_totient(1) == 1

    assert check_coprime_euler(1,1) == False

    # recall: two numbers are coprime if and only if their gcd is 1
    for i in range(1,50):
        for j in range(2,50):
            assert (gcd(i,j) == 1) == check_coprime_euler(i,j)
print('SPARK1017_TESTS_PASSED_SENTINEL')
