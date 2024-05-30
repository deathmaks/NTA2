import time
import random
import argparse
from sympy.ntheory import factorint, isprime, nextprime
import matplotlib.pyplot as plt


def discrete_log_brute_force(alpha, beta, p):
    print(f"Running Brute Force for alpha={alpha}, beta={beta}, p={p}")
    for x in range(p):
        if pow(alpha, x, p) == beta:
            print(f"Result found: x={x}")
            return x
    print("Result not found")
    return None

def silver_pohlig_hellman(alpha, beta, p):
    print(f"Running Silver-Pohlig-Hellman for alpha={alpha}, beta={beta}, p={p}")
    def crt(residues, moduli):
        x = 0
        N = 1
        for modulus in moduli:
            N *= modulus
        for residue, modulus in zip(residues, moduli):
            n = N // modulus
            inv = pow(n, -1, modulus)
            x = (x + residue * inv * n) % N
        return x

    n = p - 1
    factors = factorint(n)
    residues = []
    moduli = []

    for q, e in factors.items():
        qe = q ** e
        alpha_qe = pow(alpha, n // qe, p)
        beta_qe = pow(beta, n // qe, p)

        table = {pow(alpha_qe, j, p): j for j in range(qe)}
        x_qe = 0

        for i in range(e):
            alpha_qei = pow(alpha, n // (q ** (i + 1)), p)
            beta_qei = (pow(beta * pow(alpha, -x_qe, p), n // (q ** (i + 1)), p)) % p

            if beta_qei in table:
                x_qe += table[beta_qei] * (q ** i)

        residues.append(x_qe)
        moduli.append(qe)

    result = crt(residues, moduli)
    print(f"Result found: x={result}")
    return result


def generate_data(num_samples):
    p_values = []
    brute_force_times = []
    sp_h_times = []

    p = nextprime(random.randint(2, 10))
    for _ in range(num_samples):
        alpha = random.randint(2, p-1)
        beta = random.randint(1, p-1)

        print(f"Generated alpha={alpha}, beta={beta}, p={p}")


        start_time = time.time()
        result_brute = discrete_log_brute_force(alpha, beta, p)
        elapsed_time_brute = time.time() - start_time


        start_time = time.time()
        result_sph = silver_pohlig_hellman(alpha, beta, p)
        elapsed_time_sph = time.time() - start_time

        p_values.append((alpha, beta, p))
        brute_force_times.append(elapsed_time_brute)
        sp_h_times.append(elapsed_time_sph)

        p = nextprime(p)

    return p_values, brute_force_times, sp_h_times

# Генерация данных для 3 режимов
def generate_test_cases(max_digits):
    test_cases = []
    for digits in range(1, max_digits + 1):
        p = nextprime(10**(digits-1))
        for _ in range(2):
            if p > 2:
                alpha = random.randint(2, p-1)
                beta = random.randint(1, p-1)
                test_cases.append((alpha, beta, p, digits))
            p = nextprime(p)
    return test_cases


def plot_results(p_values, times, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, times, marker='o')
    plt.title(title)
    plt.xlabel('p')
    plt.ylabel('Execution Time (s)')
    plt.grid(True)
    plt.savefig(filename)

# Запуск алгоритмов с измерением времени
def run_tests(test_cases, algorithm):
    results = []
    for index, (alpha, beta, p, digits) in enumerate(test_cases):
        test_type = "Type 1" if index % 2 == 0 else "Type 2"
        print(f"Running test case: {test_type} alpha={alpha}, beta={beta}, p={p}, digits={digits}")
        start_time = time.time()
        if algorithm == 'bruteforce':
            result = discrete_log_brute_force(alpha, beta, p)
        else:
            result = silver_pohlig_hellman(alpha, beta, p)
        elapsed_time = time.time() - start_time
        results.append((alpha, beta, p, digits, result, elapsed_time, test_type))
    return results

# Печать результатов в табличном виде
def print_results(results, total_time):
    print(f"{'alpha':<10} {'beta':<10} {'p':<20} {'digits':<10} {'result':<10} {'time (s)':<10} {'type':<10}")
    print("="*80)
    for alpha, beta, p, digits, result, elapsed_time, test_type in results:
        result_str = str(result) if result is not None else 'None'
        print(f"{alpha:<10} {beta:<10} {p:<20} {digits:<10} {result_str:<10} {elapsed_time:<10.5f} {test_type:<10}")
    print(f"\nTotal execution time: {total_time:.5f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Discrete Logarithm Problem Solver')
    parser.add_argument('--algorithm', type=str, choices=['bruteforce', 'sph'], help='Algorithm to use: bruteforce or sph')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark with generated data')
    parser.add_argument('--test', type=int, help='Run 3 modes of testing with different number of digits up to the specified value')

    args = parser.parse_args()

    if args.benchmark:
        start_time = time.time()
        p_values, brute_force_times, sp_h_times = generate_data(10)
        plot_results([p for _, _, p in p_values], brute_force_times, 'Benchmark Brute Force Execution Time', 'benchmark_bruteforce_results.png')
        plot_results([p for _, _, p in p_values], sp_h_times, 'Benchmark SPH Execution Time', 'benchmark_sph_results.png')
        total_time = time.time() - start_time
        print_results([(alpha, beta, p, '', result, b_time, 'Brute Force') for (alpha, beta, p), (b_time, result) in zip(p_values, zip(brute_force_times, brute_force_times))], total_time)
        print_results([(alpha, beta, p, '', result, s_time, 'SPH') for (alpha, beta, p), (s_time, result) in zip(p_values, zip(sp_h_times, sp_h_times))], total_time)
    elif args.test:
        test_cases = generate_test_cases(args.test)
        for algo in ['bruteforce', 'sph']:
            print(f"Running tests with algorithm: {algo}")
            start_time = time.time()
            results = run_tests(test_cases, algo)
            total_time = time.time() - start_time
            print_results(results, total_time)
            p_values = [p for _, _, p, _, _, _, _ in results]
            times = [time for _, _, _, _, _, time, _ in results]
            plot_results(p_values, times, f'{algo} Execution Time', f'{algo}_test_results.png')
            print()
    elif args.algorithm:
        alpha = int(input("Enter the value of alpha: "))
        beta = int(input("Enter the value of beta: "))
        p = int(input("Enter the value of p (a prime number): "))

        start_time = time.time()
        if args.algorithm == 'bruteforce':
            result = discrete_log_brute_force(alpha, beta, p)
        elif args.algorithm == 'sph':
            result = silver_pohlig_hellman(alpha, beta, p)
        elapsed_time = time.time() - start_time

        print(f'Result ({args.algorithm}): x = {result}')
        print(f'Execution time: {elapsed_time} seconds')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
