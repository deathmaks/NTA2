import time
import random
import argparse
import signal
from sympy.ntheory import factorint, isprime, nextprime


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)


def discrete_log_brute_force(alpha, beta, p, time_limit=300):
    print(f"Running Brute Force for alpha={alpha}, beta={beta}, p={p}")
    start_time = time.time()
    next_report_time = start_time + 30

    signal.alarm(time_limit)

    try:
        for x in range(p):
            if pow(alpha, x, p) == beta:
                print(f"Result found: x={x}")
                return x
            current_time = time.time()
            if current_time > next_report_time:
                elapsed_time = current_time - start_time
                print(f"Progress: {elapsed_time:.2f} seconds elapsed")
                next_report_time += 30
        print("Result not found")
    except TimeoutException:
        print("Time limit exceeded")
    finally:
        signal.alarm(0)

    return None




def silver_pohlig_hellman(alpha, beta, p, time_limit=300):
    print(f"Running Silver-Pohlig-Hellman for alpha={alpha}, beta={beta}, p={p}")
    start_time = time.time()
    next_report_time = start_time + 30

    signal.alarm(time_limit)

    try:
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
            try:
                qe = q ** e
                alpha_qe = pow(alpha, n // qe, p)
                beta_qe = pow(beta, n // qe, p)

                # Use a dictionary for quick lookup
                table = {pow(alpha_qe, j, p): j for j in range(qe)}
                x_qe = 0

                for i in range(e):
                    alpha_qei = pow(alpha_qe, q ** i, p)
                    beta_qei = (pow(beta_qe * pow(alpha_qe, -x_qe, p), q ** i, p)) % p

                    print(f"alpha_qei={alpha_qei}, beta_qei={beta_qei}")  # Debug message

                    if beta_qei in table:
                        x_qe += table[beta_qei] * (q ** i)
                    else:
                        print(f"Debug: alpha_qei = {alpha_qei}, beta_qei = {beta_qei}")
                        print(f"Debug: table keys = {list(table.keys())}")
                        print(f"Debug: alpha_qe = {alpha_qe}, beta_qe = {beta_qe}")
                        raise ValueError(f"Table lookup failed for beta_qei={beta_qei}")

                residues.append(x_qe)
                moduli.append(qe)
            except Exception as e:
                print(f"An error occurred while processing factor {q}^{e}: {e}")
                raise e

            current_time = time.time()
            if current_time > next_report_time:
                elapsed_time = current_time - start_time
                print(f"Progress: {elapsed_time:.2f} seconds elapsed")
                next_report_time += 30

        result = crt(residues, moduli)
        print(f"Result found: x={result}")
        return result
    except TimeoutException:
        print("Time limit exceeded")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        signal.alarm(0)  # Disable the alarm

    return None

def generate_data(num_samples):
    p_values = []
    brute_force_times = []
    sp_h_times = []

    p = nextprime(random.randint(2, 10))
    for _ in range(num_samples):
        alpha = random.randint(2, p - 1)
        beta = random.randint(1, p - 1)

        print(f"Generated alpha={alpha}, beta={beta}, p={p}")

        start_time = time.time()
        result_brute = discrete_log_brute_force(alpha, beta, p)
        elapsed_time_brute = time.time() - start_time

        start_time = time.time()
        result_sph = silver_pohlig_hellman(alpha, beta, p)
        elapsed_time_sph = time.time() - start_time

        p_values.append(p)
        brute_force_times.append(elapsed_time_brute)
        sp_h_times.append(elapsed_time_sph)

        p = nextprime(p)

    return p_values, brute_force_times, sp_h_times


def generate_test_cases(max_digits):
    test_cases = []
    for digits in range(1, max_digits + 1):
        p = nextprime(10 ** (digits - 1))
        for _ in range(2):
            if p > 2:
                alpha = random.randint(2, p - 1)
                beta = random.randint(1, p - 1)
                test_cases.append((alpha, beta, p, digits))
            p = nextprime(p)
    return test_cases


def generate_identical_test_cases(max_digits):
    test_cases = []
    p = nextprime(10 ** (max_digits - 1))
    for _ in range(2):
        if p > 2:
            alpha = random.randint(2, p - 1)
            beta = random.randint(1, p - 1)
            test_cases.append((alpha, beta, p, max_digits))
        p = nextprime(p)
    return test_cases


def run_tests(test_cases, algorithm, total_time_limit=600):
    results = []
    start_time_total = time.time()

    for index, (alpha, beta, p, digits) in enumerate(test_cases):
        current_time_total = time.time()
        if current_time_total - start_time_total > total_time_limit:
            print("Total time limit exceeded")
            break

        test_type = "Type 1" if index % 2 == 0 else "Type 2"
        print(f"Running test case: {test_type} alpha={alpha}, beta={beta}, p={p}, digits={digits}")

        start_time = time.time()
        if algorithm == 'bruteforce':
            result = discrete_log_brute_force(alpha, beta, p)
        else:
            result = silver_pohlig_hellman(alpha, beta, p)
        elapsed_time = time.time() - start_time

        results.append((alpha, beta, p, digits, result, elapsed_time, test_type))

    total_elapsed_time = time.time() - start_time_total
    return results, total_elapsed_time


def print_results(results, total_time):
    print(f"{'alpha':<10} {'beta':<10} {'p':<20} {'digits':<10} {'result':<10} {'time (s)':<10} {'type':<10}")
    print("=" * 80)
    for alpha, beta, p, digits, result, elapsed_time, test_type in results:
        result_str = str(result) if result is not None else 'None'
        print(f"{alpha:<10} {beta:<10} {p:<20} {digits:<10} {result_str:<10} {elapsed_time:<10.5f} {test_type:<10}")
    print(f"\nTotal execution time: {total_time:.5f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Discrete Logarithm Problem Solver')
    parser.add_argument('--algorithm', type=str, choices=['bruteforce', 'sph'],
                        help='Algorithm to use: bruteforce or sph')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark with generated data')
    parser.add_argument('--test', nargs=2, type=int,
                        help='Run 3 modes of testing with different number of digits for brute force and SPH respectively')

    args = parser.parse_args()

    if args.benchmark:
        start_time = time.time()
        p_values, brute_force_times, sp_h_times = generate_data(10)
        total_time = time.time() - start_time
        print_results([(alpha, beta, p, '', result, b_time, 'Brute Force') for (alpha, beta, p), (b_time, result) in
                       zip(zip(p_values, p_values, p_values), zip(brute_force_times, brute_force_times))], total_time)
        print_results([(alpha, beta, p, '', result, s_time, 'SPH') for (alpha, beta, p), (s_time, result) in
                       zip(zip(p_values, p_values, p_values), zip(sp_h_times, sp_h_times))], total_time)
    elif args.test:
        brute_force_digits, sph_digits = args.test

        if brute_force_digits == sph_digits:
            test_cases = generate_identical_test_cases(brute_force_digits)
            print(f"Running tests with identical test cases for both algorithms")
            results_bf, total_time_bf = run_tests(test_cases, 'bruteforce')
            print_results(results_bf, total_time_bf)
            print()

            results_sph, total_time_sph = run_tests(test_cases, 'sph')
            print_results(results_sph, total_time_sph)
            print()
        else:
            bf_test_cases = generate_test_cases(brute_force_digits)
            sph_test_cases = generate_test_cases(sph_digits)

            print(f"Running tests with algorithm: bruteforce")
            results_bf, total_time_bf = run_tests(bf_test_cases, 'bruteforce')
            print_results(results_bf, total_time_bf)
            print()

            print(f"Running tests with algorithm: sph")
            results_sph, total_time_sph = run_tests(sph_test_cases, 'sph')
            print_results(results_sph, total_time_sph)
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


