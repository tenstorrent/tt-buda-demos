
# Example of a Python script with linting issues

def add_numbers(a, b):
    # Missing whitespace after comma
    return a+b  # Missing whitespace around operator


def main():
    result = add_numbers(5,10)  # Unused variable 'result'
    print("The result is:", result)


if __name__ == "__main__":
    main()
