import sympy as sp
import numpy as np


def get_str_number(str_number):  # Parse string expressions (including pi) correctly
    if isinstance(str_number, str) and 'np.pi' in str_number:
        str_number = str_number.replace('np.pi', 'pi')
    # Use sympify to convert to a SymPy expression and evaluate
    return sp.sympify(str_number).evalf()


def get_str_numbers(str_numbers):
    return np.array([get_str_number(x) for x in str_numbers])


def get_list_str_numbers(list_str_numbers):
    return [get_str_numbers(x) for x in list_str_numbers]


def save_2_decimal_places(number: float):
    return round(number, 2)


def save_1_decimal_place(number: float):
    return round(number, 1)
