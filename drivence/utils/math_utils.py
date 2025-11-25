import sympy as sp
import numpy as np


def get_str_number(str_number):  # Parse string expressions (including pi) correctly
    """
    解析字符串格式的数值或数学表达式（支持含 np.pi/pi 的表达式），返回浮点数值。

    核心逻辑：
    1. 处理含 numpy π 表示的字符串（将 'np.pi' 替换为 sympy 识别的 'pi'）；
    2. 使用 sympy.sympify 解析字符串表达式（支持加减乘除、幂运算等）；
    3. 计算表达式的浮点结果，适配数值字符串或复杂表达式的解析需求。

    Args:
        str_number (Union[str, float, int]): 输入数据，支持三种类型：
            - 字符串：可包含数值、数学运算符（+、-、*、/、** 等）、'np.pi' 或 'pi'；
            - 数值型（float/int）：直接返回原数值（兼容非字符串输入）。

    Returns:
        float: 解析后的浮点数值。
    """
    if isinstance(str_number, str) and 'np.pi' in str_number:
        str_number = str_number.replace('np.pi', 'pi')
    # Use sympify to convert to a SymPy expression and evaluate
    return sp.sympify(str_number).evalf()


def get_str_numbers(str_numbers):
    """
    批量解析字符串数值/表达式列表（支持含 np.pi/pi 的表达式），返回 numpy 数组。

    Args:
        str_numbers (Union[List[Union[str, float, int]], np.ndarray]): 输入列表/数组，
            每个元素为字符串（数值或表达式）、float 或 int。

    Returns:
        np.ndarray: 解析后的浮点型 numpy 数组，形状与输入一致。
    """
    return np.array([get_str_number(x) for x in str_numbers])


def get_list_str_numbers(list_str_numbers):
    """
    批量解析二维列表/数组中的字符串数值/表达式（支持含 np.pi/pi 的表达式），返回 numpy 数组列表。

    Args:
        list_str_numbers (List[Union[List[Union[str, float, int]], np.ndarray]]): 二维输入列表，
            每个子列表/子数组的元素为字符串（数值或表达式）、float 或 int。

    Returns:
        List[np.ndarray]: 解析后的浮点型 numpy 数组列表，每个子数组与输入子列表形状一致。
    """
    return [get_str_numbers(x) for x in list_str_numbers]


def save_2_decimal_places(number: float):
    return round(number, 2)


def save_1_decimal_place(number: float):
    return round(number, 1)
