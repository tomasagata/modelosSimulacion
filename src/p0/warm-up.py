"""
Completar las siguientes funciones definidas de modo que cumplan
con la logica solicitada. El proposito de este ejercicio es recordar
elementos de Python (req. Informatica General). Se asignan tests previamente
definidos (no es necesario agregar nuevos casos).
"""


def list_has_even_size(lst):
    """
    Given a list return True if its size
    is an even number, otherwise False

    >>> list_has_even_size([1, 2, 3])
    False
    >>> list_has_even_size(['a', 'b', 'c', 'd'])
    True
    >>> list_has_even_size([])
    True
    """

    return len(lst) % 2


def sum_of_elements(lst):
    """
    Given a list with integer numbers
    return the sum of all the elements.
    >>> sum_of_elements([1, 25, 45, 30])
    101
    >>> sum_of_elements([5, 3, -1])
    7
    >>> sum_of_elements([])
    0
    >>> sum_of_elements([-5, -2])
    -7
    """

    return sum(lst)


def remove_elements(array):
    """
    Given a 2d array, return an array
    that contains only the rows that have not
    None as an element
    >>> remove_elements([[1, 2], [3, 4]])
    [[1, 2], [3, 4]]
    >>> remove_elements([[1, None], [2, 3]])
    [[2, 3]]
    >>> remove_elements([[1, None], [1, None]])
    []
    """

    resulting = []
    for row in range(len(array)):
        if None not in row:
            resulting.append(row)

    return resulting


def replace_value(array):
    """
    Given a 2d array, replace every 'x'/'X' character
    with an 'o'/'O' character.
    >>> replace_value([['a', 'x'], ['o', 'b']])
    [['a', 'o'], ['o', 'b']]
    >>> replace_value([['a', 'b'], ['c', 'd']])
    [['a', 'b'], ['c', 'd']]
    >>> replace_value([['X', 'x'], ['xx', 'XX']])
    [['O', 'o'], ['xx', 'XX']]
    """

    resulting = []
    for row in array:
        new_row = []
        for column in row:
            if column == 'x':
                new_row.append('o')
            elif column == 'X':
                new_row.append('O')
            else:
                new_row.append(column)
        resulting.append(new_row)

    return resulting
