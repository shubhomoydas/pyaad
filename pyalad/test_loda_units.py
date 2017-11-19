from unit_tests import *


"""
python pyalad/test_loda_units.py
"""


if __name__ == '__main__':

    op = "loda"
    dataset = "toy"

    args = get_command_args(debug=True, debug_args=test_args_alad_loda(dataset, op))

    run_test(args)
