from unit_tests import *


if __name__ == '__main__':

    op = "alad"
    dataset = "toy"

    args = get_command_args(debug=True, debug_args=test_args_alad_loda(dataset, op))

    run_test(args)
