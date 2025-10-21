def prError(skk, stdout=True):
    if stdout:
        print("\x1b[91m{}\x1b[00m".format(skk))


def prSuccess(skk, stdout=True):
    if stdout:
        print("\x1b[92m{}\x1b[00m".format(skk))


def prImportant(skk, stdout=True):
    if stdout:
        print("\x1b[93m{}\x1b[00m".format(skk))


def prProcess(skk, stdout=True):
    if stdout:
        print("\x1b[94m{}\x1b[00m".format(skk))


def prWatch(skk, stdout=True):
    if stdout:
        print("\x1b[96m{}\x1b[00m".format(skk))


def debug(debug_mode=True):
    if debug_mode:
        import pdb

        pdb.set_trace()


def process_remaining_args(arg_list):
    class CustomArgs:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                key = key.replace("-", "")
                setattr(self, key, value)

        def keys(self):
            return self.__dict__.keys()

    _dict = {}
    for i in range(len(arg_list) // 2):
        _dict[arg_list[2 * i]] = arg_list[2 * i + 1]
    custom_args = CustomArgs(_dict)
    return custom_args
