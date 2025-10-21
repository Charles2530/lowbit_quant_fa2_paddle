import argparse


def get_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--method", type=str, default="fa2", choices=["fa2", "torch", "xformers"]
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    args = parser.parse_args()
    return args


def get_save_name(args):
    save_name = ""
    for arg in vars(args):
        save_name += arg + "_" + str(getattr(args, arg)) + "_"
    return save_name[:-1]
