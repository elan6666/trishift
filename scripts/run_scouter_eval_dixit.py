try:
    from scripts.scouter_eval import run_scouter_eval
except ModuleNotFoundError:
    from scouter_eval import run_scouter_eval


def main() -> None:
    run_scouter_eval("dixit")


if __name__ == "__main__":
    main()