from global_optimization.Args import Args
from global_optimization.CellUniverse import CellUniverse
import typed_argparse as tap

def main(args: Args):
    cellUniverse = CellUniverse(args)
    cellUniverse.run()

if __name__ == '__main__':
    tap.Parser(Args).bind(main).run()