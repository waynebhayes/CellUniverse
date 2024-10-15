from CellUniverse import CellUniverse, Args
import typed_argparse as tap

def main(args: Args):
    cellUniverse = CellUniverse(args)
    cellUniverse.run()

if __name__ == '__main__':
    tap.Parser(Args).bind(main).run()