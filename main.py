import sys
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural Architecture Search with Ray")
    """
        General arguments for NAS
    """
    parser.add_argument("--smoke-test", default=False, action="store_true", help="Finish quickly for testing")
    parser.add_argument("--ray-address", default=None, help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enables GPU training")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size of supervised learning")
    parser.add_argument('--num_workers', type=int, default=2, help="workers for torch data loaders")

    usage_string = "Usage: python main.py [darts|enas] [cnn|rnn|viz]"

    if len(sys.argv) < 3:
        print(usage_string)
    
    elif sys.argv[1] == 'darts':

        """
            General arguments for DARTS
        """
        parser.add_argument("--layers", default=10, type=int, help="Number of layers in model")

        if sys.argv[2] == 'cnn':
            from nas.cnn_darts import run_experiment
            sys.argv.remove(sys.argv[2])
            sys.argv.remove(sys.argv[1])
            dataset_options = ["cifar10", "mnist", "imagenet"]
            """
                CNN specific arguments for DARTS
            """
            parser.add_argument("--dataset", default="cifar10", choices=dataset_options, type=str.lower, help="Name of dataset")
            args = parser.parse_args()

            run_experiment(args)

        elif sys.argv[2] == 'rnn':
            # from nas.rnn_darts import run_experiment
            sys.argv.remove(sys.argv[2])
            sys.argv.remove(sys.argv[1])
            dataset_options = ["ptb", "wikitext"]
            """
                RNN specific arguments for DARTS
            """
            parser.add_argument("--data_path", default="~/data/ptb", type=str, help="Path to text dataset")
            args = parser.parse_args()

            raise NotImplementedError
            # run_experiment(args)

        # TODO: Add Tune checkpointing and integrate with this for visualizing
        elif sys.argv[2] == 'viz':
            from nas.darts.viz import viz_arch
            sys.argv.remove(sys.argv[2])
            sys.argv.remove(sys.argv[1])
            """
                Arguments for visualizing searched architectures
            """
            parser.add_argument("--load", default=None, type=str, help="Path to dir of a specific tune experiment")
            parser.add_argument("--save", default=None, type=str, help="Path to dir for saving the pngs of the model graph to. If unset defaults to load_dir")
            parser.add_argument("--viz", default=False, action="store_true", help="Open up vizualize pngs or not")
            args = parser.parse_args()
            viz_arch(args.load, args.save, viz=args.viz)

        else:
            print(usage_string)
            print(f"{sys.argv[2]} is not a valid option.")
            exit(-1)

    elif sys.argv[1] == 'enas':

        """
            General arguments for ENAS
        """
        parser.add_argument("--num_blocks", default=12, type=int, help="Number of layers in model")

        if sys.argv[2] == 'cnn':
            raise NotImplementedError
            from nas.cnn_enas import run_experiment
            sys.argv.remove(sys.argv[2])
            sys.argv.remove(sys.argv[1])
            dataset_options = ["cifar10", "mnist", "imagenet"]
            """
                CNN specific arguments for ENAS
            """
            parser.add_argument("--dataset", default="cifar10", choices=dataset_options, type=str.lower, help="Name of dataset")
            args = parser.parse_args()
            
            run_experiment(args)

        elif sys.argv[2] == 'rnn':
            from nas.rnn_enas import run_experiment
            sys.argv.remove(sys.argv[2])
            sys.argv.remove(sys.argv[1])
            dataset_options = ["ptb", "wikitext"]
            """
                RNN specific arguments for ENAS
            """
            parser.add_argument("--data_path", default="~/data/ptb", type=str, help="Path to text dataset")
            args = parser.parse_args()

            run_experiment(args)

        elif sys.argv[2] == 'viz':
            raise NotImplementedError
            from nas.enas.viz import viz_arch
            sys.argv.remove(sys.argv[2])
            sys.argv.remove(sys.argv[1])
            """
                Arguments for visualizing searched architectures
            """
            parser.add_argument("--load", default=None, type=str, help="Path to dir of a specific tune experiment")
            parser.add_argument("--save", default=None, type=str, help="Path to dir for saving the pngs of the model graph to. If unset defaults to load_dir")
            parser.add_argument("--viz", default=False, action="store_true", help="Open up vizualize pngs or not")
            args = parser.parse_args()
            viz_arch(args.load, args.save, viz=args.viz)

        else:
            print(usage_string)
            print(f"{sys.argv[2]} is not a valid option.")
            exit(-1)
    
    else:
        print(usage_string)
        print(f"{sys.argv[1]} is not a valid option.")
        exit(-1)

