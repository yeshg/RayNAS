import sys
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural Architecture Search with Ray")
    """
        General arguments for NAS
    """
    parser.add_argument("--smoke-test", default=False, action="store_true", help="Finish quickly for testing")
    parser.add_argument("--layers", default=20, type=int, help="Number of layers in model")
    parser.add_argument("--ray-address", default=None, help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enables GPU training")

    if len(sys.argv) < 2:
        print("Usage: python main.py [option]")
    
    elif sys.argv[1] == 'cnn':
        from nas.cnn_nas import run_experiment
        sys.argv.remove(sys.argv[1])
        dataset_options = ["cifar10", "mnist", "imagenet"]
        """
            CNN specific arguments for NAS
        """
        parser.add_argument("--dataset", default="cifar10", choices=dataset_options, type=str.lower, help="Name of dataset")
        args = parser.parse_args()
        run_experiment(args)

    elif sys.argv[1] == 'rnn':
        sys.argv.remove(sys.argv[1])
        dataset_options = ["ptb", "wikitext"]
        """
            RNN specific arguments for NAS
        """
        parser.add_argument("--dataset", default="ptb", choices=dataset_options, type=str.lower, help="Name of dataset")
        raise NotImplementedError

    # TODO: Add Tune checkpointing and integrate with this for visualizing
    elif sys.argv[1] == 'viz':
        from nas.viz import viz_arch
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
        print("Usage: python main.py [option]")
        print(f"{sys.argv[1]} is not a valid option.")
        exit(-1)

