import os
import pickle
import nas.darts.cnn.visualize as cnn_visualize
import nas.darts.rnn.visualize as rnn_visualize


def load_arch(load_dir):
    params = pickle.load(open(os.path.join(load_dir, "params.pkl"), "rb"))
    genotype = pickle.load(open(os.path.join(load_dir, "genotype.pkl"), "rb"))
    return "cnn", genotype

def viz_arch(load_dir, save_dir=None, viz=False):

    if save_dir is None:
        save_dir = os.path.dirname(load_dir)
    
    print(f"\nSaving to {save_dir}/\n")

    net_type, genotype = load_arch(load_dir)

    if net_type == "cnn":
        cnn_visualize.plot(genotype.normal, os.path.join(save_dir, "normal"), view=viz)
        cnn_visualize.plot(genotype.reduce, os.path.join(save_dir, "reduction"), view=viz)
    elif net_type == "rnn":
        rnn_visualize.plot(genotype.normal, os.path.join(save_dir, "normal"), view=viz)
        rnn_visualize.plot(genotype.reduce, os.path.join(save_dir, "reduction"), view=viz)
