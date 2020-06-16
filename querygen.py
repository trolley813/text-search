from textgenrnn import textgenrnn
import os


def train():
    data_path = "./data/Gutenberg/txt"
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    file_texts = [open(os.path.join(data_path, f), encoding="utf-8", errors="replace").read() for f in files[::20]]
    large_text = "\n".join(file_texts)
    textgen = textgenrnn()
    textgen.train_new_model([large_text], single_text=True, num_epochs=2, word_level=True, max_length=5)
    textgen.save("./tgrnn/weights.hdf5")


if __name__ == "__main__":
    # train()
    textgen = textgenrnn(weights_path="textgenrnn_weights.hdf5",
                         vocab_path="textgenrnn_vocab.json",
                         config_path="textgenrnn_config.json")
    textgen.load("./tgrnn/weights.hdf5")
    texts = textgen.generate(return_as_list=True, max_gen_length=8, temperature=0.9, n=20)
    print(texts)