from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras import initializers
from tqdm import tqdm
import numpy as np
import json

from querygen import make_good_queries, get_idfs
from querysearch import querysearch

random_dim = 100


def prepare(data):
    max_len = 8
    rows = []
    for d in data:
        for e in d:
            idfs, doc_len, cts, bi_cts, tri_cts = e
            n = len(idfs)
            x_idfs = np.pad(idfs, (0, max_len - n)) / 20
            x_cts = np.pad(cts, (0, max_len - n)) / doc_len * 100
            x_bi_cts = np.pad(bi_cts, (0, max_len - n + 1)) / doc_len * 100
            x_tri_cts = np.pad(tri_cts, (0, max_len - n + 2)) / doc_len * 100
            assert len(x_idfs) == max_len
            assert len(x_cts) == max_len
            assert len(x_bi_cts) == max_len
            assert len(x_tri_cts) == max_len
            row = np.concatenate((x_idfs, x_cts, x_bi_cts, x_tri_cts))
            rows.append(row)
    results = np.array(rows)
    return results


def prepare_data():
    data = json.load(open("ref_relevant.json"))
    return prepare(data)


def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(2048))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(32, activation='tanh'))
    generator.compile(loss='mse', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(2048, input_dim=32, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1024))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator


def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the query is relevant or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan


def train(epochs=1, batch_size=128):
    # Get the training and testing data
    data = prepare_data()
    np.random.shuffle(data)
    x_train = data
    # Split the training data into batches of size batch_size (128 by default)
    batch_count = x_train.shape[0] // batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            query_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake queries
            generated_queries = generator.predict(noise)
            X = np.concatenate([query_batch, generated_queries])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
    return gan


def prepare_gan_model():
    gan = train(400, 128)
    gan.save("./gan.hdf5")


def filter_topN(query, n, m):
    threshold = (304, 1214)
    filtered_n = []
    filtered_m = []
    results = querysearch(query, threshold)
    idfs = get_idfs(query)
    topN = [[idfs] + list(v[2:6]) for v in results[:n]]
    bottomM = [[idfs] + list(v[2:6]) for v in results[-m:]]
    filtered_n += topN
    filtered_m += bottomM
    return (filtered_n, filtered_m)


def test():
    gan: Model = load_model("gan.hdf5")
    optimizer = get_optimizer()
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    discriminator = gan.get_layer(index=-1)
    input = Input(shape=(32,))
    output = discriminator(input)
    test_net = Model(inputs=input, outputs=output)
    test_queries = make_good_queries(1000)
    f = open("query_res.txt", "a")
    for query in test_queries:
        try:
            good, bad = filter_topN(query, 10, 10)
            good_inp = prepare([good])
            bad_inp = prepare([bad])
            good_labs = test_net.predict(good_inp).transpose()
            bad_labs = test_net.predict(bad_inp).transpose()
            print(query, good_labs, bad_labs, file=f, sep="\n")
            f.flush()
        except ValueError as ve:
            print(f"Skipping query: {str(ve)}")

if __name__ == "__main__":
    #prepare_gan_model()
    test()
