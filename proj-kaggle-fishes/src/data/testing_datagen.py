def open_dump(path, textfile):
    return pickle.load(open(os.path.join(path, textfile), 'rb'))

alldescriptors = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'alldescriptors.txt')
galldescriptors = open_dump(os.path.join(INTERIM, 'train', 'generated'), 'alldescriptors.txt')

megaalldescriptors = np.concatenate((np.array(alldescriptors), np.array(galldescriptors)))

k = 128
km = KMeans(k)
concatenated = np.concatenate(megaalldescriptors)
print('Number of descriptors: {}'.format(
        len(concatenated)))
concatenated = concatenated[::64]
print('Clustering with K-means...')
km.fit(concatenated)

basedir = '/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train'
msfeatures = []
for d in megaalldescriptors:
    c = km.predict(d)
    msfeatures.append(np.bincount(c, minlength=k))

msfeatures = np.array(msfeatures, dtype=float)
try:
    with open(op.join(basedir, 'msfeatures.txt'), 'wb') as file:
        pickle.dump(msfeatures, file)
except:
    print("Sfeatures dump improperly done")


ifeatures = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'ifeatures.txt')
gifeatures = open_dump(os.path.join(INTERIM, 'train', 'generated'), 'ifeatures.txt')

mfeatures = np.concatenate((np.array(ifeatures), np.array(gifeatures)))
try:
    with open(op.join(basedir, 'mfeatures.txt'), 'wb') as file:
        pickle.dump(mfeatures, file)
except:
    print("Sfeatures dump improperly done")


labels = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'labels.txt')
glabels = open_dump(os.path.join(INTERIM, 'train', 'generated'), 'labels.txt')

mlabels = np.concatenate((np.array(labels), np.array(glabels)))
try:
    with open(op.join(basedir, 'mlabels.txt'), 'wb') as file:
        pickle.dump(mlabels, file)
except:
    print("Mlabels dump improperly done")

filenames = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'filenames.txt')
gfilenames = open_dump(os.path.join(INTERIM, 'train', 'generated'), 'filenames.txt')

mfilenames = np.concatenate((np.array(filenames), np.array(gfilenames)))
try:
    with open(op.join(basedir, 'mfilenames.txt'), 'wb') as file:
        pickle.dump(mfilenames, file)
except:
    print("Mfilenames dump improperly done")


labels = open_dump(INTERIM, 'labels.txt')
filenames = open_dump(INTERIM, 'filenames.txt')
