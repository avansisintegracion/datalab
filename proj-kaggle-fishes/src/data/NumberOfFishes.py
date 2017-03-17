from keras.preprocessing import image, sequence

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True,
        batch_size=4, class_mode='categorical', target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
        class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

path = '../../data/raw/'

print('Inside ' + path + 'train should be 3777')
batches = get_batches(path+'train', shuffle=False, batch_size=1)
print('Inside ' + path + 'test')
test_batches = get_batches(path+'test', shuffle=False, batch_size=1)

path = '../../data/interim/train/crop/'

print('Inside ' + path + 'train should be 3277')
batches = get_batches(path+'train', shuffle=False, batch_size=1)
print('Inside ' + path + 'val ')
val_batches = get_batches(path+'val', shuffle=False, batch_size=1)
print('Inside ' + path + 'test')
test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
