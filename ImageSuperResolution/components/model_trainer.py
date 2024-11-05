import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, LeakyReLU, BatchNormalization, Add, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define Generator Model
def build_generator():
    def residual_block(x):
        res = Conv2D(64, (3, 3), padding='same')(x)
        res = BatchNormalization(momentum=0.8)(res)
        res = LeakyReLU(alpha=0.2)(res)
        res = Conv2D(64, (3, 3), padding='same')(res)
        res = BatchNormalization(momentum=0.8)(res)
        return Add()([x, res])
    
    inputs = Input(shape=(64, 64, 3))  # Low-resolution input
    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    for _ in range(16):
        x = residual_block(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D(size=2)(x)
    x = UpSampling2D(size=2)(x)
    outputs = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)

    return Model(inputs, outputs)

# Define Discriminator Model
def build_discriminator():
    inputs = Input(shape=(256, 256, 3))  # High-resolution input
    x = Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    for i, filters in enumerate([64, 128, 256, 512]):
        x = Conv2D(filters, (3, 3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)

# VGG Feature Extractor (for Perceptual Loss)
def build_vgg():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    vgg.trainable = False
    return Model(vgg.input, vgg.layers[10].output)

# Compile the SRGAN model
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
vgg = build_vgg()
vgg.trainable = False

# Combine Generator and Discriminator into GAN
input_low_res = Input(shape=(64, 64, 3))
generated_high_res = generator(input_low_res)
discriminator.trainable = False
validity = discriminator(generated_high_res)
features = vgg(generated_high_res)

# GAN Model
gan = Model(input_low_res, [validity, features])
gan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Training Parameters
epochs = 10000
batch_size = 16

# Define paths and image sizes
high_res_dir = r'C:\Users\sphur\Downloads\DIV2K_train_HR'  # Path to high-res DIV2K images
low_res_dir = r'C:\Users\sphur\Downloads\High_Res' 
high_res_size = (256, 256)                  # Target size for high-res images
low_res_size = (64, 64)                     # Target size for low-res images

# Function to load images from a directory and convert them to arrays
def load_images_from_directory(directory, image_size):
    images = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=image_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Load and preprocess images
low_res_images = load_images_from_directory(low_res_dir, low_res_size)
high_res_images = load_images_from_directory(high_res_dir, high_res_size)

# Normalize images to [-1, 1] range
low_res_images = (low_res_images / 127.5) - 1
high_res_images = (high_res_images / 127.5) - 1

# Verify the shape of loaded images
print(f"Low-resolution images shape: {low_res_images.shape}")
print(f"High-resolution images shape: {high_res_images.shape}")


# Train SRGAN
def train_srgan(generator, discriminator, gan, epochs, batch_size, low_res_images, high_res_images, use_pretrained=False):
    if use_pretrained:
        generator.load_weights(r'C:\Users\sphur\OneDrive\Desktop\Image_SuperResolution\ImageSuperResolution\components\SRGAN.h5')  # Load generator weights if pretrained

    for epoch in range(epochs):
        # Select random batch of images
        idx = np.random.randint(0, low_res_images.shape[0], batch_size)
        imgs_lr, imgs_hr = low_res_images[idx], high_res_images[idx]

        # Generate high-resolution images
        fake_hr = generator.predict(imgs_lr)

        # Train Discriminator
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
        d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        image_features = vgg.predict(imgs_hr)
        g_loss = gan.train_on_batch(imgs_lr, [valid, image_features])

        # Print training progress
        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]}] [G loss: {g_loss[0]}]")

        # Save model checkpoints
        if epoch % 1000 == 0:
            generator.save_weights(f'generator_epoch_{epoch}.h5')

# Assuming low_res_images and high_res_images are prepared
# Corrected function call
train_srgan(generator, discriminator, gan, epochs=10000, batch_size=16, 
            low_res_images=low_res_images, high_res_images=high_res_images, 
            use_pretrained=False)