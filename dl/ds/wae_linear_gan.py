import torch
from torch import nn
from dl.ds.linear_network import LinearNetwork
from tqdm import tqdm_notebook as tqdm


class WAE_Linear_GAN():

    def __init__(self, input_dim, latent_dim, encoder_dims, decoder_dims, discriminator_dims, lambda_reg):
        super(WAE_Linear_GAN, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg

        # Encoder
        self.encoder = LinearNetwork(encoder_dims, last_layer_activation=None)

        # Decoder
        self.decoder = LinearNetwork(decoder_dims, last_layer_activation=None)

        # Discriminator
        self.discriminator = LinearNetwork(discriminator_dims, last_layer_activation='sigmoid')

    def __str__(self):
        return f'Encoder:\n{self.encoder}\nDecoder:\n{self.decoder}\nDiscriminator:\n{self.discriminator}\n'

    def __repr__(self):
        return self.__str__()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def discriminate(self, z):
        return self.discriminator(z)

    def sample_embeddings(self, n_samples=100):
        return torch.randn(n_samples, self.latent_dim, dtype=torch.float)

    def sample(self, n_samples=100):
        return self.decode(self.sample_embeddings(n_samples=n_samples))

    def loss_discriminator_prior(self, d_z_samples):
        temp = torch.log(d_z_samples)
        return (-1) * self.lambda_reg * torch.mean(temp)

    def loss_discriminator_encoder(self, d_z_encodings):
        temp = torch.log(1 - d_z_encodings)
        return (-1) * self.lambda_reg * torch.mean(temp)

    def loss_reconstruction(self, x, x_decoded):
        return nn.MSELoss()(x, x_decoded)

    def loss_regularization(self, d_z_encodings):
        temp = torch.log(d_z_encodings)
        return (-1) * self.lambda_reg * torch.mean(temp)

    def generator_loss(self, x, x_decoded, d_z_encodings):
        return self.loss_reconstruction(x, x_decoded) + self.loss_regularization(d_z_encodings)

    def discriminator_loss(self, d_z_encodings, d_z_samples):
        return self.loss_discriminator_prior(d_z_samples) + self.loss_discriminator_encoder(d_z_encodings)

    def zero_grads(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

    def train_step(self, x):
        # ======== Train Discriminator ======== #

        # Zero the grads
        self.zero_grads()

        # Unfreeze only discriminator
        self.discriminator.unfreeze_params()
        self.encoder.freeze_params()
        self.decoder.freeze_params()

        # Embeddings
        z_encodings = self.encode(x)

        # Sampling
        z_samples = self.sample_embeddings(n_samples=len(x))

        # Discriminator output
        d_z_encodings = self.discriminate(z_encodings)
        d_z_samples = self.discriminate(z_samples)

        # Loss
        loss_disc_prior = self.loss_discriminator_prior(d_z_samples)
        loss_disc_embedding = self.loss_discriminator_encoder(d_z_encodings)
        disc_loss = loss_disc_prior + loss_disc_embedding

        # Take the optimization step
        disc_loss.backward()
        self.discriminator.optimization_step()

        # ======== Train Generator ======== #

        # Zero the grads
        self.zero_grads()

        # Freeze only discriminator
        self.discriminator.freeze_params()
        self.encoder.unfreeze_params()
        self.decoder.unfreeze_params()

        # Embeddings
        z_encodings = self.encode(x)
        x_decoded = self.decode(z_encodings)

        # Discriminator output
        d_z_encodings = self.discriminate(z_encodings)

        # Loss
        recon_loss = self.loss_reconstruction(x, x_decoded)
        regularization_loss = self.loss_regularization(d_z_encodings)
        gen_loss = recon_loss + regularization_loss

        # Take the optimization step
        gen_loss.backward()
        self.encoder.optimization_step()
        self.decoder.optimization_step()

        return {'loss_disc_prior': loss_disc_prior, 'loss_disc_embedding': loss_disc_embedding,
                'recon_loss': recon_loss, 'regularization_loss': regularization_loss}

    def train(self, x, n_epochs=10):
        loss_disc_prior = []
        loss_disc_embedding = []
        recon_loss = []
        regularization_loss = []
        for epoch in tqdm(range(n_epochs)):
            loss_dict = self.train_step(x)
            loss_disc_prior.append(loss_dict['loss_disc_prior'])
            loss_disc_embedding.append(loss_dict['loss_disc_embedding'])
            recon_loss.append(loss_dict['recon_loss'])
            regularization_loss.append(loss_dict['regularization_loss'])

        return {'loss_disc_prior': loss_disc_prior, 'loss_disc_embedding': loss_disc_embedding,
                'recon_loss': recon_loss, 'regularization_loss': regularization_loss}
