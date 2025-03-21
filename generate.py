import click

from MeasureVAE.measure_vae import MeasureVAE
from MeasureVAE.vae_tester import VAETester
from data.dataloaders.bar_dataset import *
from utils.helpers import *
import torch
import tqdm

import os, shutil, heapq

from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import gaussian_kde
import seaborn as sns

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

@click.command()
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_encoder_layers', default=2,
              help='number of layers in encoder RNN')
@click.option('--encoder_hidden_size', default=512,
              help='hidden size of the encoder RNN')
@click.option('--encoder_dropout_prob', default=0.5,
              help='float, amount of dropout prob between encoder RNN layers')
@click.option('--has_metadata', default=False,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=256,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=512,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=30, # 30 EPOCHS
              help='number of training epochs')
@click.option('--train/--test', default=False, # TRAIN
              help='train or test the specified model')
@click.option('--plot/--no_plot', default=True,
              help='plot the training log')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--reg_loss/--no_reg_loss', default=True, # YES REG LOSS
              help='train with regularization loss')
@click.option('--reg_type', default='four_metrics', # REG TYPE FOUR METRICS
              help='attribute name string to be used for regularization')
@click.option('--reg_dim', default=0, # REG DIMS, overwritten in vae_trainer
              help='dimension along with regularization is to be carried out')
@click.option('--attr_plot/--no_attr_plot', default=True,
              help='if True plots the attribute dsitributions, else produces interpolations')
def main(note_embedding_dim,
         metadata_embedding_dim,
         num_encoder_layers,
         encoder_hidden_size,
         encoder_dropout_prob,
         latent_space_dim,
         num_decoder_layers,
         decoder_hidden_size,
         decoder_dropout_prob,
         has_metadata,
         batch_size,
         num_epochs,
         train,
         plot,
         log,
         reg_loss,
         reg_type,
         reg_dim,
         attr_plot
         ):

    is_short = False
    num_bars = 1
    folk_dataset_train = FolkNBarDataset(
        dataset_type='train',
        is_short=is_short,
        num_bars=num_bars)
    folk_dataset_test = FolkNBarDataset(
        dataset_type='test',
        is_short=is_short,
        num_bars=num_bars
    )

    model = MeasureVAE(
        dataset=folk_dataset_train,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_dropout_prob=encoder_dropout_prob,
        latent_space_dim=latent_space_dim,
        num_decoder_layers=num_decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_dropout_prob=decoder_dropout_prob,
        has_metadata=has_metadata
    )
    
    model.load(cpu = True)
    # model.cuda()
    model.eval()

    folk_dataset_train.data_loaders(
        batch_size=batch_size,
        split=(0.70, 0.20)
    )

    (generator_train,
        generator_val,
        generator_test) = folk_dataset_train.data_loaders(
        batch_size=batch_size,
        split=(0.70, 0.20)
    )

    # ************************************************ Generate MIDI Files *****************************************************************
    def generate_file(input_file_name):
        midis_path = os.getcwd() + '/generated_midi_files/'

        if not os.path.exists(midis_path):
            os.makedirs(midis_path)


        original_midi_file_path = os.getcwd() + input_file_name
        # original_abc_file_path = original_midi_file_path[:-4] + '.abc'
        s = music21.converter.parse(original_midi_file_path)
        # s.write('midi', fp = os.getcwd() + '/midi_files_metrics_final_withLSR_ver2/input_midi_COOL_INPUT_RECON.mid')

        
        # original_abc_score = get_music21_score_from_path(original_abc_file_path)
        # original_abc_score_tensor = folk_dataset_train.get_tensor(original_abc_score)
        original_abc_score_tensor = folk_dataset_train.get_tensor(s)
        z_dist = model.encoder(original_abc_score_tensor)
        # sample from distribution
        z_tilde = z_dist.rsample()

        z_original = z_tilde[0]

        measure_seq_len = 24 # hard coded, taken from vae_tester
        train = False
        batch_size_inference = 1 # hard coded, taken from vae_tester

        # midi save original source midi

        z_original = z_original.unsqueeze(0)

        dummy_score_tensor = to_cuda_variable(torch.zeros(batch_size_inference, measure_seq_len))
        _, sam1_original = model.decoder(z_original, dummy_score_tensor, train)

        sam1_score_original = folk_dataset_train.get_score_from_tensor(sam1_original.cpu())

        sam1_score_original.write('midi', os.getcwd() + '/generated_midi_files/input_midi.mid')

        midi_file_counter = 0
        num_of_samples = 2
        for rhy_complx_index in range(num_of_samples):
            for pitch_range_index in range(num_of_samples):
                for note_density_index in range(num_of_samples):
                    for avg_int_jump_index in range(num_of_samples):

                        midi_file_counter += 1

                        print('Processing: ' + str(midi_file_counter))

                        z = z_tilde[0]
                        z[0] = 0.2# 0
                        z[1] = 0.2# 1 or 2
                        z[2] = 0.2
                        z[3] = 0.2

                        z = z.unsqueeze(0)

                        dummy_score_tensor = to_cuda_variable(torch.zeros(batch_size_inference, measure_seq_len))
                        _, sam1 = model.decoder(z, dummy_score_tensor, train)

                        sam1_score = folk_dataset_train.get_score_from_tensor(sam1.cpu())

                        midi_file_name = 'midi_' + str(rhy_complx_index + 1) + '_' + str(pitch_range_index + 1) + '_' + str(note_density_index + 1) + '_' + str(avg_int_jump_index + 1) + '.mid'

                        sam1_score.write('midi', os.getcwd() + '/generated_midi_files/' + midi_file_name)

    input_file_name = '/input_midi.mid'
    generate_file(input_file_name)

if __name__ == '__main__':
    main()