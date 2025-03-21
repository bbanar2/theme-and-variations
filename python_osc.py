"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math
import random

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client

import pretty_midi
import os

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

current_directory = os.getcwd()

def generate(note_embedding_dim,
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
         attr_plot,
         z_values):

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
    original_midi_file_path = os.getcwd() + input_file_name
    s = music21.converter.parse(original_midi_file_path)
   
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

    sam1_score_original.write('midi', os.getcwd() + '/MIDI_files/input_midi.mid')

    z = z_tilde[0]
    z[0] = z_values[0]
    z[1] = z_values[1]
    z[2] = z_values[2]
    z[3] = z_values[3]

    z = z.unsqueeze(0)

    dummy_score_tensor = to_cuda_variable(torch.zeros(batch_size_inference, measure_seq_len))
    _, sam1 = model.decoder(z, dummy_score_tensor, train)

    sam1_score = folk_dataset_train.get_score_from_tensor(sam1.cpu())

    midi_file_name = 'output_midi.mid'

    sam1_score.write('midi', os.getcwd() + '/MIDI_files/' + midi_file_name)

    print('Generated!')
  
  input_file_name = '/MIDI_files/input_midi.mid'
  generate_file(input_file_name)

def normalize_given_metric(slider_type, value):
  if slider_type == 0:
    normalized_value = (value * 12) - 5.4

  elif slider_type == 1:
    normalized_value = (value * 11.5) - 6.1

  elif slider_type == 2:
    normalized_value = (value * 6) - 2.8

  else:
    normalized_value = (value * 14) - 4
  return normalized_value
  

def osc_to_midi_save_and_generate(*args):

  first_slider_type = args[0][1]
  first_slider_level = args[0][2]

  second_slider_type = args[0][3]
  second_slider_level = args[0][4]

  third_slider_type = args[0][5]
  third_slider_level = args[0][6]

  fourth_slider_type = args[0][7]
  fourth_slider_level = args[0][8]

  slider_types = [first_slider_type, second_slider_type, third_slider_type, fourth_slider_type]
  slider_values = [first_slider_level, second_slider_level, third_slider_level, fourth_slider_level]

  z_values = [0,0,0,0]

  for slider in slider_types:
    z_values[slider] = normalize_given_metric(slider, slider_values[slider])

  number_of_notes = args[0][10]

  osc_format_note_offset = 12

  note_numbers = []
  times = []
  durations = []
  velocities = []

  for note_index in range(number_of_notes):
    note_numbers.append(args[0][osc_format_note_offset + (note_index * 6)])
    times.append(args[0][osc_format_note_offset + (note_index * 6) + 1])
    durations.append(args[0][osc_format_note_offset + (note_index * 6) + 2])
    velocities.append(args[0][osc_format_note_offset + (note_index * 6) + 3])

  pm = pretty_midi.PrettyMIDI()
  
  pm_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
  pm_instrument = pretty_midi.Instrument(program=pm_program)

  for note_index in range(number_of_notes):
    note = pretty_midi.Note(velocity=velocities[note_index], pitch=note_numbers[note_index], start=times[note_index] / 2, end=(times[note_index] / 2) + (durations[note_index] / 2))
    pm_instrument.notes.append(note)

  pm.instruments.append(pm_instrument)
  pm.write(current_directory + '/MIDI_files/input_midi.mid')

  return z_values


def print_volume_handler(unused_addr, *args):
  z_values = osc_to_midi_save_and_generate(args)
  generate(note_embedding_dim = 10,
            metadata_embedding_dim = 2,
            num_encoder_layers = 2,
            encoder_hidden_size = 512,
            encoder_dropout_prob = 0.5,
            has_metadata = False,
            latent_space_dim = 256,
            num_decoder_layers = 2,
            decoder_hidden_size = 512,
            decoder_dropout_prob = 0.5,
            batch_size = 256,
            num_epochs = 30,
            train = False,
            plot = True,
            log = True,
            reg_loss = True,
            reg_type = 'four_metrics',
            reg_dim = 0,
            attr_plot = True,
            z_values = z_values)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=8001, help="The port to listen on")
  args = parser.parse_args()

  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/notes", print_volume_handler, "Volume")

  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()