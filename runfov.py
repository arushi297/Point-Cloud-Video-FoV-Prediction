# num_layers = 4
# hidden_size = 64
# teacher_forcing_ratio = 0.6
# tf_ratio_decrement = 0.1
# tf decrement on batch_loss < 0.05
# learning_rate = 0.001
# n_epochs = 600
# batch_size = 32
# 'teacher_forcing'
# target_length = 120 (2 sec)

import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import glob
import numpy as np
import pandas as pd
import os
import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):

        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size, num_layers):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)


    def train_model(self, input_tensor, target_tensor, val_input_tensor, val_target_tensor, n_epochs, target_len, batch_size, num_layers, training_prediction, teacher_forcing_ratio, learning_rate, dynamic_tf = False):

        '''
        train lstm encoder-decoder

        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''

        # initialize array of losses
        self.to(device)
        losses = np.full(n_epochs, np.nan)
        losses_sec1 = np.full(n_epochs, np.nan)
        losses_sec2 = np.full(n_epochs, np.nan)
        val_losses = np.full(n_epochs, np.nan)
        val_losses_sec1 = np.full(n_epochs, np.nan)
        val_losses_sec2 = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size) # (600, 2528, 9), input_tensor.shape[1] = 2528 / batch_size = 64

        val_n_batches = int(val_input_tensor.shape[1] / batch_size)
        
        min_val_loss = float('inf')
        
#         directory = str(batch_size)+"_"+str(hidden_size)+"_"+str(num_layers)+"_"+str(learning_rate)+str(training_prediction)+str(teacher_forcing_ratio)+"/"
#         if not os.path.exists(directory):
#             os.makedirs(directory)
        
        with trange(n_epochs) as tr:
            for it in tr:
                
                train_indices = np.arange(input_tensor.shape[1])
                np.random.shuffle(train_indices)
                
                model.train()
                
                batch_loss = 0.
                batch_loss_sec1 = 0.
                batch_loss_sec2 = 0.

                for b in range(n_batches):
                    # select data
                    batch_indices = train_indices[b: b + batch_size]
                    input_batch = input_tensor[:, batch_indices, :]
                    target_batch = target_tensor[:, batch_indices, :]

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2]).to(device)

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output

                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]

                            # predict recursively
                            else:
                                decoder_input = decoder_output

                    # compute the loss
                    loss = criterion(outputs, target_batch)
                    loss_sec1 = criterion(outputs[0:60,:,:], target_batch[0:60,:,:])
                    loss_sec2 = criterion(outputs[60:120,:,:], target_batch[60:120,:,:])
                    batch_loss += loss.item()
                    batch_loss_sec1 += loss_sec1.item()
                    batch_loss_sec2 += loss_sec2.item()

                    print(f"Iteration {it + 1}, Batch {b + 1}, Train Loss: {loss.item():.6f}")

                    # backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    plot_samples = 5
                    plt.figure(figsize=(15, 5))
                    
                    train_save_directory = curdir+'/Plot_Results/train/'+str(it+1)+"/"+str(b+1)+"/"
                    os.makedirs(train_save_directory, exist_ok=True)

                    for i in range(plot_samples):
                        feature = random.randint(0, 2)
                        input_data = input_batch[:, i, feature]
                        target_data = target_batch[:, i, feature]
                        output_data = outputs[:, i, feature]

                        plt.plot(np.arange(len(input_data)), input_data.detach().cpu().numpy(), label='History')
                        plt.plot(np.arange(len(input_data), len(input_data)+len(target_data)), target_data.detach().cpu().numpy(), label='Target')
                        plt.plot(np.arange(len(input_data), len(input_data)+len(target_data)), output_data.detach().cpu().numpy(), label='Output', linestyle='dashed')
                        plt.title(f'Train Sample {i + 1}, feature {feature}')
                        plt.xlabel('Time Step')
                        plt.ylabel('Value')
                        plt.legend()
                        
                        save_path = os.path.join(train_save_directory, f'plot_{i + 1}.png')
                        plt.savefig(save_path)
                        plt.clf()
                        
                    plt.close()

                # loss for epoch
                batch_loss /= n_batches
                batch_loss_sec1 /= n_batches
                batch_loss_sec2 /= n_batches
                losses[it] = batch_loss
                losses_sec1[it] = batch_loss_sec1
                losses_sec2[it] = batch_loss_sec2
                
                model.eval()
                val_loss = 0.
                val_loss_sec1 = 0.
                val_loss_sec2 = 0.
                
                for b in range(val_n_batches):
                    val_input_batch = val_input_tensor[:, b: b + batch_size, :]
                    val_target_batch = val_target_tensor[:, b: b + batch_size, :]
                    
                    val_outputs = torch.zeros(target_len, batch_size, val_input_batch.shape[2]).to(device)

                    with torch.no_grad():
                        # encoder outputs
                        encoder_output, encoder_hidden = self.encoder(val_input_batch)

                        # decoder without teacher forcing during validation
                        decoder_input = val_input_batch[-1, :, :]
                        decoder_hidden = encoder_hidden
                        
                        if training_prediction == 'recursive':
                            # predict recursively
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                val_outputs[t] = decoder_output
                                decoder_input = decoder_output

                        if training_prediction == 'teacher_forcing':
                            # use teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                for t in range(target_len):
                                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                    val_outputs[t] = decoder_output
                                    decoder_input = target_batch[t, :, :]

                            # predict recursively
                            else:
                                for t in range(target_len):
                                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                    val_outputs[t] = decoder_output
                                    decoder_input = decoder_output

                        if training_prediction == 'mixed_teacher_forcing':
                            # predict using mixed teacher forcing
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                val_outputs[t] = decoder_output

                                # predict with teacher forcing
                                if random.random() < teacher_forcing_ratio:
                                    decoder_input = target_batch[t, :, :]

                                # predict recursively
                                else:
                                    decoder_input = decoder_output

                    # compute the loss during validation
                    loss = criterion(val_outputs, val_target_batch)
                    loss_sec1 = criterion(val_outputs[0:60,:,:], val_target_batch[0:60,:,:])
                    loss_sec2 = criterion(val_outputs[60:120,:,:], val_target_batch[60:120,:,:])
                    val_loss += loss.item()
                    val_loss_sec1 += loss_sec1.item()
                    val_loss_sec2 += loss_sec2.item()
                    
                    print(f"Iteration {it + 1}, Batch {b + 1}, Val Loss: {loss.item():.6f}")
                    
                    val_save_directory = curdir+'/Plot_Results/val/'+str(it+1)+"/"+str(b+1)+"/"
                    os.makedirs(val_save_directory, exist_ok=True)
                    
                    plot_samples = 5
                    plt.figure(figsize=(15, 5))

                    for i in range(plot_samples):
                        feature = random.randint(0, 2)
                        index = random.randint(0, batch_size-1)
                        val_input_data = val_input_batch[:, index, feature]
                        val_target_data = val_target_batch[:, index, feature]
                        val_output_data = val_outputs[:, index, feature]
                        
                        plt.plot(np.arange(len(val_input_data)), val_input_data.cpu().numpy(), label='History')
                        plt.plot(np.arange(len(val_input_data), len(val_input_data)+len(val_target_data)), val_target_data.cpu().numpy(), label='Target')
                        plt.plot(np.arange(len(val_input_data), len(val_input_data)+len(val_target_data)), val_output_data.cpu().numpy(), label='Output', linestyle='dashed')
                        plt.title(f'Validation Sample {i + 1}, feature {feature}')
                        plt.xlabel('Time Step')
                        plt.ylabel('Value')
                        plt.legend()
                        
                        save_path = os.path.join(val_save_directory, f'plot_{i + 1}.png')
                        plt.savefig(save_path)
                        plt.clf()
                        
                    plt.close()
                    
                val_loss /= val_n_batches
                val_loss_sec1 /= val_n_batches
                val_loss_sec2 /= val_n_batches
                val_losses[it] = val_loss
                val_losses_sec1[it] = val_loss_sec1
                val_losses_sec2[it] = val_loss_sec2

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(self.state_dict(), 'best_model.pt') 
                
                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02
                    
                if teacher_forcing_ratio>0 and batch_loss < 0.05:
                    teacher_forcing_ratio = max(teacher_forcing_ratio - 0.2,0)
                    print(f"Teacher Forcing Ratio reduced to {teacher_forcing_ratio}")   
                    
                # progress bar
                tr.set_postfix(loss="{0:.6f}".format(batch_loss), val_loss="{0:.6f}".format(val_loss))
                
                
                if (it+1)%5 == 0:
                    loss_save_directory = curdir+'/Plot_Results/loss/'+str(it+1)+"/"
                    os.makedirs(loss_save_directory, exist_ok=True)
                    save_path = os.path.join(loss_save_directory, f'loss_curve.png')
                    epochs = range(1, len(losses) + 1)
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
                    # Plotting training loss on the first subplot
                    ax1.plot(epochs, losses, 'b-', label='Training Loss')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    # Plotting validation loss on the second subplot
                    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss')
                    ax2.set_xlabel('Epochs')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    # Adding a title to the entire figure
                    plt.suptitle('Training and Validation Loss')
                    plt.savefig(save_path)
                    plt.close()

                    loss_sec1_save_directory = curdir+'/Plot_Results/loss_sec1/'+str(it+1)+"/"
                    os.makedirs(loss_sec1_save_directory, exist_ok=True)
                    save_path = os.path.join(loss_sec1_save_directory, f'loss_curve.png')
                    epochs = range(1, len(losses_sec1) + 1)
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
                    # Plotting training loss on the first subplot
                    ax1.plot(epochs, losses_sec1, 'b-', label='Training Loss')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    # Plotting validation loss on the second subplot
                    ax2.plot(epochs, val_losses_sec1, 'r-', label='Validation Loss')
                    ax2.set_xlabel('Epochs')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    plt.suptitle('Training and Validation Loss for 1st sec')
                    plt.savefig(save_path)
                    plt.close()

                    loss_sec2_save_directory = curdir+'/Plot_Results/loss_sec2/'+str(it+1)+"/"
                    os.makedirs(loss_sec2_save_directory, exist_ok=True)
                    save_path = os.path.join(loss_sec2_save_directory, f'loss_curve.png')
                    epochs = range(1, len(losses_sec2) + 1)
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
                    # Plotting training loss on the first subplot
                    ax1.plot(epochs, losses_sec2, 'b-', label='Training Loss')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    # Plotting validation loss on the second subplot
                    ax2.plot(epochs, val_losses_sec2, 'r-', label='Validation Loss')
                    ax2.set_xlabel('Epochs')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    plt.suptitle('Training and Validation Loss for 2nd sec')
                    plt.savefig(save_path)
                    plt.close()


        return losses, val_losses, losses_sec1, val_losses_sec1, losses_sec2, val_losses_sec2

    def predict(self, input_tensor, target_len):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''

        self.to(device)
        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1).to(device)    # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2]).to(device)

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output

        np_outputs = outputs.detach().cpu().numpy()

        return np_outputs


x_train = np.load('/scratch/aa10350/FoV/NumpyData/x_train_new.npy')
y_train = np.load('/scratch/aa10350/FoV/NumpyData/y_train_new.npy')
x_test = np.load('/scratch/aa10350/FoV/NumpyData/x_test_new.npy')
y_test = np.load('/scratch/aa10350/FoV/NumpyData/y_test_new.npy')
x_val = np.load('/scratch/aa10350/FoV/NumpyData/x_val_new.npy')
y_val= np.load('/scratch/aa10350/FoV/NumpyData/y_val_new.npy')

x_train = x_train[:, :, :3]
y_train = y_train[:, :, :3]
x_val = x_val[:, :, :3]
y_val = y_val[:, :, :3]
x_test = x_test[:, :, :3]
y_test = y_test[:, :, :3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
curdir = '/vast/aa10350/FoV_run/exp1'

input_train_tensor = torch.from_numpy(x_train).float()  # Convert NumPy array to PyTorch tensor, x_train: (2528, 600, 9)
target_train_tensor = torch.from_numpy(y_train).float()
input_train_tensor = input_train_tensor.permute(1, 0, 2).to(device) # input_train_tensor: (600, 2528, 9)
target_train_tensor = target_train_tensor.permute(1, 0, 2).to(device)

input_val_tensor = torch.from_numpy(x_val).float()  # Convert NumPy array to PyTorch tensor, x_train: (2528, 600, 9)
target_val_tensor = torch.from_numpy(y_val).float()
input_val_tensor = input_val_tensor.permute(1, 0, 2).to(device) # input_train_tensor: (600, 2528, 9)
target_val_tensor = target_val_tensor.permute(1, 0, 2).to(device)

num_layers = 4
hidden_size = 64

model = lstm_seq2seq(input_size = input_train_tensor.shape[2], hidden_size = hidden_size, num_layers=num_layers).to(device) # input_train_tensor.shape[2] = 9

losses, val_losses, losses_sec1, val_losses_sec1, losses_sec2, val_losses_sec2 = model.train_model(
        input_tensor=input_train_tensor,
        target_tensor=target_train_tensor,
        val_input_tensor=input_val_tensor,
        val_target_tensor=target_val_tensor,
        n_epochs=600,
        target_len=120,
        batch_size=32,
        num_layers=num_layers,
        training_prediction='teacher_forcing',
        teacher_forcing_ratio=0.6,
        learning_rate=0.001,
        dynamic_tf=False
    )

loss_save_directory = curdir+'/Plot_Results/loss/final/'
os.makedirs(loss_save_directory, exist_ok=True)
save_path = os.path.join(loss_save_directory, f'loss_curve.png')
epochs = range(1, len(losses) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(epochs, losses, 'b-', label='Training Loss')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs, val_losses, 'r-', label='Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.suptitle('Training and Validation Loss')
plt.savefig(save_path)
plt.close()

loss_sec1_save_directory = curdir+'/Plot_Results/loss_sec1/final/'
os.makedirs(loss_sec1_save_directory, exist_ok=True)
save_path = os.path.join(loss_sec1_save_directory, f'loss_curve.png')
epochs = range(1, len(losses_sec1) + 1)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(epochs, losses_sec1, 'b-', label='Training Loss')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs, val_losses_sec1, 'r-', label='Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.suptitle('Training and Validation Loss for 1st sec')
plt.savefig(save_path)
plt.close()

loss_sec2_save_directory = curdir+'/Plot_Results/loss_sec2/final/'
os.makedirs(loss_sec2_save_directory, exist_ok=True)
save_path = os.path.join(loss_sec2_save_directory, f'loss_curve.png')
epochs = range(1, len(losses_sec2) + 1)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(epochs, losses_sec2, 'b-', label='Training Loss')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs, val_losses_sec2, 'r-', label='Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.suptitle('Training and Validation Loss')
plt.savefig(save_path)
plt.close()


best_model = lstm_seq2seq(input_size = input_train_tensor.shape[2], hidden_size = hidden_size, num_layers=num_layers).to(device)
best_model.load_state_dict(torch.load('best_model.pt'))
best_model.eval()

input_test_tensors = torch.from_numpy(x_test).float().to(device)  # Convert NumPy array to PyTorch tensor
target_test_tensors = torch.from_numpy(y_test).float().to(device)
target_len = 120
output_test_tensor = torch.zeros(len(input_test_tensors), target_len, target_test_tensors.shape[2]).to(device)

for i, input_tensor in enumerate(input_test_tensors):
    np_outputs = best_model.predict(input_tensor, target_len)
    output_test_tensor[i] = torch.from_numpy(np_outputs).to(device)
    
input_test_tensors = input_test_tensors.cpu()
target_test_tensors = target_test_tensors.cpu()
output_test_tensor = output_test_tensor.cpu()

plt.figure(figsize=(10, 6))

eval_save_directory = curdir+'/Plot_Results/eval/'
os.makedirs(eval_save_directory, exist_ok=True)

for feature_index in range(target_test_tensors.shape[2]):

    # Loop over each sequence
    for sequence_index in range(target_test_tensors.shape[0]):

        save_path = os.path.join(eval_save_directory, f'{feature_index}_{sequence_index}.png')
        input_sequence = input_test_tensors[sequence_index, :, feature_index].numpy()
        target_sequence = target_test_tensors[sequence_index, :, feature_index].numpy()
        predicted_sequence = output_test_tensor[sequence_index, :, feature_index].numpy()

        plt.plot(np.arange(len(input_sequence)), input_sequence, label=f'Sequence {sequence_index + 1} - History')
        plt.plot(np.arange(len(input_sequence), len(input_sequence)+len(target_sequence)), target_sequence, label=f'Sequence {sequence_index + 1} - Target')
        plt.plot(np.arange(len(input_sequence), len(input_sequence)+len(target_sequence)), predicted_sequence, label=f'Sequence {sequence_index + 1} - Predicted', linestyle='dashed')

        plt.title(f'Feature {feature_index + 1} Sequence {sequence_index + 1} - Target vs Predicted')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(save_path)
        plt.close()