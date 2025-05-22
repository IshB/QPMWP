import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FinancialDataset:
    def __init__(self, csv_path, window_size, target_column='price', split_ratio=0.8, normalize=False):
        # Read the CSV file
        self.original_file_path = csv_path
        data = pd.read_csv(csv_path)
        
        # Convert date to datetime format
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data = data.sort_values('date')
        
        # Store sequence length for future reference
        self.sequence_length = window_size
        
        # Use price as target
        target_data = data[target_column].fillna(method='pad')
        
        # Use all numeric columns as features (except 'id')
        feature_columns = [col for col in data.columns if col not in ['date', 'id', target_column]]
        driving_data = data[feature_columns].fillna(method='pad')
        
        self.train_samples = int(split_ratio * (target_data.shape[0] - window_size - 1))
        self.test_samples = target_data.shape[0] - window_size - 1 - self.train_samples
        
        self.mean_value = target_data.mean()
        self.std_value = target_data.std()
        if normalize:
            target_data = (target_data - self.mean_value) / self.std_value
            # Normalize features as well
            self.feature_means = driving_data.mean()
            self.feature_stds = driving_data.std()
            driving_data = (driving_data - self.feature_means) / self.feature_stds.replace(0, 1)
        
        self.features, self.targets, self.target_sequences = self.generate_time_series(driving_data, target_data, window_size)

    def get_sample_sizes(self):
        return self.train_samples, self.test_samples

    def get_feature_count(self):
        return self.features.shape[2]  # Now features are 3D: [samples, window_size, features]

    def get_training_data(self):
        return self.features[:self.train_samples], self.targets[:self.train_samples], self.target_sequences[:self.train_samples]

    def get_testing_data(self):
        return self.features[self.train_samples:], self.targets[self.train_samples:], self.target_sequences[self.train_samples:]

    def generate_time_series(self, driving_data, target_data, window_size):
        feature_list, target_list, target_sequence_list = [], [], []
        
        for i in range(len(target_data) - window_size - 1):
            end_idx = i + window_size
            
            # For features, use all columns in the window
            feature_window = driving_data.iloc[i:end_idx].values
            feature_list.append(feature_window)
            
            # Target is the next price after the window
            target_list.append(target_data.iloc[end_idx])
            
            # Target sequence is the price values in the window
            target_sequence_list.append(target_data.iloc[i:end_idx].values)
            
        return np.array(feature_list), np.array(target_list), np.array(target_sequence_list)


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length, dropout_rate):
        super(AttentionEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1)
        self.attn_layer1 = nn.Linear(in_features=2 * hidden_dim, out_features=sequence_length)
        self.attn_layer2 = nn.Linear(in_features=sequence_length, out_features=sequence_length)
        self.tanh_activation = nn.Tanh()
        self.attn_layer3 = nn.Linear(in_features=sequence_length, out_features=1)

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, driving_input):
        batch_size = driving_input.size(0)
        encoded_output = self.initialize_variable(batch_size, self.sequence_length, self.hidden_dim)
        hidden_state = self.initialize_variable(1, batch_size, self.hidden_dim)
        cell_state = self.initialize_variable(1, batch_size, self.hidden_dim)
        
        for t in range(self.sequence_length):
            combined_hidden = torch.cat((self.repeat_hidden(hidden_state), self.repeat_hidden(cell_state)), 2)
            attn_scores1 = self.attn_layer1(combined_hidden)
            
            # Handle the case where driving_input has 3 dimensions [batch, seq, features]
            if driving_input.dim() == 3:
                attn_scores2 = self.attn_layer2(driving_input.permute(0, 2, 1))
                attn_combined = attn_scores1 + attn_scores2
                attn_weights = self.attn_layer3(self.tanh_activation(attn_combined))
                
                if batch_size > 1:
                    attention_weights = F.softmax(attn_weights.view(batch_size, self.input_dim), dim=1)
                else:
                    attention_weights = self.initialize_variable(batch_size, self.input_dim) + 1
                    
                weighted_input = torch.mul(attention_weights, driving_input[:, t, :])
            else:
                # Legacy support for 2D input
                weighted_input = driving_input[:, t].unsqueeze(1)

            weighted_input = self.dropout_layer(weighted_input)
            _, (hidden_state, cell_state) = self.lstm_layer(weighted_input.unsqueeze(0), (hidden_state, cell_state))
            encoded_output[:, t, :] = hidden_state

        return encoded_output

    def initialize_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def repeat_hidden(self, hidden_tensor):
        return hidden_tensor.repeat(self.input_dim, 1, 1).permute(1, 0, 2)


class AttentionDecoder(nn.Module):
    def __init__(self, encoded_hidden_dim, hidden_dim, sequence_length, dropout_rate):
        super(AttentionDecoder, self).__init__()
        self.encoded_hidden_dim = encoded_hidden_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        self.attn_layer1 = nn.Linear(in_features=2 * hidden_dim, out_features=encoded_hidden_dim)
        self.attn_layer2 = nn.Linear(in_features=encoded_hidden_dim, out_features=encoded_hidden_dim)
        self.tanh_activation = nn.Tanh()
        self.attn_layer3 = nn.Linear(in_features=encoded_hidden_dim, out_features=1)
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=self.hidden_dim)
        self.concat_layer = nn.Linear(in_features=self.encoded_hidden_dim + 1, out_features=1)
        self.fc_layer1 = nn.Linear(in_features=encoded_hidden_dim + hidden_dim, out_features=hidden_dim)
        self.fc_layer2 = nn.Linear(in_features=hidden_dim, out_features=1)

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, encoded_hidden, target_sequence):
        batch_size = encoded_hidden.size(0)
        decoder_hidden = self.initialize_variable(1, batch_size, self.hidden_dim)
        cell_state = self.initialize_variable(1, batch_size, self.hidden_dim)
        context_vector = self.initialize_variable(batch_size, self.hidden_dim)

        for t in range(self.sequence_length):
            combined_hidden = torch.cat((self.repeat_hidden(decoder_hidden), self.repeat_hidden(cell_state)), 2)
            attn_scores1 = self.attn_layer1(combined_hidden)
            attn_scores2 = self.attn_layer2(encoded_hidden)
            attn_combined = attn_scores1 + attn_scores2
            attn_weights = self.attn_layer3(self.tanh_activation(attn_combined))
            
            if batch_size > 1:
                beta_t = F.softmax(attn_weights.view(batch_size, -1), dim=1)
            else:
                beta_t = self.initialize_variable(batch_size, self.encoded_hidden_dim) + 1
                
            context_vector = torch.bmm(beta_t.unsqueeze(1), encoded_hidden).squeeze(1)
            if t < self.sequence_length - 1:
                # Handle both 1D and 2D target_sequence
                if target_sequence.dim() == 1:
                    target_t = target_sequence[t].unsqueeze(0)
                else:
                    target_t = target_sequence[:, t].unsqueeze(1)
                
                concat_input = torch.cat((target_t, context_vector), dim=1)
                y_tilde = self.concat_layer(concat_input)

                y_tilde = self.dropout_layer(y_tilde)
                _, (decoder_hidden, cell_state) = self.lstm_layer(y_tilde.unsqueeze(0), (decoder_hidden, cell_state))
                
        output = self.fc_layer2(
            self.dropout_layer(self.fc_layer1(torch.cat((decoder_hidden.squeeze(0), context_vector), dim=1))))

        return output

    def initialize_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def repeat_hidden(self, hidden_tensor):
        return hidden_tensor.repeat(self.sequence_length, 1, 1).permute(1, 0, 2)


class FinancialModelTrainer:
    def __init__(self, data_file, sequence_length, split_fraction, learning_rate, dropout_rate,
                 encoder_hidden_dim, decoder_hidden_dim, target_column='price'):
        self.dataset = FinancialDataset(data_file, sequence_length, target_column, split_fraction, normalize=True)
        self.encoder = AttentionEncoder(input_dim=self.dataset.get_feature_count(), hidden_dim=encoder_hidden_dim,
                                        sequence_length=sequence_length, dropout_rate=dropout_rate)
        self.decoder = AttentionDecoder(encoded_hidden_dim=encoder_hidden_dim, hidden_dim=decoder_hidden_dim,
                                        sequence_length=sequence_length, dropout_rate=dropout_rate)
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), learning_rate, weight_decay=1e-5)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), learning_rate, weight_decay=1e-5)

        self.encoder_scheduler = ReduceLROnPlateau(self.encoder_optimizer, mode='min', factor=0.5, patience=1,
                                                   verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                   min_lr=0, eps=1e-08)
        self.decoder_scheduler = ReduceLROnPlateau(self.decoder_optimizer, mode='min', factor=0.5, patience=1,
                                                   verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                   min_lr=0, eps=1e-08)

        self.loss_function = nn.MSELoss()
        self.train_samples, self.test_samples = self.dataset.get_sample_sizes()

    def train_model(self, num_epochs, batch_size, save_interval):
        x_train, y_train, y_seq_train = self.dataset.get_training_data()
        for epoch in range(num_epochs):
            i = 0
            loss_sum = 0
            while i < self.train_samples:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                batch_end = min(i + batch_size, self.train_samples)
                var_x = self.to_variable(x_train[i:batch_end])
                var_y = self.to_variable(y_train[i:batch_end])
                var_y_seq = self.to_variable(y_seq_train[i:batch_end])
                
                encoded_sequence = self.encoder(var_x)
                y_pred = self.decoder(encoded_sequence, var_y_seq)
                loss = self.loss_function(y_pred, var_y.unsqueeze(1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                
                loss_sum += loss.item()
                i = batch_end
                
            avg_loss = loss_sum / self.train_samples
            print(f'Epoch [{epoch+1}/{num_epochs}] ==> loss: {avg_loss:.6f}')

            self.encoder_scheduler.step(loss_sum)
            self.decoder_scheduler.step(loss_sum)

            if (epoch + 1) % save_interval == 0 or epoch + 1 == num_epochs:
                self.save_model(epoch + 1)

    def test_model(self, batch_size, is_plot=True):
        x_train, y_train, y_seq_train = self.dataset.get_training_data()
        x_test, y_test, y_seq_test = self.dataset.get_testing_data()
        
        y_pred_train = self.predict(x_train, y_train, y_seq_train, batch_size)
        y_pred_test = self.predict(x_test, y_test, y_seq_test, batch_size)
        
        # Calculate metrics for testing set
        test_mse = np.mean((y_pred_test - y_test) ** 2)
        test_rmse = np.sqrt(test_mse)
        test_mae = np.mean(np.abs(y_pred_test - y_test))
        
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        if is_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(range(self.train_samples), y_train * self.dataset.std_value + self.dataset.mean_value, label='Train Actual', color='green')
            plt.plot(range(self.train_samples), y_pred_train * self.dataset.std_value + self.dataset.mean_value, label='Train Predicted', color='red')
            plt.plot(range(self.train_samples, self.train_samples + self.test_samples), 
                    y_test * self.dataset.std_value + self.dataset.mean_value, label='Test Actual', color='blue')
            plt.plot(range(self.train_samples, self.train_samples + self.test_samples), 
                    y_pred_test * self.dataset.std_value + self.dataset.mean_value, label='Test Predicted', color='purple')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.title('Financial Price Prediction')
            plt.legend()
            plt.grid(True)
            plt.savefig('prediction_plot.png')
            plt.show()
            
        return (y_pred_test * self.dataset.std_value + self.dataset.mean_value, y_test * self.dataset.std_value + self.dataset.mean_value, 
                y_pred_train * self.dataset.std_value + self.dataset.mean_value, y_train * self.dataset.std_value + self.dataset.mean_value)

    def predict(self, x_data, y_data, y_seq_data, batch_size):
        y_pred = np.zeros(x_data.shape[0])
        i = 0
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            while i < x_data.shape[0]:
                batch_end = min(i + batch_size, x_data.shape[0])
                var_x_input = self.to_variable(x_data[i:batch_end])
                var_y_input = self.to_variable(y_seq_data[i:batch_end])
                
                encoded_sequence = self.encoder(var_x_input)
                y_res = self.decoder(encoded_sequence, var_y_input)
                
                y_pred[i:batch_end] = y_res.cpu().numpy().flatten()
                i = batch_end
                
        self.encoder.train()
        self.decoder.train()
        return y_pred

    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def save_model(self, epoch):
        if not os.path.exists('models'):
            os.makedirs('models')
        encoder_path = f'models/encoder_epoch_{epoch}.model'
        decoder_path = f'models/decoder_epoch_{epoch}.model'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        print(f"Models saved to {encoder_path} and {decoder_path}")

    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())

# Example usage
def run_prediction(csv_path, target_column='price'):
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    split_fraction = 0.85
    save_interval = 5
    learning_rate = 0.0005
    sequence_length = 5
    encoder_hidden_dim = 64
    decoder_hidden_dim = 64
    dropout_rate = 0
    
    print("Initializing model...")
    trainer = FinancialModelTrainer(
        csv_path, 
        sequence_length, 
        split_fraction,
        learning_rate, 
        dropout_rate, 
        encoder_hidden_dim, 
        decoder_hidden_dim,
        target_column=target_column
    )
    
    print(f"Training with {trainer.train_samples} samples...")
    trainer.train_model(num_epochs, batch_size, save_interval)
    
    print("Testing model...")
    y_pred_test, y_test, y_pred_train, y_train = trainer.test_model(batch_size, is_plot=True)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'Test_Actual': y_test,
        'Test_Predicted': y_pred_test,
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("Results saved to prediction_results.csv")
    
    return trainer

