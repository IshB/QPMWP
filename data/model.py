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
    def __init__(self, df, window_size, target_column='price', split_ratio=0.8, normalize=True, 
                 lookahead_days=1, filter_valid=True):
        """
        Initialize dataset from a DataFrame that already contains technical indicators
        
        Args:
            df: Pandas DataFrame with date, price, and technical indicators
            window_size: Number of days to look back for prediction
            target_column: Column to predict (default: 'price')
            split_ratio: Train/test split ratio
            normalize: Whether to normalize data
            lookahead_days: How many days ahead to predict returns
            filter_valid: Whether to filter for rows marked valid_for_model
        """
        # Store original DataFrame
        self.original_df = df.copy()
        
        # Convert date to datetime format if not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Filter for valid rows if needed
        if filter_valid and 'valid_for_model' in df.columns:
            df = df[df['valid_for_model'] == True].reset_index(drop=True)
        
        # Store sequence length for future reference
        self.sequence_length = window_size
        
        # Calculate returns (target) - for predicting future returns
        df['target_return'] = df[target_column].pct_change(periods=lookahead_days).shift(-lookahead_days)
        
        # Drop rows with NaN targets (will be at the end due to lookahead)
        df = df.dropna(subset=['target_return']).reset_index(drop=True)
        
        # Use all numeric columns as features (except 'id' and target)
        exclude_cols = ['date', 'id', 'target_return', 'valid_for_model']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Extract feature and target data
        target_data = df['target_return'].values
        driving_data = df[feature_columns].values
        
        # Calculate split indices
        self.train_samples = int(split_ratio * (len(target_data) - window_size - 1))
        self.test_samples = len(target_data) - window_size - 1 - self.train_samples
        
        # Normalization
        self.feature_means = None
        self.feature_stds = None
        self.mean_value = np.mean(target_data)
        self.std_value = np.std(target_data)
        
        if normalize:
            # Normalize target
            target_data = (target_data - self.mean_value) / self.std_value
            
            # Normalize features
            self.feature_means = np.mean(driving_data, axis=0)
            self.feature_stds = np.std(driving_data, axis=0)
            # Avoid division by zero
            self.feature_stds = np.where(self.feature_stds == 0, 1, self.feature_stds)
            driving_data = (driving_data - self.feature_means) / self.feature_stds
        
        # Generate time series sequences
        self.features, self.targets, self.target_sequences = self.generate_time_series(
            driving_data, target_data, window_size
        )
        
        # Store original dates for later reference
        self.dates = df['date'].values[window_size:]

    def get_sample_sizes(self):
        return self.train_samples, self.test_samples

    def get_feature_count(self):
        return self.features.shape[2]  # [samples, window_size, features]

    def get_training_data(self):
        return self.features[:self.train_samples], self.targets[:self.train_samples], self.target_sequences[:self.train_samples]

    def get_testing_data(self):
        return self.features[self.train_samples:], self.targets[self.train_samples:], self.target_sequences[self.train_samples:]
    
    def get_dates(self):
        """Return dates corresponding to the training and testing data"""
        train_dates = self.dates[:self.train_samples]
        test_dates = self.dates[self.train_samples:self.train_samples+self.test_samples]
        return train_dates, test_dates

    def generate_time_series(self, driving_data, target_data, window_size):
        feature_list, target_list, target_sequence_list = [], [], []
        
        for i in range(len(target_data) - window_size - 1):
            end_idx = i + window_size
            
            # For features, use all columns in the window
            feature_window = driving_data[i:end_idx]
            feature_list.append(feature_window)
            
            # Target is the return after the window
            target_list.append(target_data[end_idx])
            
            # Target sequence is the target values in the window
            target_sequence_list.append(target_data[i:end_idx])
            
        return np.array(feature_list), np.array(target_list), np.array(target_sequence_list)

    def denormalize_predictions(self, predictions):
        """Convert normalized predictions back to actual returns"""
        if hasattr(self, 'std_value') and self.std_value is not None:
            return predictions * self.std_value + self.mean_value
        return predictions


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


class ExpectedReturnsTrainer:
    def __init__(self, data_df, sequence_length, split_fraction, learning_rate, dropout_rate,
                 encoder_hidden_dim, decoder_hidden_dim, lookahead_days=1, target_column='price'):
        """
        Initialize the trainer with a DataFrame that already contains technical indicators
        
        Args:
            data_df: Pandas DataFrame with date, price, and technical indicators
            sequence_length: Number of days to look back for prediction
            split_fraction: Train/test split ratio
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for neural network
            encoder_hidden_dim: Hidden dimension for encoder
            decoder_hidden_dim: Hidden dimension for decoder
            lookahead_days: How many days ahead to predict returns
            target_column: Column to predict (default: 'price')
        """
        self.dataset = FinancialDataset(
            data_df, sequence_length, target_column, split_fraction, 
            normalize=True, lookahead_days=lookahead_days
        )
        
        self.encoder = AttentionEncoder(
            input_dim=self.dataset.get_feature_count(), 
            hidden_dim=encoder_hidden_dim,
            sequence_length=sequence_length, 
            dropout_rate=dropout_rate
        )
        
        self.decoder = AttentionDecoder(
            encoded_hidden_dim=encoder_hidden_dim, 
            hidden_dim=decoder_hidden_dim,
            sequence_length=sequence_length, 
            dropout_rate=dropout_rate
        )
        
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), learning_rate, weight_decay=1e-5)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), learning_rate, weight_decay=1e-5)

        self.encoder_scheduler = ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', factor=0.5, patience=2,
            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-08
        )
        
        self.decoder_scheduler = ReduceLROnPlateau(
            self.decoder_optimizer, mode='min', factor=0.5, patience=2,
            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-08
        )

        self.loss_function = nn.MSELoss()
        self.train_samples, self.test_samples = self.dataset.get_sample_sizes()

    def train_model(self, num_epochs, batch_size, save_interval):
        """Train the model for predicting expected returns"""
        x_train, y_train, y_seq_train = self.dataset.get_training_data()
        
        # Track losses for plotting
        train_losses = []
        
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
                
                loss_sum += loss.item() * (batch_end - i)  # Weight by batch size
                i = batch_end
                
            avg_loss = loss_sum / self.train_samples
            train_losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}] ==> loss: {avg_loss:.6f}')

            self.encoder_scheduler.step(avg_loss)
            self.decoder_scheduler.step(avg_loss)

            if (epoch + 1) % save_interval == 0 or epoch + 1 == num_epochs:
                self.save_model(epoch + 1)
                
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()

    def test_model(self, batch_size, is_plot=True):
        """Test the model and evaluate performance"""
        x_train, y_train, y_seq_train = self.dataset.get_training_data()
        x_test, y_test, y_seq_test = self.dataset.get_testing_data()
        
        # Get dates for plotting
        train_dates, test_dates = self.dataset.get_dates()
        
        # Predict on train and test sets
        y_pred_train = self.predict(x_train, y_train, y_seq_train, batch_size)
        y_pred_test = self.predict(x_test, y_test, y_seq_test, batch_size)
        
        # Denormalize predictions and actual values
        y_pred_train_denorm = self.dataset.denormalize_predictions(y_pred_train)
        y_train_denorm = self.dataset.denormalize_predictions(y_train)
        y_pred_test_denorm = self.dataset.denormalize_predictions(y_pred_test)
        y_test_denorm = self.dataset.denormalize_predictions(y_test)
        
        # Calculate metrics for testing set
        test_mse = np.mean((y_pred_test_denorm - y_test_denorm) ** 2)
        test_rmse = np.sqrt(test_mse)
        test_mae = np.mean(np.abs(y_pred_test_denorm - y_test_denorm))
        
        # Calculate directional accuracy (how often the sign is correct)
        test_dir_acc = np.mean((np.sign(y_pred_test_denorm) == np.sign(y_test_denorm)).astype(float))
        
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test Directional Accuracy: {test_dir_acc:.4f}")
        
        if is_plot:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot returns
            ax1.set_title('Expected Returns Prediction')
            ax1.plot(train_dates, y_train_denorm, label='Train Actual', color='green', alpha=0.5)
            ax1.plot(train_dates, y_pred_train_denorm, label='Train Predicted', color='red', alpha=0.5)
            ax1.plot(test_dates, y_test_denorm, label='Test Actual', color='blue')
            ax1.plot(test_dates, y_pred_test_denorm, label='Test Predicted', color='purple')
            ax1.set_ylabel('Return')
            ax1.legend()
            ax1.grid(True)
            
            # Plot cumulative returns
            train_cum_actual = np.cumprod(1 + y_train_denorm) - 1
            train_cum_pred = np.cumprod(1 + y_pred_train_denorm) - 1
            test_cum_actual = np.cumprod(1 + y_test_denorm) - 1
            test_cum_pred = np.cumprod(1 + y_pred_test_denorm) - 1
            
            ax2.set_title('Cumulative Returns')
            ax2.plot(train_dates, train_cum_actual, label='Train Actual', color='green', alpha=0.5)
            ax2.plot(train_dates, train_cum_pred, label='Train Predicted', color='red', alpha=0.5)
            ax2.plot(test_dates, test_cum_actual, label='Test Actual', color='blue')
            ax2.plot(test_dates, test_cum_pred, label='Test Predicted', color='purple')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Return')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('return_prediction_results.png')
            plt.show()
            
            # Create a histogram of prediction errors
            plt.figure(figsize=(10, 6))
            plt.hist(y_pred_test_denorm - y_test_denorm, bins=50, alpha=0.75)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig('error_distribution.png')
            plt.show()
            
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'Date': test_dates,
            'Actual_Return': y_test_denorm,
            'Predicted_Return': y_pred_test_denorm,
            'Error': y_pred_test_denorm - y_test_denorm
        })
        results_df.to_csv('return_prediction_results.csv', index=False)
        print("Results saved to return_prediction_results.csv")
            
        return (y_pred_test_denorm, y_test_denorm, 
                y_pred_train_denorm, y_train_denorm)

    def predict(self, x_data, y_data, y_seq_data, batch_size):
        """Make predictions with the model"""
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
        """Load saved model weights"""
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def save_model(self, epoch):
        """Save model weights"""
        if not os.path.exists('models'):
            os.makedirs('models')
        encoder_path = f'models/encoder_epoch_{epoch}.model'
        decoder_path = f'models/decoder_epoch_{epoch}.model'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        print(f"Models saved to {encoder_path} and {decoder_path}")

    def to_variable(self, x):
        """Convert numpy array to PyTorch Variable"""
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())

# Function to run the full prediction workflow
def predict_expected_returns(csv_path, lookahead_days=1):
    """
    Run the full prediction workflow on a CSV file
    
    Args:
        csv_path: Path to CSV file with date, price, and technical indicators
        lookahead_days: Number of days ahead to predict returns
    """
    # Load the CSV data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Hyperparameters
    num_epochs = 30
    batch_size = 32
    split_fraction = 0.8
    save_interval = 5
    learning_rate = 0.001
    sequence_length = 10  # Number of days to look back
    encoder_hidden_dim = 64
    decoder_hidden_dim = 64
    dropout_rate = 0.2
    
    print("Initializing model...")
    trainer = ExpectedReturnsTrainer(
        df,
        sequence_length, 
        split_fraction,
        learning_rate, 

        dropout_rate, 
        encoder_hidden_dim, 
        decoder_hidden_dim,
        lookahead_days=lookahead_days
    )
    
    print(f"Training with {trainer.train_samples} samples...")
    trainer.train_model(num_epochs, batch_size, save_interval)
    
    print("Testing model...")
    trainer.test_model(batch_size, is_plot=True)
    
    return trainer

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_file_path = "data/test_v2.csv"
    trainer = predict_expected_returns(csv_file_path, lookahead_days=1)