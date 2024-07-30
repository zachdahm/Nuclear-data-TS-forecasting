import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import torch  # Library for implementing Deep Neural Network
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from numpy import array
import time
import ray

from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def get_data(filename):
    #Function that reads a dataset from a .csv file and returns a dataframe
    df = pd.read_csv(filename)
    df = df.loc[:,['index','nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']]
    df.fillna(0, inplace=True)
    return df


def train_test_split(df):
    #Function that splits the larger normal dataset into training, testing, and validation
    train_len = int(len(df) * 0.6)
    val_len = int(len(df) * 0.2)
    test_len = len(df) - train_len - val_len
    train_data = df[0:train_len]
    val_data = df[train_len:(train_len + val_len)]
    test_data = df[(train_len + val_len):(train_len + val_len + test_len)]
    return train_data, val_data, test_data


def standardize_data(train_inputs, val_inputs, test_inputs):
    #Function that fits a scaler to the training data and then applies it to the validation and testing data
    # Note that this function is not used in this version of the code because only the normalized data has been uploaded
    scaler = MinMaxScaler()
    train_inputs.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']] = scaler.fit_transform(train_inputs.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']])
    val_inputs.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']] = scaler.transform(val_inputs.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']])
    test_inputs.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']] = scaler.transform(test_inputs.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']])
    return train_inputs, val_inputs, test_inputs, scaler


def split_sequences(sequences, n_steps, batch_size, shuffle, future):
    #This function splits the 2D matrix (features, time) into a 3D matrix (index of time window, features, time). Inputs to the model is a window of inputs so this function
    #transforms the dataset into a list of these input time windows
    X, y = [], []
    sequences = array(sequences)
    for i in range(len(sequences) - n_steps - future):
        check = 1
        section = sequences[i:i+n_steps+future+1]
        for j in range(len(section)-1):
            if int(section[j, 0]) == int((section[j+1,0] - 1)):
                pass
            else:
                check = 0
        if check == 1:
            X.append(sequences[i:i + n_steps])
            y.append(sequences[i + n_steps + future, 1])
    X, y = np.array(X), np.array(y)
    X = X[:,:, 1:]
    X = torch.tensor(X, dtype=torch.float32)
    X = X.squeeze()
    y = torch.tensor(y, dtype=torch.float32)
    y = y.squeeze()
    y = y[:, None]
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, y


class LSTMModel(nn.Module):
    #Class for the LSTM Network
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers, n_steps):
        super(LSTMModel, self).__init__()  # initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.flatten = nn.Flatten(end_dim=2, start_dim=1)
        self.linear = nn.Linear((hidden_size * n_steps), 1)

    def forward(self, x):  # defines forward pass of the neural network
        x, _ = self.lstm(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out

class ANNModel(nn.Module):
    #Class for the ANN Network
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers, n_steps):
        super(ANNModel, self).__init__()  # initializes the parent class nn.Module
        self.input = nn.Flatten()
        self.hidden1 = nn.LazyLinear(hidden_size)
        self.hidden2 = nn.LazyLinear(hidden_size)
        self.output = nn.LazyLinear(1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):  # defines forward pass of the neural network
        x = self.input(x)
        x = nn.functional.relu(self.hidden1(x))
        for i in range(1, self.num_layers):
            x = nn.functional.relu(self.hidden2(x))
        out = (self.output(x))
        return out

class GRUModel(nn.Module):
    #Class for the GRU Network
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers, n_steps):
        super(GRUModel, self).__init__()  # initializes the parent class nn.Module
        self.hidden = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.flatten = nn.Flatten(end_dim=2, start_dim=1)
        self.output = nn.Linear((hidden_size * n_steps), 1)

    def forward(self, x):  # defines forward pass of the neural network
        x, _ = self.hidden(x)
        x = self.flatten(x)
        out = self.output(x)
        return out

class RNNModel(nn.Module):
    #Class for the RNN Network
    # input_size : number of features in input at each time step
    # hidden_size : Number of LSTM units
    # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers, n_steps):
        super(RNNModel, self).__init__()  # initializes the parent class nn.Module
        self.hidden = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.flatten = nn.Flatten(end_dim=2, start_dim=1)
        self.output = nn.Linear((hidden_size * n_steps), 1)

    def forward(self, x):  # defines forward pass of the neural network:
        x, _ = self.hidden(x)
        x = self.flatten(x)
        out = self.output(x)
        return out


def training_loop(config, future):
    #Training Loop
    t = time.time()
    #Note that the full path may be necessary for the filename if using Ray for hyperparameter tuning
    filename = 'real_dataset_normalized.csv'
    n_steps = config['n_steps']
    #Setting device to GPU if avaliable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = get_data(filename)
    train_data, val_data, test_data = train_test_split(df)
    #Data is already normalized for the upload, left as a comment if code if used for data which needs to be normalized
    #train_data, val_data, test_data, scaler = standardize_data(train_data, val_data, test_data)
    train_loader, y_train = split_sequences(train_data, n_steps, config['batch_size'], True, future)
    val_loader, y_val = split_sequences(val_data, n_steps, config['batch_size'], False, future)
    test_loader, y_test = split_sequences(test_data, n_steps, config['batch_size'], False, future)
    n_features = 8
    model = GRUModel(n_features, config['n_neurons'], config['n_layers'], config['n_steps']).to(device)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    num_epochs = config['n_epochs']
    train_hist = []
    test_hist = []
    test_losses = []
    train_losses = []
    train_rmses = []
    test_rmses = []
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_percent_loss_train = 0.0

        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            percent_loss = abs((predictions / batch_y) - 1) * 100

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_percent_loss_train += torch.mean(percent_loss)

        # Calculate average training loss and accuracy
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)
        average_rmse_loss_train = np.sqrt(average_loss)

        # Validation on Validation dataset
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0
            total_percent_loss = 0.0

            for batch_X_test, batch_y_test in val_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                percent_loss = abs((predictions_test / batch_y_test) - 1) * 100
                test_loss = loss_fn(predictions_test, batch_y_test)

                total_test_loss += test_loss.item()
                total_percent_loss += torch.mean(percent_loss)

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(val_loader)
            average_rmse = np.sqrt(average_test_loss)
            train.report({"loss": average_test_loss})
            test_hist.append(average_test_loss)
        if (epoch + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Validation Loss: {average_test_loss:.4f}, RMSE Train Loss: {average_rmse_loss_train: .4f}, RMSE Validation Loss: {average_rmse: .4f}')
        train_losses.append(float(average_loss))
        test_losses.append(float(average_test_loss))
        train_rmses.append(float(average_rmse_loss_train))
        test_rmses.append(float(average_rmse))
    runtime = time.time() - t
    #torch.save(model.state_dict(), 'GRU_Final.pt')
    print('Runtime:', runtime)
    training_history = pd.DataFrame(list(zip(train_losses, test_losses, train_rmses, test_rmses)), columns=['Train Loss', 'Test Loss', 'Train rmse', 'Test rmse'])
    return average_loss, average_rmse_loss_train,average_test_loss, average_rmse, model


def model_eval(config, filename):
    #Function for applying trained model to evaluation dataset
    model = GRUModel(8, 100, 1, 30)
    model.load_state_dict(torch.load('GRU_Final.pt'))
    #scaler = torch.load('forecast_journal_scaler.pt')
    future = 5
    df = get_data(filename)
    if filename == 'real_dataset_normalized.csv':
        df = df.tail(40000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_hist = []
    loss_fn = torch.nn.MSELoss(reduction='mean')
    #Commented out normalization because provided data is normalized
    #df.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']] = scaler.transform(df.loc[:,['nfd-1-cps', 'nfd-1-cr', 'rr-active-state', 'rr-position', 'ss1-active-state', 'ss1-position', 'ss2-active-state', 'ss2-position']])
    test_loader, y_test = split_sequences(df,config['n_steps'], 1, False, future)

    with torch.no_grad():
        total_test_loss = 0.0
        total_percent_loss = 0.0
        predictions = []
        true_values = []
        errors = []

        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            predictions_test = model(batch_X_test)
            predictions.append(predictions_test)
            true_values.append(batch_y_test)
            error = abs(batch_y_test - predictions_test)
            errors.append(error)
            percent_loss = abs(((predictions_test / batch_y_test) - 1) * 100)
            test_loss = loss_fn(predictions_test, batch_y_test)


            total_test_loss += test_loss.item()
            total_percent_loss += torch.mean(percent_loss)

        # Calculate average test loss and accuracy
        data_for_plot = pd.DataFrame(list(zip(predictions, true_values, errors)),
                                     columns=['Model Predictions', 'Actual Values','Error'])

        average_test_loss = total_test_loss / len(test_loader)
        average_rmse = np.sqrt(average_test_loss)
        train.report({"loss": average_test_loss})
        test_hist.append(average_test_loss)
        print(filename,f' Testing RMSE: {average_rmse: .6f}')
    return average_rmse, average_test_loss
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Set this variable to True to perform the random grid search hyperparameter tuning
    hyperparameter_tuning = False
    if hyperparameter_tuning:
        config = {'n_neurons': tune.choice([5, 10, 15, 20, 50, 100]),
                  'n_layers': tune.choice([1, 2, 3, 4]),
                  'lr': tune.choice([0.0001, 0.0005, 0.001, 0.005]),
                  'batch_size': tune.choice([4, 8, 16, 32, 64]),
                  'n_steps': tune.choice([2, 5, 10, 20, 30, 50]),
                  'n_epochs': tune.choice([3, 5, 10, 15, 20, 50])
                  }
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"])
        result = tune.run(
            partial(training_loop),
            config=config,
            scheduler=scheduler,
            progress_reporter=reporter, num_samples=1000, reuse_actors=False)

        best_trial = result.get_best_trial("loss", "min", "avg")
        print("Best trial config: {}".format(best_trial.config))
    else:
        config = {'n_neurons': 100,
                  'n_layers': 1,
                  'lr': 0.0001,
                  'batch_size': 4,
                  'n_steps': 30,
                  'n_epochs': 50}
    history_results = pd.DataFrame(columns=['Train Loss', 'Test Loss', 'Transient Test Loss', 'FDI 1 Test Loss', 'FDI 2 Test Loss', 'Scrams Test Loss',
                     'Train Percent Error', 'Test Percent Error', 'Transient Percent Error', 'FDI 1 Percent Error', 'FDI 2 Percent Error',
                     'Scrams Percent Error'])
    for future in range(5,6):
        #train_loss, train_percent, test_loss, average_percent_loss, model = training_loop(config, future)
        testing_loss_trans, percent_loss_trans = model_eval(config,'transient_normal.csv')
        testing_loss_fdi1, percent_loss_fdi1 = model_eval(config, 'FDI1_normal.csv')
        testing_loss_fdi2, percent_loss_fdi2 = model_eval(config, 'FDI2_normal.csv')
        testing_loss_scrams, percent_loss_scrams = model_eval(config, 'scrams_normal.csv')
        test_loss, percent_loss = model_eval(config, 'real_dataset_normalized.csv')

        current_results = pd.DataFrame(
            data=[[test_loss, testing_loss_trans, testing_loss_fdi1, testing_loss_fdi2, testing_loss_scrams]],
            columns=['Test Loss', 'Transient Test Loss', 'FDI 1 Test Loss', 'FDI 2 Test Loss', 'Scrams Test Loss',])
        history_results = pd.concat([history_results, current_results], axis=0)

main()
