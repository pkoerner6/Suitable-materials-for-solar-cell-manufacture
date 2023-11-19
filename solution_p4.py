
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from torch.nn.init import xavier_normal_ 
from torch.utils.data import DataLoader
import structlog
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

log = structlog.get_logger()


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.
    input: None
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.linear1 = nn.Linear(1000, 1000)
        self.dropout1 = nn.Dropout(p=0.2) 
        self.activation1 = nn.ReLU() 
        self.linear2 = nn.Linear(1000, 1000)
        self.dropout2 = nn.Dropout(p=0.2) 
        self.activation2 = nn.ReLU() 
        self.linear3 = nn.Linear(1000, 900)
        self.dropout3 = nn.Dropout(p=0.2) 
        self.activation3 = nn.ReLU()
        self.linear4 = nn.Linear(900, 900)
        self.dropout4 = nn.Dropout(p=0.2) 
        self.activation4 = nn.ReLU()
        self.linear5 = nn.Linear(900, 800)
        self.dropout5 = nn.Dropout(p=0.2) 
        self.activation5 = nn.ReLU()
        self.linear6 = nn.Linear(800, 800)
        self.dropout6 = nn.Dropout(p=0.2) 
        self.activation6 = nn.ReLU()
        self.linear7 = nn.Linear(800, 700)
        self.dropout7 = nn.Dropout(p=0.2) 
        self.activation7 = nn.ReLU()
        self.linear8 = nn.Linear(700, 700)
        self.dropout8 = nn.Dropout(p=0.2)
        self.activation8 = nn.ReLU()
        self.linear9 = nn.Linear(700, 600)
        self.dropout9 = nn.Dropout(p=0.2)
        self.activation9 = nn.ReLU()
        self.linear10 = nn.Linear(600, 500)
        self.dropout10 = nn.Dropout(p=0.2)
        self.activation10 = nn.ReLU()
        self.linear11 = nn.Linear(500, 400)
        self.dropout11 = nn.Dropout(p=0.2)
        self.activation11 = nn.ReLU()
        self.linear12 = nn.Linear(400, 300)
        self.dropout12 = nn.Dropout(p=0.2)
        self.activation12 = nn.ReLU()
        self.linear13 = nn.Linear(300, 1)
        xavier_normal_(self.linear1.weight) 
        xavier_normal_(self.linear2.weight) 
        xavier_normal_(self.linear3.weight) 
        xavier_normal_(self.linear4.weight) 
        xavier_normal_(self.linear5.weight) 
        xavier_normal_(self.linear6.weight)
        xavier_normal_(self.linear7.weight)
        xavier_normal_(self.linear8.weight)
        xavier_normal_(self.linear9.weight)
        xavier_normal_(self.linear10.weight)
        xavier_normal_(self.linear11.weight)
        xavier_normal_(self.linear12.weight)

        xavier_normal_(self.linear13.weight)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.dropout3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.dropout4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        x = self.dropout5(x)
        x = self.activation5(x)
        x = self.linear6(x)
        x = self.dropout6(x)
        x = self.activation6(x)
        x = self.linear7(x)
        x = self.dropout7(x)
        x = self.activation7(x)
        x = self.linear8(x)
        x = self.dropout8(x)
        x = self.activation8(x)
        x = self.linear9(x)
        x = self.dropout9(x)
        x = self.activation9(x)
        x = self.linear10(x)
        x = self.dropout10(x)
        x = self.activation10(x)
        x = self.linear11(x)
        x = self.dropout11(x)
        x = self.activation11(x)
        x = self.linear12(x)
        x = self.dropout12(x)
        x = self.activation12(x)
        x = self.linear13(x)
        return x


class torch_dataset(torch.utils.data.Dataset):
  '''Prepare the pretrain dataset for regression'''
  def __init__(self, X, y):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


def make_feature_extractor(x, y, train_new, batch_size=250, eval_size=2000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.
    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    model.train() 
    
    learning_rate = 1e-3
    epochs = 40

    loss_function = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pretrain_dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
    trainloader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    pretrain_dataset_val = torch.utils.data.TensorDataset(x_val, y_val)
    valloader = DataLoader(pretrain_dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    def train_one_epoch(epoch_index, tb_writer):
        cur_loss = 0.0
        last_loss = 0.0 
        
        for indx, data in enumerate(trainloader): 
            x, y = data
            x, y = x.float(), y.float()
            optimizer.zero_grad() 
            y_pred = model(x) 
            loss = loss_function(y_pred.squeeze(), y) 
            loss.backward() 
            optimizer.step()  
            cur_loss += loss.item() 
            if indx % batch_size == ((50000-eval_size)/batch_size)-1:
                last_loss = cur_loss / batch_size 
                tb_x = epoch_index * len(trainloader) + indx + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                cur_loss = 0.
        return last_loss

    if train_new: 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        best_val_loss = 1000000

        for epoch in range(epochs):
            print(f'Starting epoch {epoch}') 
            model.train(True)
            avg_loss = train_one_epoch(epoch, writer)
            model.eval()
            current_loss_val = 0.0

            for indx, data in enumerate(valloader): 
                x, y = data
                x, y = x.float(), y.float()
                y_pred = model(x)  
                loss = loss_function(y_pred.squeeze(), y)  
                current_loss_val += loss.item()
            
            avg_vloss = current_loss_val / (indx + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
            writer.flush()
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
            model_path = 'model_{}'.format(epoch)
            torch.save(model.state_dict(), model_path)


        print('Training process has finished.')
    

    def make_features(x_train, model_path):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.
        input: x: np.ndarray, the features of the training or test set
        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model = Net()    
        model.load_state_dict(torch.load(model_path))

        x_train = torch.tensor(x_train, dtype=torch.float)

        twelveth_to_last_layer = nn.Sequential(*list(model.children())[:-34]) 
        twelveth_to_last_layer.eval()
        features12 = twelveth_to_last_layer(x_train)
        eleventh_to_last_layer = nn.Sequential(*list(model.children())[:-31]) 
        eleventh_to_last_layer.eval()
        features11 = eleventh_to_last_layer(x_train)
        tenth_to_last_layer = nn.Sequential(*list(model.children())[:-28]) 
        tenth_to_last_layer.eval()
        features10 = tenth_to_last_layer(x_train)
        nineth_to_last_layer = nn.Sequential(*list(model.children())[:-25]) 
        nineth_to_last_layer.eval()
        features9 = nineth_to_last_layer(x_train)
        eightth_to_last_layer = nn.Sequential(*list(model.children())[:-22]) 
        eightth_to_last_layer.eval()
        features8 = eightth_to_last_layer(x_train)
        seventh_to_last_layer = nn.Sequential(*list(model.children())[:-19]) 
        seventh_to_last_layer.eval()
        features7 = seventh_to_last_layer(x_train)
        sixth_to_last_layer = nn.Sequential(*list(model.children())[:-16]) 
        sixth_to_last_layer.eval()
        features6 = sixth_to_last_layer(x_train)
        fiveth_to_last_layer = nn.Sequential(*list(model.children())[:-13]) 
        fiveth_to_last_layer.eval()
        features5 = fiveth_to_last_layer(x_train)
        fourth_to_last_layer = nn.Sequential(*list(model.children())[:-10]) 
        fourth_to_last_layer.eval()
        features4 = fourth_to_last_layer(x_train)
        third_to_last_layer = nn.Sequential(*list(model.children())[:-7]) 
        third_to_last_layer.eval()
        features3 = third_to_last_layer(x_train)
        second_to_last_layer = nn.Sequential(*list(model.children())[:-4]) 
        second_to_last_layer.eval()
        features2 = second_to_last_layer(x_train)
        first_to_last_layer = nn.Sequential(*list(model.children())[:-1]) 
        first_to_last_layer.eval()
        features1 = first_to_last_layer(x_train)
        features = torch.cat((features12, features11, features10, features9, features8, features7, features6, features5, features4, features3, features1), 1) 
        features = features.detach().numpy()

        return features

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    input: feature_extractors: dict, a dictionary of feature extractors
    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X, model_path):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X, model_path)
            return X_new
        
    return PretrainedFeatures

def get_regression_model_svr(x_train, y_train):
    """
    This function returns the regression model used in the pipeline.
    input: None
    output: model: sklearn compatible model, the regression model
    """

    model = svm.SVR()
    search_spaces = {
        'kernel' : ['sigmoid'],
        'C' : [1000],
        'degree' : [2],
        'coef0' : [0.01],
        'gamma' : ['auto'],
        'tol': [1e-2],
        'epsilon': [0.01], 
        }
    grids = GridSearchCV(model, search_spaces, cv=5, scoring="neg_root_mean_squared_error", refit=True)
    grids.fit(x_train, y_train)
    
    best_score = (grids.best_score_)*(-1)
    best_params = grids.best_params_
    print("Best Score: ", best_score)
    print("Best Params: ", best_params)

    model = svm.SVR(**best_params)
    return model, best_score


if __name__ == '__main__':
    best_model_num = ""
    best_score = 1000

    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    # Utilize pretraining data by creating feature extractor which extracts lumo energy features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain, train_new=True)

    for model_num in ["7_best_11layers_0_219"]:
        x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()

        # Utilize pretraining data by creating feature extractor which extracts lumo energy features from available initial features
        feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain, train_new=False)
        PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})

        print(f"Current model is model_{model_num}")
        x_train = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_train, model_path=f"model_{model_num}")
        model, best_score_model = get_regression_model_svr(x_train, y_train)

        if best_score_model < best_score:
            best_score = best_score_model
            best_model_num = model_num
            log.info("New best score is: ", best_score=best_score)
    
    log.info("Best model is: ", best_model=best_model_num)
    log.info("The best model had this mean score: ", best_score=best_score)

    # For the best model: 
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain, train_new=False)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    x_train = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_train, model_path=f"model_{best_model_num}")
    model, _ = get_regression_model_svr(x_train, y_train)

    model.fit(x_train, y_train)

    log.info("Making prediction on x_test")
    x_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float)
    x_test_tensor = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_test_tensor, model_path=f"model_{best_model_num}")
    y_pred = model.predict(x_test_tensor)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred_df = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred_df.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")



