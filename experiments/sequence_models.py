import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec
from solidity_parser import parser
from tqdm import tqdm
import os

# --- Code from the previous Canvas: Solidity Code Embedder ---

def embed_solidity_code(source_code, vector_size=128, window=5, min_count=1, workers=4):
    """
    Embeds a Solidity source code string into a fixed-length vector.
    """
    try:
        ast = parser.parse(source_code, loc=False)
        tokens = []

        def traverse_ast(node):
            if isinstance(node, dict):
                if 'type' in node:
                    tokens.append(node['type'])
                for key, value in node.items():
                    if isinstance(value, str):
                        split_tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W)|[\d\W]+', value)
                        tokens.extend(t.lower() for t in split_tokens if t.isalnum())
                    elif isinstance(value, (dict, list)):
                        traverse_ast(value)
            elif isinstance(node, list):
                for item in node:
                    traverse_ast(item)

        traverse_ast(ast)

        if not tokens:
            return np.zeros(vector_size)

        model = Word2Vec(
            sentences=[tokens],
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1
        )

        embedding = np.zeros(vector_size)
        num_tokens = 0
        for token in tokens:
            if token in model.wv:
                embedding += model.wv[token]
                num_tokens += 1

        if num_tokens > 0:
            embedding /= num_tokens
        return embedding
    except Exception:
        return np.zeros(vector_size)

# --- PyTorch Dataset and Model ---

class SolidityBugDataset(Dataset):
    """
    PyTorch Dataset for loading Solidity contracts and their bug labels.
    """
    def __init__(self, data_file, base_path, benign_base_path, vector_size=128):
        self.data = []
        self.base_path = base_path
        self.benign_base_path = benign_base_path
        self.vector_size = vector_size
        with open(data_file, 'r') as f:
            # for line in f:
            #     self.data.append(json.loads(line))
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item['targets']
        if label == 0:
            contract_path = os.path.join(self.benign_base_path, item['contract_name'])
        else:
            contract_path = os.path.join(self.base_path, item['contract_name'])
        
        try:
            with open(contract_path, 'r') as f:
                source_code = f.read()
        except (FileNotFoundError, IOError) as e:
            print(f"Warning: Could not read file {contract_path}. Skipping. Error: {e}")
            # Return a zero vector and a placeholder label if file is not found
            return torch.zeros(self.vector_size), torch.tensor(-1.0, dtype=torch.float32)


        embedding = embed_solidity_code(source_code, vector_size=self.vector_size)
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class BugClassifierLSTM(nn.Module):
    """
    An LSTM model for classifying Solidity contracts.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_layers=2, drop_prob=0.5):
        super(BugClassifierLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # self.lstm = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.lstm = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # The input x is expected to be (batch_size, sequence_length, input_dim)
        # For our case, sequence_length is 1 as we feed the whole contract embedding at once.
        x = x.unsqueeze(1) # Reshape to (batch_size, 1, input_dim)
        lstm_out, _ = self.lstm(x)
        # We only need the output from the last time step
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

# --- Training and Evaluation Script ---

def train_model(model, train_loader, test_loader, epochs, criterion, optimizer, device):
    """
    Main training loop for the LSTM model.
    """
    model.to(device)
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            # Filter out placeholder labels
            valid_indices = labels != -1
            if not any(valid_indices):
                continue
            inputs, labels = inputs[valid_indices], labels[valid_indices]

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predicted = torch.round(torch.sigmoid(outputs))
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Evaluation loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Testing]"):
                # Filter out placeholder labels
                valid_indices = labels != -1
                if not any(valid_indices):
                    continue
                inputs, labels = inputs[valid_indices], labels[valid_indices]

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(torch.sigmoid(outputs))
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total
        test_loss /= test_total
        test_acc = test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        # Add F1 score and confusion matrix
        print(classification_report(labels.cpu(), predicted.cpu(), digits=4))
        print(confusion_matrix(labels.cpu(), predicted.cpu()))

    print("Training finished.")

def main():
    # --- Hyperparameters ---
    VECTOR_SIZE = 128
    HIDDEN_DIM = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 2048
    TRAIN_TEST_RATIO = 0.6

    # --- Setup ---
    # NOTE: You need to download the dataset and place it in a 'DAppSCAN-source' directory
    # or update the base_path to the correct location.
    # The JSONL file should be in the same directory as this script.
    # access_control | arithmetic | denial_of_service | front_running | reentrancy | time_manipulation | unchecked_low_level_calls
    bug_type = 'access_control'
    # DATA_FILE = f'experiments/benign_mixed/labels/graph_labels_{bug_type}.json'
    # BASE_PATH = 'experiments/DAppSCAN' # Assumes DAppSCAN-source is in the same directory
    # BENIGN_BASE_PATH = 'experiments/smartbugs-wild-clean-contracts'
    
    DATA_FILE = f'experiments/ge-sc-data/source_code/{bug_type}/mixed_buggy_curated_graph_labels.json'
    BASE_PATH = f'experiments/ge-sc-data/source_code/{bug_type}/cleaned_buggy_curated'
    BENIGN_BASE_PATH = 'experiments/smartbugs-wild-clean-contracts'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    dataset = SolidityBugDataset(data_file=DATA_FILE, base_path=BASE_PATH, benign_base_path=BENIGN_BASE_PATH, vector_size=VECTOR_SIZE)

    # Split dataset
    train_size = int(TRAIN_TEST_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples.")

    # --- Model, Loss, and Optimizer ---
    model = BugClassifierLSTM(input_dim=VECTOR_SIZE, hidden_dim=HIDDEN_DIM)
    criterion = nn.BCEWithLogitsLoss() # Good for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Run Training ---
    train_model(model, train_loader, test_loader, EPOCHS, criterion, optimizer, device)


if __name__ == '__main__':
    # To run this, you need to install the required libraries:
    # pip install torch numpy gensim solidity-parser tqdm
    main()
