import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from collections import OrderedDict
import warnings
from sklearn.model_selection import train_test_split

# --- PyTorch Geometric Check ---
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
except ImportError:
    raise ImportError("PyTorch Geometric is required for this model. Please install it with `pip install torch_geometric`.")

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. Specialized Modules
# ==============================================================================

class FourierFeatures(nn.Module):
    """
    Adds Fourier features for positional encoding of coordinate data.
    """
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        # Random projection matrix for Fourier features
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x has shape (batch_size, num_coord_features)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ==============================================================================
# 2. The Spatial GNN Model
# ==============================================================================

class SpatialGNN(nn.Module):
    """
    A Graph Neural Network that operates on a spatial graph.
    Node features are a combination of numerical, categorical, and coordinate features.
    """
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_mapping_size, gnn_hidden_dim, gnn_layers, mlp_hidden_sizes, output_size,
                 gnn_type='gat', gat_heads=4):
        super().__init__()
        
        # --- Feature Encoders ---
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        
        self.fourier_features = FourierFeatures(num_coord_features, fourier_mapping_size)
        fourier_output_dim = 2 * fourier_mapping_size
        
        gnn_input_dim = total_embedding_dim + num_numerical_features + fourier_output_dim
        
        # --- GNN Layers (Message Passing) ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.gnn_type = gnn_type
        
        # Initial linear projection for residual connections if dimensions differ
        self.initial_projection = nn.Linear(gnn_input_dim, gnn_hidden_dim)

        for i in range(gnn_layers):
            in_channels = gnn_hidden_dim
            if gnn_type == 'gat':
                self.convs.append(GATConv(in_channels, gnn_hidden_dim, heads=gat_heads, concat=False))
            else: # Default to GCN
                self.convs.append(GCNConv(in_channels, gnn_hidden_dim))
            self.bns.append(nn.BatchNorm1d(gnn_hidden_dim))
            
        # --- Final MLP Head ---
        mlp_input_dim = gnn_hidden_dim
        mlp_layers = []
        for size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(mlp_input_dim, size))
            mlp_layers.append(nn.ReLU())
            mlp_input_dim = size
        mlp_layers.append(nn.Linear(mlp_input_dim, output_size))
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x_cat, x_num, x_coord, edge_index):
        # 1. Encode all features to create initial node features
        cat_embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        fourier_encodings = self.fourier_features(x_coord)
        x = torch.cat(cat_embeddings + [x_num, fourier_encodings], dim=1)
        
        # Initial projection for residual connection
        x = self.initial_projection(x)

        # 2. Pass through GNN layers with residual connections
        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = x + identity # Add residual connection
            
        # 3. Pass final node representations through the MLP head
        return self.mlp_head(x)

# ==============================================================================
# 3. The Main User-Facing Wrapper Class
# ==============================================================================
"""
model = SpatialGNNRegressor(
    categorical_features: list[str],
    coord_features: list[str],
    graph_cat_features: list[str],
    k_neighbors: int = 8,
    output_size: int = 1,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    gnn_type: str = 'gat',
    gat_heads: int = 4,
    gnn_hidden_dim: int = 64,
    gnn_layers: int = 3,
    mlp_hidden_sizes: list[int] = [32, 16],
    loss_fn: str = 'huber',
    random_state: int | None = None
)

Args:
    categorical_features (list[str]): A list of column names to be treated as standard
                                      categorical features and converted into embeddings.
    coord_features (list[str]): A list of column names for the geographic coordinates
                                (e.g., ['loc_longitude', 'loc_latitude']). These are
                                used for building the spatial graph and for Fourier feature
                                encoding.
    graph_cat_features (list[str]): A list of key categorical features (e.g., neighborhood,
                                      school district) to be used alongside coordinates for
                                      building the hybrid k-NN graph. This creates a more
                                      realistic graph based on both proximity and administrative
                                      similarity.
    k_neighbors (int, optional): The number of nearest neighbors (the 'k' in k-NN) to use
                                 when constructing the spatial graph. Defaults to 8.
    output_size (int, optional): The number of output neurons, typically 1 for regression.
                                 Defaults to 1.
    batch_size (int, optional): The number of central nodes to process in each mini-batch
                                during training with the NeighborLoader. Defaults to 512.
    learning_rate (float, optional): The learning rate for the Adam optimizer.
                                     Defaults to 0.001.
    num_epochs (int, optional): The total number of passes through the training dataset.
                                Defaults to 100.
    gnn_type (str, optional): The type of graph convolution layer to use. Options are
                              'gat' (Graph Attention) or 'gcn' (Graph Convolutional Network).
                              Defaults to 'gat'.
    gat_heads (int, optional): The number of attention heads to use if `gnn_type` is 'gat'.
                               Defaults to 4.
    gnn_hidden_dim (int, optional): The dimensionality of the hidden representations
                                    within the GNN layers. Defaults to 64.
    gnn_layers (int, optional): The number of graph convolution (message-passing) layers
                                in the GNN. Defaults to 3.
    mlp_hidden_sizes (list[int], optional): A list of integers defining the size and number
                                            of hidden layers in the final MLP prediction head.
                                            Defaults to [32, 16].
    loss_fn (str, optional): The loss function to use for training. Options are 'huber'
                             (robust to outliers) or 'mse'. Defaults to 'huber'.
    random_state (int | None, optional): Seed for reproducibility. Defaults to None.
"""
class SpatialGNNRegressor:

    def __init__(self, categorical_features, coord_features, graph_cat_features, k_neighbors=8, output_size=1,
                 batch_size=512, learning_rate=0.001, num_epochs=100, gnn_type='gat', gat_heads=4,
                 gnn_hidden_dim=64, gnn_layers=3, mlp_hidden_sizes=[32, 16], loss_fn='huber',
                 random_state=None):
        self.categorical_features = categorical_features
        self.coord_features = coord_features
        self.graph_cat_features = graph_cat_features
        self.k_neighbors = k_neighbors
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gnn_type = gnn_type
        self.gat_heads = gat_heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layers = gnn_layers
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.loss_fn_name = loss_fn
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.graph_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.category_mappings = {}

    def _build_graph(self, X_fit):
        """Builds a hybrid k-NN graph and returns the edge_index."""
        print("Building hybrid k-NN graph...")
        # Scale coordinates
        coords = self.scaler.fit_transform(X_fit[self.coord_features])
        # One-hot encode categorical graph features
        graph_cats = self.graph_ohe.fit_transform(X_fit[self.graph_cat_features])
        
        # Combine features for k-NN
        graph_features = np.hstack([coords, graph_cats])
        
        nn = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='ball_tree')
        nn.fit(graph_features)
        _, indices = nn.kneighbors(graph_features)
        
        senders = np.repeat(np.arange(len(coords)), self.k_neighbors - 1)
        receivers = indices[:, 1:].flatten() # Exclude self-loops
        
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        print(f"Graph built with {edge_index.shape[1]} edges.")
        return edge_index

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        self.numerical_features = [col for col in X_train.columns if col not in self.categorical_features and col not in self.coord_features]
        
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # Preprocess data
        X_train_fit = X_train.copy()
        if self.numerical_features:
            self.scaler.fit(X_train_fit[self.numerical_features])
            X_train_fit[self.numerical_features] = self.scaler.transform(X_train_fit[self.numerical_features])
        
        for col in self.categorical_features:
            self.category_mappings[col] = {cat: i for i, cat in enumerate(X_train_fit[col].unique())}
        
        # Create PyG Data object for training
        edge_index = self._build_graph(X_train_fit)
        X_cat = torch.stack([torch.tensor(X_train_fit[col].map(self.category_mappings[col]).values, dtype=torch.long) for col in self.categorical_features], dim=1)
        X_num = torch.tensor(X_train_fit[self.numerical_features].values, dtype=torch.float32)
        X_coord = torch.tensor(X_train_fit[self.coord_features].values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        train_data = Data(x_cat=X_cat, x_num=X_num, x_coord=X_coord, y=y_tensor, edge_index=edge_index)

        # Create NeighborLoader for mini-batch training
        train_loader = NeighborLoader(train_data, num_neighbors=[self.k_neighbors] * self.gnn_layers, batch_size=self.batch_size, shuffle=True)
        
        embedding_specs = [(len(self.category_mappings[col]), min(50, (len(self.category_mappings[col])+1)//2)) for col in self.categorical_features]
        
        self.model = SpatialGNN(
            embedding_specs=embedding_specs, num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features), fourier_mapping_size=16,
            gnn_hidden_dim=self.gnn_hidden_dim, gnn_layers=self.gnn_layers,
            mlp_hidden_sizes=self.mlp_hidden_sizes, output_size=self.output_size,
            gnn_type=self.gnn_type, gat_heads=self.gat_heads
        ).to(device)
        
        criterion = nn.HuberLoss() if self.loss_fn == 'huber' else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        print("Starting Spatial GNN training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                y_pred = self.model(batch.x_cat, batch.x_num, batch.x_coord, batch.edge_index)
                loss = criterion(y_pred[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.batch_size
            
            avg_loss = total_loss / len(train_loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
            
            scheduler.step(avg_loss)

        print("Training finished.")

    def predict(self, X):
        if self.model is None: raise RuntimeError("You must call fit() before predicting.")
        X_pred = X.copy()
        
        if self.numerical_features:
            X_pred[self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])
        for col in self.categorical_features:
            X_pred[col] = X_pred[col].map(self.category_mappings[col]).fillna(0)
        
        edge_index = self._build_graph(X_pred)
        X_cat = torch.stack([torch.tensor(X_pred[col].values, dtype=torch.long) for col in self.categorical_features], dim=1)
        X_num = torch.tensor(X_pred[self.numerical_features].values, dtype=torch.float32)
        X_coord = torch.tensor(X_pred[self.coord_features].values, dtype=torch.float32)
        
        pred_data = Data(x_cat=X_cat, x_num=X_num, x_coord=X_coord, edge_index=edge_index)
        pred_loader = NeighborLoader(pred_data, num_neighbors=[-1] * self.gnn_layers, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch in pred_loader:
                batch = batch.to(device)
                y_pred = self.model(batch.x_cat, batch.x_num, batch.x_coord, batch.edge_index)
                all_predictions.append(y_pred[:batch.batch_size].cpu().numpy())
                
        return np.concatenate(all_predictions).flatten()

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy dataset that mimics the user's data structure
    n_samples = 1000
    data = {
        'meta_nbhd_code': [str(random.randint(1, 10)) for _ in range(n_samples)],
        'loc_school_elementary_district_geoid': [str(random.randint(1, 5)) for _ in range(n_samples)],
        'char_bldg_sf': np.random.uniform(800, 4000, n_samples),
        'char_beds': np.random.randint(1, 6, n_samples),
        'char_yrblt': np.random.randint(1900, 2023, n_samples),
        'loc_longitude': np.random.uniform(-87.9, -87.5, n_samples),
        'loc_latitude': np.random.uniform(41.6, 42.1, n_samples),
        'sale_price': np.random.uniform(100000, 800000, n_samples)
    }
    df = pd.DataFrame(data)
    for col in ['meta_nbhd_code', 'loc_school_elementary_district_geoid']:
        df[col] = df[col].astype('category')

    X = df.drop('sale_price', axis=1)
    y = df['sale_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define feature types for the model
    categorical_features = ['meta_nbhd_code', 'loc_school_elementary_district_geoid']
    coord_features = ['loc_longitude', 'loc_latitude']
    # Use these categorical features in addition to coordinates to build the graph
    graph_cat_features = ['meta_nbhd_code', 'loc_school_elementary_district_geoid']

    # Initialize and train the model
    gnn_model = SpatialGNNRegressor(
        categorical_features=categorical_features,
        coord_features=coord_features,
        graph_cat_features=graph_cat_features,
        k_neighbors=10,
        num_epochs=50, # Increased epochs for better convergence
        gnn_type='gat', # Use the more powerful GAT layer
        loss_fn='huber' # Use robust Huber loss
    )
    
    gnn_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = gnn_model.predict(X_test)
    print("\n--- Example Predictions ---")
    print(predictions[:5])
